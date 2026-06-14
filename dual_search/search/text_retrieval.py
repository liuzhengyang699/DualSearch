import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np


DEFAULT_BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_DENSE_INDEX_NAME = "bge_m3_Flat.index"
DEFAULT_EMBEDDING_NAME = "emb_bge_m3.memmap"
DEFAULT_BM25_NAME = "bm25_rank_bm25.pkl"
DEFAULT_META_NAME = "text_index_meta.json"
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def load_jsonl_corpus(corpus_path: str | Path) -> list[dict[str, Any]]:
    corpus = []
    with Path(corpus_path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "contents" not in item:
                raise ValueError(f"Corpus row {line_no} is missing required 'contents'.")
            corpus.append(item)
    if not corpus:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    return corpus


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embeddings / norms).astype(np.float32, order="C")


class BGEM3DenseEncoder:
    def __init__(
        self,
        model_path: str = DEFAULT_BGE_M3_MODEL,
        device: str | None = None,
        use_fp16: bool = True,
        normalize: bool = True,
    ):
        from FlagEmbedding import BGEM3FlagModel

        self.model_path = model_path
        self.device = device
        self.use_fp16 = use_fp16
        self.normalize = normalize

        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if device:
            kwargs["devices"] = [device]
        try:
            self.model = BGEM3FlagModel(model_path, **kwargs)
        except TypeError:
            kwargs.pop("devices", None)
            self.model = BGEM3FlagModel(model_path, **kwargs)

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 16,
        max_length: int = 8192,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        try:
            output = self.model.encode(
                list(texts),
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except TypeError:
            output = self.model.encode(list(texts), batch_size=batch_size, max_length=max_length)

        embeddings = output["dense_vecs"] if isinstance(output, dict) else output
        embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if self.normalize:
            embeddings = normalize_embeddings(embeddings)
        return embeddings


class RankBM25Index:
    def __init__(self, bm25: Any, corpus_size: int):
        self.bm25 = bm25
        self.corpus_size = corpus_size

    @classmethod
    def build(cls, corpus: Sequence[dict[str, Any]]) -> "RankBM25Index":
        from rank_bm25 import BM25Okapi

        tokenized = [tokenize_for_bm25(str(item.get("contents", ""))) for item in corpus]
        return cls(BM25Okapi(tokenized), corpus_size=len(corpus))

    @classmethod
    def load(cls, path: str | Path) -> "RankBM25Index":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        return cls(payload["bm25"], int(payload["corpus_size"]))

    def save(self, path: str | Path):
        payload = {
            "bm25": self.bm25,
            "corpus_size": self.corpus_size,
            "tokenizer": "lower_regex_v1",
        }
        with Path(path).open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def search(self, query: str, topk: int) -> tuple[list[int], list[float]]:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens or topk <= 0:
            return [], []
        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        if scores.size == 0:
            return [], []
        k = min(topk, scores.size)
        candidate_idxs = np.argpartition(-scores, k - 1)[:k]
        sorted_idxs = candidate_idxs[np.argsort(-scores[candidate_idxs])]
        return sorted_idxs.astype(int).tolist(), scores[sorted_idxs].astype(float).tolist()


def build_faiss_index(embeddings: np.ndarray, faiss_type: str = "Flat", faiss_gpu: bool = False):
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D embedding array.")

    dim = embeddings.shape[-1]
    index = faiss.index_factory(dim, faiss_type, faiss.METRIC_INNER_PRODUCT)
    if faiss_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co)
        if not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)
        return faiss.index_gpu_to_cpu(index)

    if not index.is_trained:
        index.train(embeddings)
    index.add(embeddings)
    return index


def rrf_fuse(
    ranked_lists: Sequence[Sequence[int]],
    rrf_k: int = 60,
    topk: int | None = None,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked, start=1):
            if doc_idx < 0:
                continue
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return fused if topk is None else fused[:topk]


class BGEReranker:
    def __init__(
        self,
        model_path: str = DEFAULT_BGE_RERANKER_MODEL,
        device: str | None = None,
        use_fp16: bool = True,
        normalize: bool = True,
    ):
        from FlagEmbedding import FlagReranker

        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if device:
            kwargs["devices"] = [device]
        try:
            self.model = FlagReranker(model_path, **kwargs)
        except TypeError:
            kwargs.pop("devices", None)
            self.model = FlagReranker(model_path, **kwargs)
        self.normalize = normalize

    def score(
        self,
        query: str,
        passages: Sequence[str],
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> list[float]:
        if not passages:
            return []
        pairs = [[query, passage] for passage in passages]
        try:
            scores = self.model.compute_score(
                pairs,
                batch_size=batch_size,
                max_length=max_length,
                normalize=self.normalize,
            )
        except TypeError:
            scores = self.model.compute_score(pairs, normalize=self.normalize)
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]


@dataclass
class HybridRetrieverConfig:
    index_path: str
    bm25_path: str
    corpus_path: str
    dense_model_path: str = DEFAULT_BGE_M3_MODEL
    reranker_model_path: str = DEFAULT_BGE_RERANKER_MODEL
    device: str | None = None
    faiss_gpu: bool = False
    dense_topk: int = 50
    bm25_topk: int = 50
    hybrid_topk: int = 100
    rrf_k: int = 60
    retrieval_topk: int = 3
    dense_batch_size: int = 16
    reranker_batch_size: int = 16
    dense_max_length: int = 8192
    reranker_max_length: int = 1024
    use_fp16: bool = True


class HybridRetriever:
    def __init__(self, config: HybridRetrieverConfig):
        self.config = config
        self.corpus = load_jsonl_corpus(config.corpus_path)
        self.index = faiss.read_index(config.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        self.bm25 = RankBM25Index.load(config.bm25_path)
        self.encoder = BGEM3DenseEncoder(
            model_path=config.dense_model_path,
            device=config.device,
            use_fp16=config.use_fp16,
        )
        self.reranker = BGEReranker(
            model_path=config.reranker_model_path,
            device=config.device,
            use_fp16=config.use_fp16,
            normalize=True,
        )

    def _dense_search(self, query: str, topk: int) -> list[int]:
        query_emb = self.encoder.encode(
            query,
            batch_size=1,
            max_length=self.config.dense_max_length,
        )
        _, idxs = self.index.search(query_emb, k=min(topk, len(self.corpus)))
        return [int(idx) for idx in idxs[0] if int(idx) >= 0]

    def search(self, query: str, topk: int | None = None, return_score: bool = False):
        topk = topk or self.config.retrieval_topk
        dense_idxs = self._dense_search(query, self.config.dense_topk)
        bm25_idxs, _ = self.bm25.search(query, self.config.bm25_topk)
        fused = rrf_fuse(
            [dense_idxs, bm25_idxs],
            rrf_k=self.config.rrf_k,
            topk=self.config.hybrid_topk,
        )
        candidate_idxs = [idx for idx, _ in fused]
        candidates = [self.corpus[idx] for idx in candidate_idxs]
        passages = [str(doc.get("contents", "")) for doc in candidates]
        rerank_scores = self.reranker.score(
            query,
            passages,
            batch_size=self.config.reranker_batch_size,
            max_length=self.config.reranker_max_length,
        )
        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda item: item[1],
            reverse=True,
        )[:topk]

        docs = [doc for doc, _ in ranked]
        scores = [score for _, score in ranked]
        return (docs, scores) if return_score else docs

    def batch_search(self, queries: Sequence[str], topk: int | None = None, return_score: bool = False):
        all_docs = []
        all_scores = []
        for query in queries:
            docs, scores = self.search(query, topk=topk, return_score=True)
            all_docs.append(docs)
            all_scores.append(scores)
        return (all_docs, all_scores) if return_score else all_docs
