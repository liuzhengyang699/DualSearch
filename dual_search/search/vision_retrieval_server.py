import argparse
import warnings
from typing import Any, List, Optional

import faiss
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm

try:
    from dual_search.search.vision_retrieval import (
        Qwen3VLImageEncoder,
        load_corpus,
        load_docs,
    )
except ModuleNotFoundError:
    from vision_retrieval import (
        Qwen3VLImageEncoder,
        load_corpus,
        load_docs,
    )


class VisionSearchQuery(BaseModel):
    sample_index: Optional[int] = None
    query: Optional[str] = None
    image_index: Optional[int] = None
    image: Any
    images: Optional[List[Any]] = None


class VisionQueryRequest(BaseModel):
    queries: List[VisionSearchQuery]
    topk: Optional[int] = None
    return_scores: bool = False


class Config:
    def __init__(
        self,
        retrieval_topk: int = 3,
        index_path: str = "./index/qwen3_vl_embedding_Flat.index",
        corpus_path: str = "./data/vision_corpus.jsonl",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        retrieval_batch_size: int = 32,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
    ):
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_batch_size = retrieval_batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.truncate_dim = truncate_dim


class VisionDenseRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        self.index = faiss.read_index(config.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(config.corpus_path)
        self.encoder = Qwen3VLImageEncoder(
            model_path=config.retrieval_model_path,
            batch_size=config.retrieval_batch_size,
            device=config.device,
            normalize_embeddings=config.normalize_embeddings,
            truncate_dim=config.truncate_dim,
        )

    def _search(self, query: VisionSearchQuery, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode([query.image])
        scores, idxs = self.index.search(query_emb, k=num)
        docs, item_scores = self._load_scored_docs(idxs[0], scores[0])
        if return_score:
            return docs, item_scores
        return docs

    def _batch_search(
        self,
        query_list: List[VisionSearchQuery],
        num: int = None,
        return_score: bool = False,
    ):
        if num is None:
            num = self.topk
        if not query_list:
            if return_score:
                return [], []
            return []

        results = []
        scores = []
        for start_idx in tqdm(
            range(0, len(query_list), self.batch_size),
            desc="Vision retrieval process: ",
        ):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            image_batch = [query.image for query in query_batch]
            batch_emb = self.encoder.encode(image_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)

            for item_idxs, item_scores in zip(batch_idxs, batch_scores):
                item_docs, valid_scores = self._load_scored_docs(item_idxs, item_scores)
                results.append(item_docs)
                scores.append(valid_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, image_batch
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        return results

    def _load_scored_docs(self, idxs, scores):
        valid_pairs = [
            (int(idx), float(score))
            for idx, score in zip(idxs, scores)
            if int(idx) >= 0
        ]
        if len(valid_pairs) < len(idxs):
            warnings.warn("FAISS returned invalid document ids.", UserWarning)
        if not valid_pairs:
            return [], []

        valid_idxs = [idx for idx, _ in valid_pairs]
        valid_scores = [score for _, score in valid_pairs]
        docs = load_docs(self.corpus, valid_idxs)
        return docs, valid_scores

    def search(self, query: VisionSearchQuery, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(
        self,
        query_list: List[VisionSearchQuery],
        num: int = None,
        return_score: bool = False,
    ):
        return self._batch_search(query_list, num, return_score)


app = FastAPI()


@app.post("/vision_retrieve")
def vision_retrieve_endpoint(request: VisionQueryRequest):
    """
    Endpoint that accepts agent vision_search requests and performs image retrieval.
    The free-text query field is metadata only; retrieval uses queries[*].image.
    """
    topk = request.topk or config.retrieval_topk
    if request.return_scores:
        results, scores = retriever.batch_search(
            query_list=request.queries,
            num=topk,
            return_score=True,
        )
    else:
        results = retriever.batch_search(
            query_list=request.queries,
            num=topk,
            return_score=False,
        )
        scores = None

    response = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            response.append(combined)
        else:
            response.append(single_result)
    return {"result": response}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local Qwen3-VL image retriever.")
    parser.add_argument("--index_path", type=str, default="./index/qwen3_vl_embedding_Flat.index", help="Image corpus FAISS index file.")
    parser.add_argument("--corpus_path", type=str, default="./data/vision_corpus.jsonl", help="Local image corpus JSONL file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved images for one query image.")
    parser.add_argument("--retriever_model", type=str, default="Qwen/Qwen3-VL-Embedding-2B", help="Path or HF id of the image embedding model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for image embedding.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--truncate_dim", type=int, default=None, help="Optional embedding dimension truncation.")
    parser.add_argument("--no_normalize", action="store_true", default=False, help="Disable L2 normalization for embeddings.")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU FAISS index.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    config = Config(
        retrieval_topk=args.topk,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=not args.no_normalize,
        truncate_dim=args.truncate_dim,
    )
    retriever = VisionDenseRetriever(config)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
