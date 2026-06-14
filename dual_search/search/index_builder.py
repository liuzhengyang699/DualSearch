import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

try:
    from dual_search.search.text_retrieval import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BM25_NAME,
        DEFAULT_DENSE_INDEX_NAME,
        DEFAULT_EMBEDDING_NAME,
        DEFAULT_META_NAME,
        BGEM3DenseEncoder,
        RankBM25Index,
        build_faiss_index,
        load_jsonl_corpus,
    )
except ModuleNotFoundError:
    from text_retrieval import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BM25_NAME,
        DEFAULT_DENSE_INDEX_NAME,
        DEFAULT_EMBEDDING_NAME,
        DEFAULT_META_NAME,
        BGEM3DenseEncoder,
        RankBM25Index,
        build_faiss_index,
        load_jsonl_corpus,
    )


class HybridTextIndexBuilder:
    def __init__(
        self,
        corpus_path: str,
        save_dir: str,
        model_path: str = DEFAULT_BGE_M3_MODEL,
        batch_size: int = 16,
        max_length: int = 8192,
        device: str | None = None,
        use_fp16: bool = True,
        faiss_type: str = "Flat",
        faiss_gpu: bool = False,
        save_embedding: bool = False,
        embedding_path: str | None = None,
        embedding_dim: int = 1024,
    ):
        self.corpus_path = corpus_path
        self.save_dir = Path(save_dir)
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.use_fp16 = use_fp16
        self.faiss_type = faiss_type
        self.faiss_gpu = faiss_gpu
        self.save_embedding = save_embedding
        self.embedding_path = Path(embedding_path) if embedding_path else None
        self.embedding_dim = embedding_dim

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.save_dir / DEFAULT_DENSE_INDEX_NAME
        self.embedding_save_path = self.save_dir / DEFAULT_EMBEDDING_NAME
        self.bm25_path = self.save_dir / DEFAULT_BM25_NAME
        self.meta_path = self.save_dir / DEFAULT_META_NAME

        self.corpus = load_jsonl_corpus(corpus_path)

    def _load_embedding(self) -> np.ndarray:
        if self.embedding_path is None:
            raise ValueError("embedding_path is required when loading precomputed embeddings.")
        embeddings = np.memmap(
            self.embedding_path,
            mode="r",
            dtype=np.float32,
        ).reshape(len(self.corpus), self.embedding_dim)
        return np.asarray(embeddings, dtype=np.float32, order="C")

    def _save_embedding(self, embeddings: np.ndarray):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=embeddings.shape,
            mode="w+",
            dtype=np.float32,
        )
        memmap[:] = embeddings
        memmap.flush()

    def encode_corpus(self) -> np.ndarray:
        encoder = BGEM3DenseEncoder(
            model_path=self.model_path,
            device=self.device,
            use_fp16=self.use_fp16,
        )
        contents = [str(item["contents"]) for item in self.corpus]
        embeddings = encoder.encode(
            contents,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        return embeddings.astype(np.float32, order="C")

    def write_meta(self, embeddings: np.ndarray):
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_path": self.corpus_path,
            "corpus_size": len(self.corpus),
            "model_path": self.model_path,
            "embedding_dim": int(embeddings.shape[-1]),
            "max_length": self.max_length,
            "dense_index": self.index_path.name,
            "embedding": self.embedding_save_path.name if self.save_embedding else None,
            "bm25_index": self.bm25_path.name,
            "faiss_type": self.faiss_type,
            "bm25_tokenizer": "lower_regex_v1",
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def build(self):
        if self.embedding_path is not None:
            embeddings = self._load_embedding()
        else:
            embeddings = self.encode_corpus()
            if self.save_embedding:
                self._save_embedding(embeddings)

        print("Building FAISS dense index...")
        faiss_index = build_faiss_index(
            embeddings,
            faiss_type=self.faiss_type,
            faiss_gpu=self.faiss_gpu,
        )
        faiss.write_index(faiss_index, str(self.index_path))

        print("Building rank_bm25 sparse index...")
        bm25 = RankBM25Index.build(self.corpus)
        bm25.save(self.bm25_path)

        self.write_meta(embeddings)
        print(f"Dense index: {self.index_path}")
        print(f"BM25 index: {self.bm25_path}")
        print(f"Meta: {self.meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build EVQA BGE-M3 hybrid text retrieval indexes.")
    parser.add_argument("--corpus_path", type=str, required=True, help="EVQA text corpus JSONL.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory for dense, BM25, and meta files.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_BGE_M3_MODEL)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_fp16", action="store_false", dest="use_fp16")
    parser.set_defaults(use_fp16=True)
    parser.add_argument("--faiss_type", type=str, default="Flat")
    parser.add_argument("--faiss_gpu", action="store_true", default=False)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    args = parser.parse_args()

    builder = HybridTextIndexBuilder(
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        use_fp16=args.use_fp16,
        faiss_type=args.faiss_type,
        faiss_gpu=args.faiss_gpu,
        save_embedding=args.save_embedding,
        embedding_path=args.embedding_path,
        embedding_dim=args.embedding_dim,
    )
    builder.build()


if __name__ == "__main__":
    main()
