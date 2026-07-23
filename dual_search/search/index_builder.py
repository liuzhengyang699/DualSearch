import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from dual_search.data.fingerprints import (
    canonical_model_reference,
    corpus_fingerprint,
    stable_digest,
    validate_embedding_sidecar,
)

try:
    from dual_search.search.text_retrieval import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BM25_NAME,
        DEFAULT_DENSE_INDEX_NAME,
        DEFAULT_EMBEDDING_NAME,
        DEFAULT_META_NAME,
        BGEM3DenseEncoder,
        RankBM25Index,
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
        embedding_meta_path: str | None = None,
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
        self.embedding_meta_path = Path(embedding_meta_path) if embedding_meta_path else None
        self.embedding_dim = embedding_dim

        if self.embedding_path is not None and self.save_embedding:
            raise ValueError(
                "save_embedding=true is only valid for newly generated embeddings; "
                "a precomputed embedding_path already owns its sidecar."
            )

        self.encoder_config = {
            "encoder": "BGEM3DenseEncoder",
            "model_reference": canonical_model_reference(self.model_path),
            "normalize_embeddings": True,
            "max_length": int(self.max_length),
            "use_fp16": bool(self.use_fp16),
            "input_mode": "text",
        }

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.save_dir / DEFAULT_DENSE_INDEX_NAME
        self.embedding_save_path = self.save_dir / DEFAULT_EMBEDDING_NAME
        self.bm25_path = self.save_dir / DEFAULT_BM25_NAME
        self.meta_path = self.save_dir / DEFAULT_META_NAME
        self.embedding_meta_save_path = Path(str(self.embedding_save_path) + ".meta.json")

        self.corpus = load_jsonl_corpus(corpus_path)
        self.corpus_fingerprint = corpus_fingerprint(
            self.corpus_path,
            id_keys=("id",),
        )

    def _load_embedding(self) -> np.ndarray:
        if self.embedding_path is None:
            raise ValueError("embedding_path is required when loading precomputed embeddings.")
        sidecar_path = self.embedding_meta_path or Path(str(self.embedding_path) + ".meta.json")
        validate_embedding_sidecar(
            self.embedding_path,
            sidecar_path,
            self.corpus_path,
            expected_rows=len(self.corpus),
            expected_dim=self.embedding_dim,
            id_keys=("id",),
            expected_encoder_config=self.encoder_config,
        )
        embeddings = np.memmap(
            self.embedding_path,
            mode="r",
            dtype=np.float32,
        ).reshape(len(self.corpus), self.embedding_dim)
        return embeddings

    def encode_corpus_to_memmap(self, path: str | Path) -> np.memmap:
        encoder = BGEM3DenseEncoder(
            model_path=self.model_path,
            device=self.device,
            use_fp16=self.use_fp16,
        )
        if not self.corpus:
            raise ValueError("Text corpus is empty.")
        first_end = min(self.batch_size, len(self.corpus))
        first_embeddings = encoder.encode(
            [str(item["contents"]) for item in self.corpus[:first_end]],
            batch_size=self.batch_size,
            max_length=self.max_length,
        ).astype(np.float32, order="C")
        inferred_dim = int(first_embeddings.shape[-1])
        if self.embedding_dim and inferred_dim != self.embedding_dim:
            raise ValueError(
                f"Text encoder dimension {inferred_dim} does not match configured "
                f"embedding_dim {self.embedding_dim}."
            )
        self.embedding_dim = inferred_dim
        embeddings = np.memmap(
            path,
            shape=(len(self.corpus), inferred_dim),
            mode="w+",
            dtype=np.float32,
        )
        embeddings[:first_end] = first_embeddings
        for start_idx in range(first_end, len(self.corpus), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.corpus))
            batch = encoder.encode(
                [str(item["contents"]) for item in self.corpus[start_idx:end_idx]],
                batch_size=self.batch_size,
                max_length=self.max_length,
            ).astype(np.float32, order="C")
            expected_shape = (end_idx - start_idx, inferred_dim)
            if batch.shape != expected_shape:
                raise ValueError(
                    f"Text encoder returned {batch.shape}, expected {expected_shape}."
                )
            embeddings[start_idx:end_idx] = batch
        embeddings.flush()
        return embeddings

    def _build_faiss_incrementally(self, embeddings: np.ndarray):
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Expected a non-empty 2D embedding array.")
        dim = int(embeddings.shape[1])
        index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            training_count = min(len(embeddings), 100_000)
            positions = np.linspace(0, len(embeddings) - 1, training_count, dtype=np.int64)
            training = np.asarray(embeddings[positions], dtype=np.float32, order="C")
            index.train(training)
        if self.faiss_gpu:
            options = faiss.GpuMultipleClonerOptions()
            options.useFloat16 = True
            options.shard = True
            index = faiss.index_cpu_to_all_gpus(index, options)
        for start_idx in range(0, len(embeddings), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(embeddings))
            batch = np.asarray(embeddings[start_idx:end_idx], dtype=np.float32, order="C")
            index.add(batch)
        if self.faiss_gpu:
            index = faiss.index_gpu_to_cpu(index)
        return index

    def _write_embedding_meta(self, embedding_dim: int):
        meta = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_file": self.embedding_save_path.name,
            "row_count": len(self.corpus),
            "embedding_dim": int(embedding_dim),
            "dtype": "float32",
            "corpus_fingerprint": self.corpus_fingerprint,
            "encoder_config": self.encoder_config,
            "encoder_config_sha256": stable_digest(self.encoder_config),
        }
        self.embedding_meta_save_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def write_meta(self, embedding_dim: int):
        meta = {
            "schema_version": 1,
            "index_kind": "text",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_path": self.corpus_path,
            "corpus_size": len(self.corpus),
            "corpus_fingerprint": self.corpus_fingerprint,
            "encoder_config": self.encoder_config,
            "encoder_config_sha256": stable_digest(self.encoder_config),
            "model_path": self.model_path,
            "embedding_dim": int(embedding_dim),
            "max_length": self.max_length,
            "dense_index": self.index_path.name,
            "embedding": self.embedding_save_path.name if self.save_embedding else None,
            "bm25_index": self.bm25_path.name,
            "faiss_type": self.faiss_type,
            "bm25_tokenizer": "lower_regex_v1",
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def build(self):
        temp_index = Path(str(self.index_path) + ".tmp")
        generated_path = None
        try:
            if self.embedding_path is not None:
                embeddings = self._load_embedding()
            else:
                fd, generated_path = tempfile.mkstemp(
                    prefix=".text-embeddings-",
                    suffix=".memmap",
                    dir=self.save_dir,
                )
                os.close(fd)
                embeddings = self.encode_corpus_to_memmap(generated_path)

            print("Building FAISS dense index incrementally...")
            faiss_index = self._build_faiss_incrementally(embeddings)
            faiss.write_index(faiss_index, str(temp_index))
            os.replace(temp_index, self.index_path)

            print("Building rank_bm25 sparse index...")
            bm25 = RankBM25Index.build(self.corpus)
            bm25.save(self.bm25_path)

            embedding_dim = int(embeddings.shape[-1])
            if generated_path and self.save_embedding:
                del embeddings
                os.replace(generated_path, self.embedding_save_path)
                generated_path = None
                self._write_embedding_meta(embedding_dim)
            elif generated_path:
                del embeddings
                os.unlink(generated_path)
                generated_path = None

            self.write_meta(embedding_dim)
        finally:
            if generated_path and os.path.exists(generated_path):
                os.unlink(generated_path)
            if temp_index.exists():
                temp_index.unlink()
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
    parser.add_argument("--embedding_meta_path", type=str, default=None)
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
        embedding_meta_path=args.embedding_meta_path,
        embedding_dim=args.embedding_dim,
    )
    builder.build()


if __name__ == "__main__":
    main()
