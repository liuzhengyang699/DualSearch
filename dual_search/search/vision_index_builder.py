import argparse
import json
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from tqdm import tqdm

from dual_search.data.fingerprints import (
    canonical_model_reference,
    corpus_fingerprint,
    stable_digest,
    validate_embedding_sidecar,
)

try:
    from dual_search.search.vision_retrieval import (
        Qwen3VLImageEncoder,
        load_corpus,
        resolve_image_reference,
    )
except ModuleNotFoundError:
    from vision_retrieval import (
        Qwen3VLImageEncoder,
        load_corpus,
        resolve_image_reference,
    )


class VisionIndexBuilder:
    def __init__(
        self,
        model_path: str,
        corpus_path: str,
        save_dir: str,
        retrieval_method: str = "qwen3_vl_embedding",
        image_key: str = "image",
        batch_size: int = 32,
        faiss_type: Optional[str] = None,
        embedding_path: Optional[str] = None,
        embedding_meta_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        save_embedding: bool = False,
        faiss_gpu: bool = False,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
    ):
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.retrieval_method = retrieval_method.lower()
        self.image_key = image_key
        self.batch_size = batch_size
        self.faiss_type = faiss_type if faiss_type is not None else "Flat"
        self.embedding_path = embedding_path
        self.embedding_meta_path = embedding_meta_path
        self.embedding_dim = embedding_dim
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.truncate_dim = truncate_dim

        if self.embedding_path is not None and self.save_embedding:
            raise ValueError(
                "save_embedding=true is only valid for newly generated embeddings; "
                "a precomputed embedding_path already owns its sidecar."
            )

        self.encoder_config = {
            "encoder": "Qwen3VLImageEncoder",
            "model_reference": canonical_model_reference(self.model_path),
            "normalize_embeddings": bool(self.normalize_embeddings),
            "truncate_dim": self.truncate_dim,
            "corpus_input_mode": "image_only",
            "query_input_mode": "image_text_joint",
        }

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        elif os.listdir(self.save_dir):
            warnings.warn(
                "Some files already exist in save dir and may be overwritten.",
                UserWarning,
            )

        self.index_save_path = os.path.join(
            self.save_dir,
            f"{self.retrieval_method}_{self.faiss_type}.index",
        )
        self.embedding_save_path = os.path.join(
            self.save_dir,
            f"emb_{self.retrieval_method}.memmap",
        )
        self.meta_save_path = os.path.join(self.save_dir, "vision_index_meta.json")
        self.embedding_meta_save_path = self.embedding_save_path + ".meta.json"
        self.corpus = load_corpus(self.corpus_path)
        if len(self.corpus) == 0:
            raise ValueError(f"Vision corpus is empty: {self.corpus_path}")
        self.corpus_fingerprint = corpus_fingerprint(
            self.corpus_path,
            id_keys=("id", "image_key"),
        )
        self.encoder = Qwen3VLImageEncoder(
            model_path=self.model_path,
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
            truncate_dim=self.truncate_dim,
        )

    def _load_embedding(self) -> np.ndarray:
        embedding_dim = self.embedding_dim or self.encoder.get_embedding_dim()
        sidecar_path = self.embedding_meta_path or f"{self.embedding_path}.meta.json"
        validate_embedding_sidecar(
            self.embedding_path,
            sidecar_path,
            self.corpus_path,
            expected_rows=len(self.corpus),
            expected_dim=embedding_dim,
            id_keys=("id", "image_key"),
            expected_encoder_config=self.encoder_config,
        )
        return np.memmap(
            self.embedding_path,
            mode="r",
            dtype=np.float32,
        ).reshape(len(self.corpus), embedding_dim)

    def _write_embedding_meta(self, embedding_dim: int):
        meta = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_file": os.path.basename(self.embedding_save_path),
            "row_count": len(self.corpus),
            "embedding_dim": int(embedding_dim),
            "dtype": "float32",
            "corpus_fingerprint": self.corpus_fingerprint,
            "encoder_config": self.encoder_config,
            "encoder_config_sha256": stable_digest(self.encoder_config),
        }
        Path(self.embedding_meta_save_path).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_meta(self, embedding_dim: int):
        meta = {
            "schema_version": 1,
            "index_kind": "vision",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_path": self.corpus_path,
            "corpus_size": len(self.corpus),
            "corpus_fingerprint": self.corpus_fingerprint,
            "encoder_config": self.encoder_config,
            "encoder_config_sha256": stable_digest(self.encoder_config),
            "model_path": self.model_path,
            "encoder": "Qwen3VLImageEncoder",
            "weight_prefix_repair": "auto_if_needed",
            "image_key": self.image_key,
            "embedding_dim": int(embedding_dim),
            "normalize_embeddings": self.normalize_embeddings,
            "truncate_dim": self.truncate_dim,
            "index": os.path.basename(self.index_save_path),
            "embedding": os.path.basename(self.embedding_save_path)
            if self.save_embedding
            else None,
            "faiss_type": self.faiss_type,
            "faiss_gpu": self.faiss_gpu,
            "batch_size": self.batch_size,
            "retrieval_method": self.retrieval_method,
            "corpus_embedding_mode": "image_only",
            "query_embedding_mode": "image_text_joint",
        }
        with open(self.meta_save_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def encode_to_memmap(self, path: str) -> np.memmap:
        embedding_dim = int(self.encoder.get_embedding_dim())
        all_embeddings = np.memmap(
            path,
            shape=(len(self.corpus), embedding_dim),
            mode="w+",
            dtype=np.float32,
        )
        for start_idx in tqdm(
            range(0, len(self.corpus), self.batch_size),
            desc="Inference image embeddings:",
        ):
            end_idx = min(start_idx + self.batch_size, len(self.corpus))
            batch = [self.corpus[idx] for idx in range(start_idx, end_idx)]
            image_list = [
                resolve_image_reference(item, image_key=self.image_key)
                for item in batch
            ]
            embeddings = self.encoder.encode(image_list)
            embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
            expected_shape = (end_idx - start_idx, embedding_dim)
            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"Vision encoder returned {embeddings.shape}, expected {expected_shape}."
                )
            all_embeddings[start_idx:end_idx] = embeddings
        all_embeddings.flush()
        return all_embeddings

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

    def build_index(self):
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        temp_index = self.index_save_path + ".tmp"
        generated_path = None
        try:
            if self.embedding_path is not None:
                all_embeddings = self._load_embedding()
            else:
                fd, generated_path = tempfile.mkstemp(
                    prefix=".vision-embeddings-",
                    suffix=".memmap",
                    dir=self.save_dir,
                )
                os.close(fd)
                all_embeddings = self.encode_to_memmap(generated_path)

            print("Creating image index incrementally")
            faiss_index = self._build_faiss_incrementally(all_embeddings)
            faiss.write_index(faiss_index, temp_index)
            os.replace(temp_index, self.index_save_path)
            embedding_dim = int(all_embeddings.shape[-1])
            if generated_path and self.save_embedding:
                del all_embeddings
                os.replace(generated_path, self.embedding_save_path)
                generated_path = None
                self._write_embedding_meta(embedding_dim)
            elif generated_path:
                del all_embeddings
                os.unlink(generated_path)
                generated_path = None
            self.write_meta(embedding_dim)
        finally:
            if generated_path and os.path.exists(generated_path):
                os.unlink(generated_path)
            if os.path.exists(temp_index):
                os.unlink(temp_index)
        print(f"Meta: {self.meta_save_path}")
        print("Finish!")


def main():
    parser = argparse.ArgumentParser(description="Creating a Qwen3-VL image index.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="indexes/")
    parser.add_argument("--retrieval_method", type=str, default="qwen3_vl_embedding")
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--faiss_type", type=str, default=None)
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--embedding_meta_path", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--truncate_dim", type=int, default=None)
    parser.add_argument("--no_normalize", action="store_true", default=False)

    args = parser.parse_args()

    index_builder = VisionIndexBuilder(
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        retrieval_method=args.retrieval_method,
        image_key=args.image_key,
        batch_size=args.batch_size,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        embedding_meta_path=args.embedding_meta_path,
        embedding_dim=args.embedding_dim,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        device=args.device,
        normalize_embeddings=not args.no_normalize,
        truncate_dim=args.truncate_dim,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
