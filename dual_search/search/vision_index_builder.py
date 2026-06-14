import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from typing import Optional

import faiss
import numpy as np
from tqdm import tqdm

try:
    from dual_search.search.vision_retrieval import (
        Qwen3VLImageEncoder,
        build_faiss_index,
        load_corpus,
        resolve_image_reference,
    )
except ModuleNotFoundError:
    from vision_retrieval import (
        Qwen3VLImageEncoder,
        build_faiss_index,
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
        self.embedding_dim = embedding_dim
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.truncate_dim = truncate_dim

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
        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Qwen3VLImageEncoder(
            model_path=self.model_path,
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
            truncate_dim=self.truncate_dim,
        )

    def _load_embedding(self) -> np.ndarray:
        embedding_dim = self.embedding_dim or self.encoder.get_embedding_dim()
        return np.memmap(
            self.embedding_path,
            mode="r",
            dtype=np.float32,
        ).reshape(len(self.corpus), embedding_dim)

    def _save_embedding(self, all_embeddings: np.ndarray):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype,
        )
        memmap[:] = all_embeddings
        memmap.flush()

    def write_meta(self, embeddings: np.ndarray):
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_path": self.corpus_path,
            "corpus_size": len(self.corpus),
            "model_path": self.model_path,
            "encoder": "Qwen3VLImageEncoder",
            "weight_prefix_repair": "auto_if_needed",
            "image_key": self.image_key,
            "embedding_dim": int(embeddings.shape[-1]),
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
        }
        with open(self.meta_save_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def encode_all(self) -> np.ndarray:
        all_embeddings = []
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
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0).astype(np.float32, order="C")

    def build_index(self):
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        if self.embedding_path is not None:
            all_embeddings = self._load_embedding()
        else:
            all_embeddings = self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)

        print("Creating image index")
        faiss_index = build_faiss_index(
            all_embeddings,
            faiss_type=self.faiss_type,
            faiss_gpu=self.faiss_gpu,
        )
        faiss.write_index(faiss_index, self.index_save_path)
        self.write_meta(all_embeddings)
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
