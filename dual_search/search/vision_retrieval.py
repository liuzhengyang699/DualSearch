import base64
import io
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import datasets
import faiss
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        num_proc=4,
    )
    return corpus


def resolve_image_reference(item: Any, image_key: str = "image") -> Any:
    if isinstance(item, dict):
        for key in (image_key, "image", "image_url", "image_path", "path"):
            image = item.get(key)
            if image:
                return image
    return item


def make_image_embedding_input(image: Any, image_key: str = "image") -> Dict[str, Any]:
    image = resolve_image_reference(image, image_key=image_key)
    if isinstance(image, dict):
        item = dict(image)
        if isinstance(item.get("image"), str):
            item["image"] = decode_image_data_url(item["image"]) or os.path.expanduser(item["image"])
        return item
    if isinstance(image, Image.Image):
        return {"image": image.convert("RGB")}
    if not isinstance(image, str):
        raise TypeError(f"Expected an image path string, got {type(image).__name__}")

    return {"image": decode_image_data_url(image) or os.path.expanduser(image)}


def decode_image_data_url(image: str) -> Optional[Image.Image]:
    if not image.startswith("data:image/"):
        return None
    _, encoded = image.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def normalize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = dict(doc)
    if doc.get("contents") is not None:
        return doc

    title = doc.get("title") or doc.get("id") or ""
    text = doc.get("caption") or doc.get("text") or doc.get("description") or ""
    if title:
        doc["contents"] = f"\"{title}\"\n{text}".rstrip()
    else:
        doc["contents"] = str(text)
    return doc


def load_docs(corpus, doc_idxs: Sequence[int]) -> List[Dict[str, Any]]:
    results = []
    for idx in doc_idxs:
        if int(idx) < 0:
            continue
        results.append(normalize_document(corpus[int(idx)]))
    return results


class Qwen3VLImageEncoder:
    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.truncate_dim = truncate_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hf_verbosity = hf_logging.get_verbosity()
        try:
            hf_logging.set_verbosity_error()
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
        finally:
            hf_logging.set_verbosity(hf_verbosity)
        self._repair_qwen3_vl_embedding_weights()
        self.model.eval()

    def _repair_qwen3_vl_embedding_weights(self) -> None:
        """Load local Qwen3-VL embedding weights into SentenceTransformer's Qwen3VLModel.

        The current SentenceTransformer/Transformers stack loads Qwen3-VL-Embedding-2B
        as a base Qwen3VLModel, while the checkpoint stores keys under the wrapper
        prefix "model.". Stripping that prefix avoids randomly initialized weights.
        """
        weight_path = Path(self.model_path) / "model.safetensors"
        if not weight_path.exists() or not hasattr(self.model[0], "auto_model"):
            return

        state_dict = load_file(str(weight_path), device="cpu")
        if not any(key.startswith("model.language_model.") for key in state_dict):
            return

        stripped = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        missing, unexpected = self.model[0].auto_model.load_state_dict(stripped, strict=False)
        if missing or unexpected:
            warnings.warn(
                "Qwen3-VL embedding weight prefix repair was incomplete: "
                f"missing={len(missing)}, unexpected={len(unexpected)}.",
                UserWarning,
            )

    def get_embedding_dim(self) -> int:
        if self.truncate_dim is not None:
            return self.truncate_dim
        return self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, image_list: Sequence[Any]) -> np.ndarray:
        if isinstance(image_list, str):
            image_list = [image_list]

        all_embeddings = []
        for start_idx in range(0, len(image_list), self.batch_size):
            batch = image_list[start_idx:start_idx + self.batch_size]
            images = [make_image_embedding_input(image) for image in batch]
            encode_kwargs = {
                "batch_size": len(images),
                "convert_to_numpy": True,
                "normalize_embeddings": self.normalize_embeddings,
                "show_progress_bar": False,
            }
            if self.truncate_dim is not None:
                encode_kwargs["truncate_dim"] = self.truncate_dim

            embeddings = self.model.encode(images, **encode_kwargs)
            embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            all_embeddings.append(embeddings)

            for image in images:
                raw_image = image.get("image")
                if isinstance(raw_image, Image.Image):
                    raw_image.close()
            torch.cuda.empty_cache()

        if not all_embeddings:
            return np.empty((0, self.get_embedding_dim()), dtype=np.float32)

        return np.concatenate(all_embeddings, axis=0).astype(np.float32, order="C")


def build_faiss_index(
    embeddings: np.ndarray,
    faiss_type: str = "Flat",
    faiss_gpu: bool = False,
):
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D embedding array.")

    dim = embeddings.shape[-1]
    faiss_index = faiss.index_factory(dim, faiss_type, faiss.METRIC_INNER_PRODUCT)

    if faiss_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
        if not faiss_index.is_trained:
            faiss_index.train(embeddings)
        faiss_index.add(embeddings)
        faiss_index = faiss.index_gpu_to_cpu(faiss_index)
    else:
        if not faiss_index.is_trained:
            faiss_index.train(embeddings)
        faiss_index.add(embeddings)

    return faiss_index
