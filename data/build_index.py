import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_search.search.fingerprints import (
    artifact_fingerprint,
    bge_m3_encoder_config,
    corpus_fingerprint,
    model_fingerprint,
    qwen3_vl_encoder_config,
    sha256_file,
    stable_digest,
    validate_embedding_sidecar,
)

INDEX_BUILD_SCHEMA_VERSION = 2
INDEX_CACHE_MANIFEST_NAME = "index_build_manifest.json"
DEFAULT_BM25_NAME = "bm25_rank_bm25.pkl"
DEFAULT_BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_DENSE_INDEX_NAME = "bge_m3_Flat.index"
DEFAULT_EMBEDDING_NAME = "emb_bge_m3.memmap"
DEFAULT_META_NAME = "text_index_meta.json"
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def load_jsonl_corpus(corpus_path: str | Path) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    with Path(corpus_path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or "contents" not in item:
                raise ValueError(
                    f"Text corpus row {line_no} must be an object with 'contents'."
                )
            corpus.append(item)
    if not corpus:
        raise ValueError(f"Text corpus is empty: {corpus_path}")
    return corpus


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
        model_fingerprint_value: Mapping[str, Any] | None = None,
        corpus_fingerprint_value: Mapping[str, Any] | None = None,
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

        self.encoder_config = bge_m3_encoder_config(
            self.model_path,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
            fingerprint=model_fingerprint_value,
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.save_dir / DEFAULT_DENSE_INDEX_NAME
        self.embedding_save_path = self.save_dir / DEFAULT_EMBEDDING_NAME
        self.bm25_path = self.save_dir / DEFAULT_BM25_NAME
        self.meta_path = self.save_dir / DEFAULT_META_NAME
        self.embedding_meta_save_path = Path(str(self.embedding_save_path) + ".meta.json")

        self.corpus = load_jsonl_corpus(corpus_path)
        self.corpus_fingerprint = dict(
            corpus_fingerprint_value
            or corpus_fingerprint(
                self.corpus_path,
                id_keys=("id",),
            )
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
        from dual_search.search.text_retrieval import BGEM3DenseEncoder

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
        import faiss

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
        import faiss
        from dual_search.search.text_retrieval import RankBM25Index

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


def load_vision_corpus(corpus_path: str | Path) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    with Path(corpus_path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(
                    f"Vision corpus row {line_no} is not a JSON object."
                )
            corpus.append(item)
    if not corpus:
        raise ValueError(f"Vision corpus is empty: {corpus_path}")
    return corpus


class VisionIndexBuilder:
    """Incremental Qwen3-VL image-only corpus index builder."""

    def __init__(
        self,
        model_path: str,
        corpus_path: str,
        save_dir: str,
        retrieval_method: str = "qwen3_vl_embedding",
        image_key: str = "image",
        batch_size: int = 32,
        faiss_type: str = "Flat",
        embedding_path: str | None = None,
        embedding_meta_path: str | None = None,
        embedding_dim: int | None = None,
        save_embedding: bool = False,
        faiss_gpu: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = True,
        truncate_dim: int | None = None,
        model_fingerprint_value: Mapping[str, Any] | None = None,
        corpus_fingerprint_value: Mapping[str, Any] | None = None,
    ):
        from dual_search.search.vision_retrieval import Qwen3VLImageEncoder

        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = Path(save_dir)
        self.retrieval_method = retrieval_method.lower()
        self.image_key = image_key
        self.batch_size = batch_size
        self.faiss_type = faiss_type
        self.embedding_path = Path(embedding_path) if embedding_path else None
        self.embedding_meta_path = (
            Path(embedding_meta_path) if embedding_meta_path else None
        )
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

        self.encoder_config = qwen3_vl_encoder_config(
            self.model_path,
            normalize_embeddings=self.normalize_embeddings,
            truncate_dim=self.truncate_dim,
            fingerprint=model_fingerprint_value,
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.index_save_path = (
            self.save_dir
            / f"{self.retrieval_method}_{self.faiss_type}.index"
        )
        self.embedding_save_path = (
            self.save_dir / f"emb_{self.retrieval_method}.memmap"
        )
        self.meta_save_path = self.save_dir / "vision_index_meta.json"
        self.embedding_meta_save_path = Path(
            str(self.embedding_save_path) + ".meta.json"
        )
        self.corpus = load_vision_corpus(self.corpus_path)
        self.corpus_fingerprint = dict(
            corpus_fingerprint_value
            or corpus_fingerprint(
                self.corpus_path,
                id_keys=("id", "image_key"),
            )
        )
        self.encoder = Qwen3VLImageEncoder(
            model_path=self.model_path,
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
            truncate_dim=self.truncate_dim,
        )

    def _load_embedding(self) -> np.ndarray:
        if self.embedding_path is None:
            raise ValueError(
                "embedding_path is required when loading precomputed embeddings."
            )
        embedding_dim = self.embedding_dim or self.encoder.get_embedding_dim()
        sidecar_path = self.embedding_meta_path or Path(
            str(self.embedding_path) + ".meta.json"
        )
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

    def _write_embedding_meta(self, embedding_dim: int) -> None:
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
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_meta(self, embedding_dim: int) -> None:
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
            "index": self.index_save_path.name,
            "embedding": (
                self.embedding_save_path.name if self.save_embedding else None
            ),
            "faiss_type": self.faiss_type,
            "faiss_gpu": self.faiss_gpu,
            "batch_size": self.batch_size,
            "retrieval_method": self.retrieval_method,
            "corpus_embedding_mode": "image_only",
            "query_embedding_mode": "image_text_joint",
        }
        self.meta_save_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def encode_to_memmap(self, path: str | Path) -> np.memmap:
        from dual_search.search.vision_retrieval import resolve_image_reference

        embedding_dim = int(self.encoder.get_embedding_dim())
        embeddings = np.memmap(
            path,
            shape=(len(self.corpus), embedding_dim),
            mode="w+",
            dtype=np.float32,
        )
        for start_idx in range(0, len(self.corpus), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.corpus))
            images = [
                resolve_image_reference(item, image_key=self.image_key)
                for item in self.corpus[start_idx:end_idx]
            ]
            batch = np.asarray(
                self.encoder.encode(images),
                dtype=np.float32,
                order="C",
            )
            expected_shape = (end_idx - start_idx, embedding_dim)
            if batch.shape != expected_shape:
                raise ValueError(
                    f"Vision encoder returned {batch.shape}, "
                    f"expected {expected_shape}."
                )
            embeddings[start_idx:end_idx] = batch
        embeddings.flush()
        return embeddings

    def _build_faiss_incrementally(self, embeddings: np.ndarray):
        import faiss

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Expected a non-empty 2D embedding array.")
        dim = int(embeddings.shape[1])
        index = faiss.index_factory(
            dim,
            self.faiss_type,
            faiss.METRIC_INNER_PRODUCT,
        )
        if not index.is_trained:
            training_count = min(len(embeddings), 100_000)
            positions = np.linspace(
                0,
                len(embeddings) - 1,
                training_count,
                dtype=np.int64,
            )
            training = np.asarray(
                embeddings[positions],
                dtype=np.float32,
                order="C",
            )
            index.train(training)
        if self.faiss_gpu:
            options = faiss.GpuMultipleClonerOptions()
            options.useFloat16 = True
            options.shard = True
            index = faiss.index_cpu_to_all_gpus(index, options)
        for start_idx in range(0, len(embeddings), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(embeddings))
            batch = np.asarray(
                embeddings[start_idx:end_idx],
                dtype=np.float32,
                order="C",
            )
            index.add(batch)
        if self.faiss_gpu:
            index = faiss.index_gpu_to_cpu(index)
        return index

    def build_index(self) -> None:
        import faiss

        temporary_index = Path(str(self.index_save_path) + ".tmp")
        generated_path: str | None = None
        try:
            if self.embedding_path is not None:
                embeddings = self._load_embedding()
            else:
                descriptor, generated_path = tempfile.mkstemp(
                    prefix=".vision-embeddings-",
                    suffix=".memmap",
                    dir=self.save_dir,
                )
                os.close(descriptor)
                embeddings = self.encode_to_memmap(generated_path)
            index = self._build_faiss_incrementally(embeddings)
            faiss.write_index(index, str(temporary_index))
            os.replace(temporary_index, self.index_save_path)
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
            temporary_index.unlink(missing_ok=True)


VISION_CONFIG_KEYS = {
    "model_path",
    "batch_size",
    "device",
    "faiss_type",
    "faiss_gpu",
    "save_embedding",
    "truncate_dim",
}
TEXT_CONFIG_KEYS = {
    "model_path",
    "batch_size",
    "device",
    "max_length",
    "faiss_type",
    "faiss_gpu",
    "save_embedding",
    "no_fp16",
}


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false.")
    return value


def _local_model_path(
    value: Any,
    *,
    label: str,
    config_dir: Path,
) -> str:
    text = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not text:
        raise ValueError(f"{label} is required.")
    path = Path(text)
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} must be an existing local model directory: {path}. "
            "Automatic model downloads are disabled."
        )
    return str(path)


def _load_index_config(
    config_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Config root must be a JSON object.")

    output_text = os.path.expandvars(
        os.path.expanduser(str(value.get("output_dir") or ""))
    )
    output_dir = Path(output_text)
    if not output_text or not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute path.")
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError(
            "output_dir must be outside the DualSearch repository checkout: "
            f"{output_dir}"
        )

    indexes = value.get("indexes")
    if not isinstance(indexes, Mapping):
        raise ValueError("indexes must be a JSON object.")
    unknown_index_keys = sorted(set(indexes).difference({"vision", "text"}))
    if unknown_index_keys:
        raise ValueError(f"Unsupported indexes config fields: {unknown_index_keys}")
    raw_vision = indexes.get("vision")
    raw_text = indexes.get("text")
    if not isinstance(raw_vision, Mapping) or not isinstance(raw_text, Mapping):
        raise ValueError("indexes.vision and indexes.text must be JSON objects.")
    unknown_vision = sorted(set(raw_vision).difference(VISION_CONFIG_KEYS))
    unknown_text = sorted(set(raw_text).difference(TEXT_CONFIG_KEYS))
    if unknown_vision:
        raise ValueError(
            f"Unsupported indexes.vision config fields: {unknown_vision}"
        )
    if unknown_text:
        raise ValueError(
            f"Unsupported indexes.text config fields: {unknown_text}"
        )

    vision = dict(raw_vision)
    text = dict(raw_text)
    vision["model_path"] = _local_model_path(
        vision.get("model_path"),
        label="indexes.vision.model_path",
        config_dir=path.parent,
    )
    text["model_path"] = _local_model_path(
        text.get("model_path"),
        label="indexes.text.model_path",
        config_dir=path.parent,
    )
    vision["batch_size"] = _positive_int(
        vision.get("batch_size", 32),
        label="indexes.vision.batch_size",
    )
    text["batch_size"] = _positive_int(
        text.get("batch_size", 16),
        label="indexes.text.batch_size",
    )
    text["max_length"] = _positive_int(
        text.get("max_length", 8192),
        label="indexes.text.max_length",
    )
    truncate_dim = vision.get("truncate_dim")
    if truncate_dim is not None:
        truncate_dim = _positive_int(
            truncate_dim,
            label="indexes.vision.truncate_dim",
        )
    vision["truncate_dim"] = truncate_dim
    for section_name, section in (("vision", vision), ("text", text)):
        for field in ("faiss_gpu", "save_embedding"):
            section[field] = _boolean(
                section.get(field, False),
                label=f"indexes.{section_name}.{field}",
            )
        faiss_type = str(section.get("faiss_type", "Flat") or "").strip()
        if not faiss_type:
            raise ValueError(
                f"indexes.{section_name}.faiss_type must be non-empty."
            )
        section["faiss_type"] = faiss_type
        device = section.get("device")
        if device is not None and not str(device).strip():
            raise ValueError(
                f"indexes.{section_name}.device must be null or non-empty."
            )
        section["device"] = str(device).strip() if device is not None else None
    text["no_fp16"] = _boolean(
        text.get("no_fp16", False),
        label="indexes.text.no_fp16",
    )
    return output_dir, vision, text


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _validate_manifest_corpus(
    *,
    manifest: Mapping[str, Any],
    corpus_path: Path,
    artifact_name: str,
    id_keys: Sequence[str],
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("RL build_manifest.json has no artifacts object.")
    expected = artifacts.get(artifact_name)
    if not isinstance(expected, Mapping):
        raise ValueError(
            f"RL build_manifest.json has no {artifact_name!r} artifact record."
        )
    actual = corpus_fingerprint(corpus_path, id_keys=id_keys)
    for key in ("sha256", "id_order_sha256", "row_count"):
        if key not in expected:
            raise ValueError(
                f"RL manifest artifact {artifact_name!r} is missing {key!r}."
            )
        if str(expected[key]) != str(actual[key]):
            raise ValueError(
                f"{artifact_name} does not belong to the published RL generation: "
                f"{key} expected {expected[key]!r}, got {actual[key]!r}."
            )
    return actual


def _validate_rl_generation(
    output_dir: Path,
    vision_corpus: Path,
    text_corpus: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "build_manifest.json"
    manifest = _load_json_object(manifest_path, label="RL build manifest")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("stage") != "fixed_11k_rl"
        or manifest.get("status") != "complete"
    ):
        raise ValueError(
            "build_manifest.json is not a completed fixed-11K RL generation; "
            "rerun data/build_rl.py."
        )
    build_fingerprint = manifest.get("build_fingerprint")
    if not isinstance(build_fingerprint, str) or not build_fingerprint:
        raise ValueError("RL build_manifest.json has no build_fingerprint.")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "build_fingerprint": build_fingerprint,
        "vision_corpus": _validate_manifest_corpus(
            manifest=manifest,
            corpus_path=vision_corpus,
            artifact_name="vision_corpus.jsonl",
            id_keys=("id", "image_key"),
        ),
        "text_corpus": _validate_manifest_corpus(
            manifest=manifest,
            corpus_path=text_corpus,
            artifact_name="text_corpus.jsonl",
            id_keys=("id",),
        ),
    }


def _reuse_status(
    cache_manifest_path: Path,
    *,
    build_fingerprint: str,
    outputs: Mapping[str, Path],
) -> tuple[bool, str]:
    if not cache_manifest_path.is_file():
        return False, "index cache manifest is missing"
    try:
        cache_manifest = _load_json_object(
            cache_manifest_path,
            label="index cache manifest",
        )
    except (OSError, ValueError) as exc:
        return False, str(exc)
    if (
        cache_manifest.get("schema_version") != INDEX_BUILD_SCHEMA_VERSION
        or cache_manifest.get("stage") != "index"
        or cache_manifest.get("status") != "complete"
    ):
        return False, "index cache manifest is incomplete or incompatible"
    if cache_manifest.get("build_fingerprint") != build_fingerprint:
        return False, "index input/model/config fingerprint changed"

    recorded_outputs = cache_manifest.get("outputs")
    if not isinstance(recorded_outputs, Mapping):
        return False, "index cache manifest has no output fingerprints"
    for name, path in outputs.items():
        record = recorded_outputs.get(name)
        if not isinstance(record, Mapping):
            return False, f"index cache manifest is missing output {name!r}"
        expected = record.get("fingerprint")
        if not isinstance(expected, Mapping):
            return False, f"index cache output {name!r} has no fingerprint"
        try:
            actual = artifact_fingerprint(path)
        except (OSError, ValueError) as exc:
            return False, str(exc)
        if stable_digest(dict(expected)) != stable_digest(actual):
            return False, f"index output {name!r} was modified"
    return True, "all input, model, and output fingerprints match"


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_generation(
    staged: Sequence[Path],
    final: Sequence[Path],
    *,
    backup_dir: Path,
) -> None:
    if len(staged) != len(final):
        raise ValueError("Staged/final artifact count mismatch.")
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, target in enumerate(final):
            if not target.exists():
                continue
            backup = backup_dir / f"{index:02d}-{target.name}"
            os.replace(target, backup)
            backups.append((target, backup))
        for source, target in zip(staged, final):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            _remove_path(target)
        for target, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DualSearch vision and hybrid text indexes."
    )
    parser.add_argument("--config", required=True, help="Shared DualSearch JSON config.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore a matching index cache manifest and rebuild.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir, vision, text = _load_index_config(args.config)
    vision_corpus = output_dir / "vision_corpus.jsonl"
    text_corpus = output_dir / "text_corpus.jsonl"
    for corpus in (vision_corpus, text_corpus):
        if not corpus.is_file():
            raise FileNotFoundError(f"Required corpus does not exist: {corpus}")

    index_root = output_dir / "indexes"
    vision_final = index_root / "vision"
    text_final = index_root / "text"
    report_final = output_dir / "index_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".build_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_final = cache_dir / INDEX_CACHE_MANIFEST_NAME

    rl_generation = _validate_rl_generation(
        output_dir,
        vision_corpus,
        text_corpus,
    )
    vision_model_fingerprint = model_fingerprint(vision["model_path"])
    text_model_fingerprint = model_fingerprint(text["model_path"])
    vision_encoder_config = qwen3_vl_encoder_config(
        vision["model_path"],
        normalize_embeddings=True,
        truncate_dim=vision["truncate_dim"],
        fingerprint=vision_model_fingerprint,
    )
    text_encoder_config = bge_m3_encoder_config(
        text["model_path"],
        max_length=text["max_length"],
        use_fp16=not text["no_fp16"],
        fingerprint=text_model_fingerprint,
    )
    build_inputs = {
        "schema_version": INDEX_BUILD_SCHEMA_VERSION,
        "builder": "dual_search_indexes",
        "rl_generation": rl_generation,
        "vision_encoder_config": vision_encoder_config,
        "text_encoder_config": text_encoder_config,
        "vision_index_config": vision,
        "text_index_config": text,
    }
    build_fingerprint = stable_digest(build_inputs)
    output_paths = {
        "vision": vision_final,
        "text": text_final,
        "report": report_final,
    }
    if not args.force:
        reusable, reason = _reuse_status(
            cache_manifest_final,
            build_fingerprint=build_fingerprint,
            outputs=output_paths,
        )
        if reusable:
            print(f"Reusing verified index generation: {reason}.", file=sys.stderr)
            report = _load_json_object(report_final, label="index report")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        print(f"Rebuilding index generation: {reason}.", file=sys.stderr)
    else:
        print("Rebuilding index generation because --force was specified.", file=sys.stderr)

    staging = Path(
        tempfile.mkdtemp(prefix=".index-stage-", dir=output_dir)
    )
    try:
        vision_stage = staging / "vision"
        text_stage = staging / "text"
        # The model references were resolved to existing local directories.
        # Offline flags ensure nested Transformers/SentenceTransformers code
        # cannot silently fall back to a Hub download.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        VisionIndexBuilder(
            model_path=vision["model_path"],
            corpus_path=str(vision_corpus),
            save_dir=str(vision_stage),
            batch_size=vision["batch_size"],
            faiss_type=vision["faiss_type"],
            save_embedding=vision["save_embedding"],
            faiss_gpu=vision["faiss_gpu"],
            device=vision["device"],
            truncate_dim=vision["truncate_dim"],
            model_fingerprint_value=vision_model_fingerprint,
            corpus_fingerprint_value=rl_generation["vision_corpus"],
        ).build_index()
        HybridTextIndexBuilder(
            corpus_path=str(text_corpus),
            save_dir=str(text_stage),
            model_path=text["model_path"],
            batch_size=text["batch_size"],
            max_length=text["max_length"],
            device=text["device"],
            use_fp16=not text["no_fp16"],
            faiss_type=text["faiss_type"],
            faiss_gpu=text["faiss_gpu"],
            save_embedding=text["save_embedding"],
            model_fingerprint_value=text_model_fingerprint,
            corpus_fingerprint_value=rl_generation["text_corpus"],
        ).build()

        report = {
            "schema_version": INDEX_BUILD_SCHEMA_VERSION,
            "stage": "index",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_fingerprint": build_fingerprint,
            "offline_models": True,
            "rl_generation": rl_generation,
            "vision": {
                "model_path": vision["model_path"],
                "corpus": str(vision_corpus),
                "corpus_fingerprint": rl_generation["vision_corpus"],
                "encoder_config": vision_encoder_config,
                "index_dir": str(vision_final),
            },
            "text": {
                "model_path": text["model_path"],
                "corpus": str(text_corpus),
                "corpus_fingerprint": rl_generation["text_corpus"],
                "encoder_config": text_encoder_config,
                "index_dir": str(text_final),
            },
        }
        report_stage = staging / "index_report.json"
        report_stage.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staged_outputs = {
            "vision": vision_stage,
            "text": text_stage,
            "report": report_stage,
        }
        cache_manifest = {
            "schema_version": INDEX_BUILD_SCHEMA_VERSION,
            "stage": "index",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_fingerprint": build_fingerprint,
            "inputs": build_inputs,
            "outputs": {
                name: {
                    "path": str(output_paths[name]),
                    "fingerprint": artifact_fingerprint(path),
                }
                for name, path in staged_outputs.items()
            },
        }
        cache_manifest_stage = staging / INDEX_CACHE_MANIFEST_NAME
        cache_manifest_stage.write_text(
            json.dumps(
                cache_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _publish_generation(
            (
                vision_stage,
                text_stage,
                report_stage,
                cache_manifest_stage,
            ),
            (
                vision_final,
                text_final,
                report_final,
                cache_manifest_final,
            ),
            backup_dir=staging / "backups",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
