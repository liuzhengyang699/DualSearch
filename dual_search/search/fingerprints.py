"""Stable fingerprints shared by standalone builders and retrieval runtime."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


FINGERPRINT_SCHEMA_VERSION = 1
MODEL_FINGERPRINT_SCHEMA_VERSION = 1
DEFAULT_ID_KEYS = ("id", "sample_id", "image_key")
_MODEL_FINGERPRINT_IGNORED_DIRECTORIES = {
    ".cache",
    ".git",
    "__pycache__",
}
_MODEL_FINGERPRINT_IGNORED_SUFFIXES = {
    ".lock",
    ".log",
    ".pyc",
    ".pyo",
    ".tmp",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: str | Path) -> dict[str, Any]:
    """Fingerprint one file or a complete artifact directory.

    Directory identity includes every relative path, byte size, and file
    digest in deterministic order.  Builders use this to detect index files
    changed in place after their cache manifest was published.
    """

    path = Path(path)
    if path.is_file():
        return {
            "kind": "file",
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact does not exist: {path}")

    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"Artifact directory is empty: {path}")
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "kind": "directory",
        "files_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def stable_digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def normalize_local_path(path: str | Path) -> str:
    """Return the canonical path spelling used for leak checks.

    ``strict=False`` is deliberate: catalog creation happens before images are
    necessarily extracted from their archives.
    """

    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    return os.path.normcase(str(Path(expanded).resolve(strict=False)))


def canonical_model_reference(value: str | Path) -> str:
    """Canonicalize a local model path while preserving remote model IDs."""

    text = str(value).strip()
    candidate = Path(os.path.expandvars(os.path.expanduser(text)))
    return normalize_local_path(candidate) if candidate.exists() else text


def model_fingerprint(value: str | Path) -> dict[str, Any]:
    """Fingerprint a model reference, including local checkpoint contents.

    A path alone is not a sufficient encoder identity: replacing weights in
    the same directory would otherwise leave an old index looking valid.  For
    local checkpoints we therefore hash every non-transient regular file in
    deterministic relative-path order. Python caches, lock files, and VCS
    metadata are excluded so merely loading a model cannot invalidate it.
    Remote model IDs remain representable for legacy runtime callers, but
    standalone builders require existing local directories before reaching
    this helper.
    """

    reference = canonical_model_reference(value)
    candidate = Path(os.path.expandvars(os.path.expanduser(str(value).strip())))
    if not candidate.is_dir():
        return {
            "model_fingerprint_schema_version": MODEL_FINGERPRINT_SCHEMA_VERSION,
            "kind": "reference",
            "reference": reference,
        }

    root = candidate.resolve()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not set(path.relative_to(root).parts).intersection(
                _MODEL_FINGERPRINT_IGNORED_DIRECTORIES
            )
            and path.suffix.casefold() not in _MODEL_FINGERPRINT_IGNORED_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"Local model directory is empty: {root}")

    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        file_size = file_path.stat().st_size
        file_sha256 = sha256_file(file_path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
        total_bytes += file_size

    return {
        "model_fingerprint_schema_version": MODEL_FINGERPRINT_SCHEMA_VERSION,
        "kind": "local_directory",
        "reference": normalize_local_path(root),
        "files_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def bge_m3_encoder_config(
    model_path: str | Path,
    *,
    max_length: int,
    use_fp16: bool,
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical BGE-M3 vector-space configuration."""

    return {
        "encoder": "BGEM3DenseEncoder",
        "model_reference": canonical_model_reference(model_path),
        "model_fingerprint": dict(fingerprint or model_fingerprint(model_path)),
        "normalize_embeddings": True,
        "max_length": int(max_length),
        "use_fp16": bool(use_fp16),
        "input_mode": "text",
    }


def qwen3_vl_encoder_config(
    model_path: str | Path,
    *,
    normalize_embeddings: bool,
    truncate_dim: int | None,
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical Qwen3-VL image/query vector-space configuration."""

    return {
        "encoder": "Qwen3VLImageEncoder",
        "model_reference": canonical_model_reference(model_path),
        "model_fingerprint": dict(fingerprint or model_fingerprint(model_path)),
        "normalize_embeddings": bool(normalize_embeddings),
        "truncate_dim": truncate_dim,
        "corpus_input_mode": "image_only",
        "query_input_mode": "image_text_joint",
    }


def assert_encoder_config(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    artifact: str,
) -> None:
    """Reject an index/embedding built in a different vector space."""

    actual = metadata.get("encoder_config")
    if not isinstance(actual, Mapping):
        raise ValueError(
            f"{artifact} metadata has no encoder_config; rebuild this legacy artifact."
        )
    actual_plain = dict(actual)
    expected_plain = dict(expected)
    actual_digest = metadata.get("encoder_config_sha256")
    if actual_digest != stable_digest(actual_plain):
        raise ValueError(f"{artifact} encoder_config fingerprint is missing or corrupt.")
    if canonical_json(actual_plain) != canonical_json(expected_plain):
        raise ValueError(
            f"{artifact} encoder configuration mismatch: expected {expected_plain!r}, "
            f"got {actual_plain!r}. Rebuild embeddings and indexes."
        )


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_no} in {path} is not an object.")
            yield value


def row_identity(row: Mapping[str, Any], id_keys: Sequence[str] = DEFAULT_ID_KEYS) -> str:
    for key in id_keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Corpus row has no stable identity in keys {tuple(id_keys)}: {row!r}")


def id_order_sha256(
    rows: Iterable[Mapping[str, Any]],
    id_keys: Sequence[str] = DEFAULT_ID_KEYS,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    seen: set[str] = set()
    for row in rows:
        identity = row_identity(row, id_keys=id_keys)
        if identity in seen:
            raise ValueError(f"Duplicate corpus identity: {identity}")
        seen.add(identity)
        digest.update(identity.encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def corpus_fingerprint(
    path: str | Path,
    *,
    id_keys: Sequence[str] = DEFAULT_ID_KEYS,
) -> dict[str, Any]:
    path = Path(path)
    order_digest, count = id_order_sha256(iter_jsonl(path), id_keys=id_keys)
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "sha256": sha256_file(path),
        "id_order_sha256": order_digest,
        "row_count": count,
    }


def assert_corpus_fingerprint(
    expected: Mapping[str, Any],
    corpus_path: str | Path,
    *,
    id_keys: Sequence[str] = DEFAULT_ID_KEYS,
) -> dict[str, Any]:
    actual = corpus_fingerprint(corpus_path, id_keys=id_keys)
    for key in ("sha256", "id_order_sha256", "row_count"):
        if key not in expected:
            raise ValueError(f"Index metadata is missing corpus fingerprint field {key!r}.")
        if str(expected[key]) != str(actual[key]):
            raise ValueError(
                f"Corpus fingerprint mismatch for {key}: expected {expected[key]!r}, "
                f"got {actual[key]!r}. Rebuild embeddings and indexes."
            )
    return actual


def load_and_validate_index_meta(
    meta_path: str | Path,
    corpus_path: str | Path,
    *,
    expected_kind: str | None = None,
    id_keys: Sequence[str] = DEFAULT_ID_KEYS,
    expected_encoder_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Required index metadata does not exist: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if expected_kind and meta.get("index_kind") != expected_kind:
        raise ValueError(
            f"Index metadata kind mismatch: expected {expected_kind!r}, "
            f"got {meta.get('index_kind')!r}."
        )
    expected = meta.get("corpus_fingerprint")
    if not isinstance(expected, dict):
        raise ValueError(f"Index metadata has no corpus_fingerprint object: {meta_path}")
    assert_corpus_fingerprint(expected, corpus_path, id_keys=id_keys)
    if expected_encoder_config is not None:
        assert_encoder_config(meta, expected_encoder_config, artifact="Index")
    return meta


def validate_embedding_sidecar(
    embedding_path: str | Path,
    sidecar_path: str | Path,
    corpus_path: str | Path,
    *,
    expected_rows: int,
    expected_dim: int,
    id_keys: Sequence[str] = DEFAULT_ID_KEYS,
    expected_encoder_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Precomputed embeddings require a fingerprint sidecar: {sidecar_path}"
        )
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = meta.get("corpus_fingerprint")
    if not isinstance(expected, dict):
        raise ValueError(f"Embedding sidecar has no corpus_fingerprint: {sidecar_path}")
    assert_corpus_fingerprint(expected, corpus_path, id_keys=id_keys)
    if expected_encoder_config is not None:
        assert_encoder_config(meta, expected_encoder_config, artifact="Embedding")
    if int(meta.get("row_count", -1)) != int(expected_rows):
        raise ValueError("Embedding row count does not match the current corpus.")
    if int(meta.get("embedding_dim", -1)) != int(expected_dim):
        raise ValueError("Embedding dimension does not match the requested dimension.")
    expected_bytes = int(expected_rows) * int(expected_dim) * 4
    actual_bytes = Path(embedding_path).stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Embedding file size mismatch: expected {expected_bytes} bytes, "
            f"got {actual_bytes}."
        )
    return meta
