"""Leak-free EVQA/iNaturalist dataset construction pipeline.

The pipeline is intentionally local-only: every input path is supplied by a
JSON config and no stage contains download logic.  Large iNaturalist metadata
is streamed into SQLite, which is then used as the canonical image catalog.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, TextIO

from dual_search.data.fingerprints import (
    canonical_json,
    corpus_fingerprint,
    normalize_local_path,
    sha256_file,
    stable_digest,
)


PIPELINE_SCHEMA_VERSION = 2
SUPPORTED_DATASET = "inaturalist"
GLDV2_NAMES = {"gldv2", "google_landmarks", "google_landmarks_v2", "landmarks"}
DEFAULT_PROMPT_TEMPLATE = (
    "{image_block}\n"
    "Images are numbered from 1 in the order shown above.\n"
    "Answer the question about the images. You may use the available "
    "vision_search and search tools when external evidence is needed, but call "
    "at most one tool in each assistant turn. Reason "
    "inside <think>...</think> and return the final answer inside "
    "<answer>...</answer>.\nQuestion: {question}"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: Any) -> str:
    value = clean_text(url)
    if value.startswith("http://"):
        value = "https://" + value[len("http://") :]
    return value.rstrip("/")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def split_delimited(value: Any, delimiter: str = "|") -> list[str]:
    return [part.strip() for part in clean_text(value).split(delimiter) if part.strip()]


def deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def parse_answers(row: Mapping[str, Any]) -> list[str]:
    # EVQA uses ``|`` for acceptable answer variants.  ``&&`` joins the
    # required components of a multi-answer target and must remain intact;
    # splitting it would incorrectly reward a partially correct response.
    answers = split_delimited(row.get("answer"), "|")
    multi_answer = clean_text(row.get("multi_answer"))
    if multi_answer and multi_answer.lower() not in {"0", "1", "false", "true"}:
        answers.extend(split_delimited(multi_answer, "|"))
    return deduplicate(" && ".join(split_delimited(answer, "&&")) for answer in answers)


def row_wiki_pairs(row: Mapping[str, Any]) -> list[dict[str, str]]:
    titles = split_delimited(row.get("wikipedia_title"))
    urls = split_delimited(row.get("wikipedia_url"))
    pairs: list[dict[str, str]] = []
    for index, url in enumerate(urls):
        title = titles[index] if index < len(titles) else (titles[0] if titles else "")
        pairs.append({"title": title, "url": clean_text(url), "normalized_url": normalize_url(url)})
    return pairs


@dataclass(frozen=True)
class FileSource:
    path: Path
    sha256: str | None = None

    def verify(self, label: str) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Configured {label} does not exist: {self.path}")
        if self.sha256:
            actual = sha256_file(self.path)
            if actual.lower() != self.sha256.lower():
                raise ValueError(
                    f"Checksum mismatch for {label}: expected {self.sha256}, got {actual}."
                )


def _source_fingerprint(source: FileSource, label: str) -> dict[str, Any]:
    if not source.path.is_file():
        raise FileNotFoundError(f"Configured {label} does not exist: {source.path}")
    actual = sha256_file(source.path)
    if source.sha256 and actual.lower() != source.sha256.lower():
        raise ValueError(
            f"Checksum mismatch for {label}: expected {source.sha256}, got {actual}."
        )
    return {
        "path": str(source.path),
        "sha256": actual,
        "configured_sha256": source.sha256,
    }


def _consumed_input_fingerprints(sources: "PipelineSources") -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "evqa_train_csv": _source_fingerprint(sources.evqa_train, "evqa.train_csv"),
        "evqa_val_csv": _source_fingerprint(sources.evqa_val, "evqa.val_csv"),
        "inaturalist_train_metadata": _source_fingerprint(
            sources.inat_train.metadata, "inaturalist.train.metadata"
        ),
        "inaturalist_val_metadata": _source_fingerprint(
            sources.inat_val.metadata, "inaturalist.val.metadata"
        ),
        "wikipedia_kb": _source_fingerprint(sources.wikipedia_kb, "wikipedia.kb"),
    }
    if sources.inat_train.archive is not None:
        inputs["inaturalist_train_archive"] = _source_fingerprint(
            sources.inat_train.archive, "inaturalist.train.archive"
        )
    if sources.inat_val.archive is not None:
        inputs["inaturalist_val_archive"] = _source_fingerprint(
            sources.inat_val.archive, "inaturalist.val.archive"
        )
    return inputs


@dataclass(frozen=True)
class InatSplitSource:
    metadata: FileSource
    image_root: Path
    archive: FileSource | None = None


@dataclass(frozen=True)
class PipelineSources:
    config_path: Path
    output_dir: Path
    evqa_train: FileSource
    evqa_val: FileSource
    inat_train: InatSplitSource
    inat_val: InatSplitSource
    wikipedia_kb: FileSource
    seed: int
    raw: dict[str, Any]

    @property
    def catalog_path(self) -> Path:
        return self.output_dir / "catalog.sqlite"


def _resolve_config_path(base: Path, value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded.resolve(strict=False) if expanded.is_absolute() else (base / expanded).resolve(strict=False)


def _file_source(value: Any, *, base: Path, label: str, required: bool = True) -> FileSource | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing required sources config entry: {label}")
        return None
    if isinstance(value, str):
        return FileSource(_resolve_config_path(base, value))
    if not isinstance(value, dict) or not value.get("path"):
        raise ValueError(f"{label} must be a path string or an object with a path field.")
    return FileSource(
        _resolve_config_path(base, str(value["path"])),
        clean_text(value.get("sha256")) or None,
    )


def load_sources(config_path: str | Path) -> PipelineSources:
    config_path = Path(config_path).resolve(strict=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Sources config must be a JSON object.")
    base = config_path.parent
    evqa = raw.get("evqa") or {}
    inat = raw.get("inaturalist") or {}
    wikipedia = raw.get("wikipedia") or {}

    def inat_source(split: str) -> InatSplitSource:
        block = inat.get(split) or {}
        metadata = _file_source(block.get("metadata"), base=base, label=f"inaturalist.{split}.metadata")
        archive = _file_source(
            block.get("archive"),
            base=base,
            label=f"inaturalist.{split}.archive",
            required=False,
        )
        image_root_value = block.get("image_root")
        if not image_root_value:
            raise ValueError(f"Missing required sources config entry: inaturalist.{split}.image_root")
        return InatSplitSource(
            metadata=metadata,
            archive=archive,
            image_root=_resolve_config_path(base, str(image_root_value)),
        )

    output_value = raw.get("output_dir")
    if not output_value:
        raise ValueError("Missing required sources config entry: output_dir")
    kb_value = wikipedia.get("kb") or wikipedia.get("kb_json") or wikipedia.get("kb_zip")
    return PipelineSources(
        config_path=config_path,
        output_dir=_resolve_config_path(base, str(output_value)),
        evqa_train=_file_source(evqa.get("train_csv"), base=base, label="evqa.train_csv"),
        evqa_val=_file_source(evqa.get("val_csv"), base=base, label="evqa.val_csv"),
        inat_train=inat_source("train"),
        inat_val=inat_source("val"),
        wikipedia_kb=_file_source(kb_value, base=base, label="wikipedia.kb"),
        seed=int(raw.get("seed", 42)),
        raw=raw,
    )


def source_for_split(sources: PipelineSources, split: str) -> InatSplitSource:
    if split == "train":
        return sources.inat_train
    if split == "val":
        return sources.inat_val
    raise ValueError(f"Unsupported iNaturalist source split: {split}")


def _atomic_replace_writer(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        write_fn(temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_replace_writer(
        path,
        lambda temp: temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        ),
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0

    def write(temp: Path) -> None:
        nonlocal count
        with temp.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1

    _atomic_replace_writer(path, write)
    return count


def _publish_artifact_set(
    staged_paths: Mapping[str, Path],
    final_paths: Mapping[str, Path],
) -> None:
    """Publish a related artifact set with best-effort transactional rollback.

    POSIX has no atomic rename for multiple files. Move any previous generation
    aside first, publish every staged member, and restore the complete previous
    generation if one replacement fails. The manifest/report must be ordered
    last by callers so consumers never receive a commit marker for an
    incomplete generation.
    """

    if list(staged_paths) != list(final_paths):
        raise ValueError("staged and final artifact sets must have the same ordered keys")
    for name, staged in staged_paths.items():
        if not staged.is_file():
            raise FileNotFoundError(f"staged artifact {name!r} is missing: {staged}")

    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for name, final in final_paths.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            if not final.exists():
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{final.name}.",
                suffix=".backup",
                dir=final.parent,
            )
            os.close(descriptor)
            backup = Path(backup_name)
            backup.unlink()
            os.replace(final, backup)
            backups[name] = backup

        for name, staged in staged_paths.items():
            os.replace(staged, final_paths[name])
            published.append(name)
    except Exception as publish_error:
        rollback_errors: list[str] = []
        for name in reversed(published):
            try:
                final_paths[name].unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"remove {final_paths[name]}: {exc}")
        for name, backup in reversed(list(backups.items())):
            try:
                if backup.exists():
                    os.replace(backup, final_paths[name])
            except OSError as exc:
                rollback_errors.append(f"restore {final_paths[name]}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "Artifact publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from publish_error
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            yield value


class JsonCharReader:
    """Small streaming JSON reader used for multi-gigabyte top-level objects."""

    def __init__(self, stream: TextIO, chunk_size: int = 1 << 20):
        self.stream = stream
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.pending: list[str] = []

    def read_char(self) -> str:
        if self.pending:
            return self.pending.pop()
        if self.position >= len(self.buffer):
            self.buffer = self.stream.read(self.chunk_size)
            self.position = 0
            if not self.buffer:
                return ""
        char = self.buffer[self.position]
        self.position += 1
        return char

    def unread(self, char: str) -> None:
        if char:
            self.pending.append(char)


def _skip_ws(reader: JsonCharReader) -> str:
    while True:
        char = reader.read_char()
        if not char or not char.isspace():
            return char


def _read_raw_string(reader: JsonCharReader) -> str:
    output = ['"']
    escaped = False
    while True:
        char = reader.read_char()
        if not char:
            raise ValueError("Unexpected EOF in JSON string.")
        output.append(char)
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return "".join(output)


def _collect_json_value(reader: JsonCharReader, first: str) -> str:
    if first == '"':
        return _read_raw_string(reader)
    output = [first]
    if first not in "[{":
        while True:
            char = reader.read_char()
            if not char or char in ",}]":
                reader.unread(char)
                return "".join(output).strip()
            output.append(char)

    stack = [first]
    while stack:
        char = reader.read_char()
        if not char:
            raise ValueError("Unexpected EOF in JSON value.")
        output.append(char)
        if char == '"':
            raw_tail = _read_raw_string(reader)
            output.append(raw_tail[1:])
        elif char in "[{":
            stack.append(char)
        elif char == "}" and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack[-1] == "[":
            stack.pop()
    return "".join(output)


def _skip_json_value(reader: JsonCharReader, first: str) -> None:
    _collect_json_value(reader, first)


def _iter_array_values(reader: JsonCharReader) -> Iterator[dict[str, Any]]:
    char = _skip_ws(reader)
    if char == "]":
        return
    while True:
        value = json.loads(_collect_json_value(reader, char))
        if not isinstance(value, dict):
            raise ValueError("Expected objects inside metadata array.")
        yield value
        delimiter = _skip_ws(reader)
        if delimiter == "]":
            return
        if delimiter != ",":
            raise ValueError(f"Expected ',' or ']' in JSON array, got {delimiter!r}.")
        char = _skip_ws(reader)


def iter_top_level_arrays(
    stream: TextIO,
    selected_keys: set[str],
) -> Iterator[tuple[str, dict[str, Any]]]:
    reader = JsonCharReader(stream)
    if _skip_ws(reader) != "{":
        raise ValueError("Metadata JSON must be a top-level object.")
    while True:
        char = _skip_ws(reader)
        if char == "}":
            return
        if char != '"':
            raise ValueError(f"Expected top-level JSON key, got {char!r}.")
        key = json.loads(_read_raw_string(reader))
        if _skip_ws(reader) != ":":
            raise ValueError("Expected ':' after top-level JSON key.")
        first = _skip_ws(reader)
        if key in selected_keys:
            if first != "[":
                raise ValueError(f"Expected top-level {key!r} to be an array.")
            for value in _iter_array_values(reader):
                yield key, value
        else:
            _skip_json_value(reader, first)
        delimiter = _skip_ws(reader)
        if delimiter == "}":
            return
        if delimiter != ",":
            raise ValueError(f"Expected ',' or '}}', got {delimiter!r}.")


def _create_catalog_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE categories (
            category_id TEXT PRIMARY KEY,
            category_json TEXT NOT NULL
        );
        CREATE TABLE image_categories (
            image_key TEXT PRIMARY KEY,
            category_id TEXT NOT NULL
        );
        CREATE TABLE images (
            image_key TEXT PRIMARY KEY,
            image_id TEXT NOT NULL,
            source_split TEXT NOT NULL,
            file_name TEXT NOT NULL,
            normalized_path TEXT NOT NULL,
            category_id TEXT,
            category_key TEXT,
            category_json TEXT
        );
        CREATE UNIQUE INDEX images_split_id ON images(source_split, image_id);
        CREATE INDEX images_category ON images(category_id);
        CREATE INDEX images_normalized_path ON images(normalized_path);
        """
    )


def _stream_metadata_into_catalog(
    connection: sqlite3.Connection,
    source_split: str,
    source: InatSplitSource,
) -> dict[str, int]:
    counters: Counter[str] = Counter()
    with source.metadata.path.open("r", encoding="utf-8") as stream:
        for key, item in iter_top_level_arrays(stream, {"images", "annotations", "categories"}):
            if key == "categories":
                category_id = clean_text(item.get("id"))
                if not category_id:
                    raise ValueError(f"Invalid {source_split} category without id: {item!r}")
                category_json = canonical_json(item)
                existing = connection.execute(
                    "SELECT category_json FROM categories WHERE category_id = ?", (category_id,)
                ).fetchone()
                if existing and existing[0] != category_json:
                    raise ValueError(f"Conflicting iNaturalist category metadata for {category_id}.")
                connection.execute(
                    "INSERT OR IGNORE INTO categories(category_id, category_json) VALUES (?, ?)",
                    (category_id, category_json),
                )
            elif key == "annotations":
                image_id = clean_text(item.get("image_id"))
                category_id = clean_text(item.get("category_id"))
                if not image_id or not category_id:
                    raise ValueError(
                        f"Invalid {source_split} annotation without image/category id: {item!r}"
                    )
                image_key = f"inaturalist:{image_id}"
                existing = connection.execute(
                    "SELECT category_id FROM image_categories WHERE image_key = ?", (image_key,)
                ).fetchone()
                if existing and existing[0] != category_id:
                    raise ValueError(f"Conflicting iNaturalist categories for {image_key}.")
                connection.execute(
                    "INSERT OR REPLACE INTO image_categories(image_key, category_id) VALUES (?, ?)",
                    (image_key, category_id),
                )
            else:
                image_id = clean_text(item.get("id"))
                file_name = clean_text(item.get("file_name"))
                if not image_id or not file_name:
                    raise ValueError(
                        f"Invalid {source_split} image without id/file_name: {item!r}"
                    )
                file_name, normalized_path = _safe_image_destination(
                    source.image_root, file_name
                )
                image_key = f"inaturalist:{image_id}"
                try:
                    connection.execute(
                        """
                        INSERT INTO images(
                            image_key, image_id, source_split, file_name, normalized_path
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (image_key, image_id, source_split, file_name, normalized_path),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate iNaturalist image identity: {image_key}") from exc
            counters[key] += 1
            if sum(counters.values()) % 10000 == 0:
                connection.commit()
    connection.commit()
    return dict(counters)


def _build_catalog_impl(sources: PipelineSources) -> dict[str, Any]:
    catalog_inputs = {
        "inaturalist_train_metadata": _source_fingerprint(
            sources.inat_train.metadata, "inaturalist.train.metadata"
        ),
        "inaturalist_val_metadata": _source_fingerprint(
            sources.inat_val.metadata, "inaturalist.val.metadata"
        ),
    }
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".catalog.", suffix=".sqlite", dir=sources.output_dir
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink()
    try:
        connection = sqlite3.connect(temp_path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            _create_catalog_schema(connection)
            split_reports = {
                "train": _stream_metadata_into_catalog(connection, "train", sources.inat_train),
                "val": _stream_metadata_into_catalog(connection, "val", sources.inat_val),
            }
            connection.execute(
                """
                UPDATE images
                SET category_id = (
                        SELECT category_id FROM image_categories
                        WHERE image_categories.image_key = images.image_key
                    )
                """
            )
            connection.execute(
                "UPDATE images SET category_key = 'inaturalist:' || category_id "
                "WHERE category_id IS NOT NULL"
            )
            connection.execute(
                """
                UPDATE images
                SET category_json = (
                    SELECT category_json FROM categories
                    WHERE categories.category_id = images.category_id
                )
                """
            )
            connection.commit()
            image_count = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
            uncategorized = int(
                connection.execute("SELECT COUNT(*) FROM images WHERE category_id IS NULL").fetchone()[0]
            )
            category_count = int(connection.execute("SELECT COUNT(*) FROM categories").fetchone()[0])
            orphan_annotations = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM image_categories
                    LEFT JOIN images USING(image_key)
                    WHERE images.image_key IS NULL
                    """
                ).fetchone()[0]
            )
            unknown_categories = int(
                connection.execute(
                    "SELECT COUNT(*) FROM images WHERE category_id IS NOT NULL AND category_json IS NULL"
                ).fetchone()[0]
            )
            if uncategorized or orphan_annotations or unknown_categories:
                raise ValueError(
                    "iNaturalist metadata is incomplete: "
                    f"uncategorized_images={uncategorized}, "
                    f"orphan_annotations={orphan_annotations}, "
                    f"unknown_categories={unknown_categories}."
                )
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode=DELETE")
        finally:
            connection.close()
        os.replace(temp_path, sources.catalog_path)
    finally:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(temp_path) + suffix)
            if candidate.exists():
                candidate.unlink()

    report = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "catalog",
        "created_at": utc_now(),
        "catalog_path": str(sources.catalog_path),
        "catalog_sha256": sha256_file(sources.catalog_path),
        "images": image_count,
        "categories": category_count,
        "uncategorized_images": uncategorized,
        "splits": split_reports,
        "input_sha256": {
            "inaturalist.train.metadata": catalog_inputs["inaturalist_train_metadata"]["sha256"],
            "inaturalist.val.metadata": catalog_inputs["inaturalist_val_metadata"]["sha256"],
        },
        "inputs": catalog_inputs,
    }
    atomic_write_json(sources.output_dir / "catalog_manifest.json", report)
    return report


def build_catalog(sources: PipelineSources) -> dict[str, Any]:
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _build_catalog_impl(sources)
    except Exception as exc:
        atomic_write_json(
            sources.output_dir / "catalog_preflight_report.json",
            {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "stage": "catalog",
                "status": "failed",
                "created_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "inputs": {
                    "inaturalist_train_metadata": str(sources.inat_train.metadata.path),
                    "inaturalist_val_metadata": str(sources.inat_val.metadata.path),
                },
                "partial_catalog_written": False,
            },
        )
        raise


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _lookup_query_image(
    connection: sqlite3.Connection,
    image_id: str,
    expected_split: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT image_key, image_id, source_split, file_name, normalized_path,
               category_id, category_key, category_json
        FROM images WHERE source_split = ? AND image_id = ?
        """,
        (expected_split, image_id),
    ).fetchone()
    if not row:
        return None
    keys = (
        "image_key",
        "image_id",
        "source_split",
        "file_name",
        "normalized_path",
        "category_id",
        "category_key",
        "category_json",
    )
    value = dict(zip(keys, row))
    value["category"] = json.loads(value.pop("category_json") or "{}")
    return value


def _sample_id(
    row: Mapping[str, Any],
    official_split: str,
    source_row_index: int,
    image_keys: list[str],
) -> str:
    question_id = clean_text(row.get("question_id") or row.get("id"))
    identity = {
        "official_split": official_split,
        "question_id": question_id or f"source-row:{source_row_index}",
        "question": clean_text(row.get("question")),
        "question_original": clean_text(row.get("question_original")),
        "question_type": clean_text(row.get("question_type")),
        "answer": clean_text(row.get("answer")),
        "multi_answer": clean_text(row.get("multi_answer")),
        "evidence": clean_text(row.get("evidence")),
        "evidence_section_id": clean_text(row.get("evidence_section_id")),
        "evidence_section_title": clean_text(row.get("evidence_section_title")),
        "wikipedia_url": [pair["normalized_url"] for pair in row_wiki_pairs(row)],
        "dataset_category_id": clean_text(row.get("dataset_category_id")),
        # The order is part of the identity because it defines image_index.
        "image_keys": image_keys,
    }
    return f"evqa:{stable_digest(identity)[:24]}"


def _logical_sample(
    row: Mapping[str, Any],
    *,
    official_split: str,
    source_row_index: int,
    images: list[Mapping[str, Any]],
) -> dict[str, Any]:
    query_images = [
        {
            "image_index": image_index,
            "dataset_image_id": str(image["image_id"]),
            "image_key": str(image["image_key"]),
            "image": str(image["normalized_path"]),
            "source_file_name": str(image["file_name"]),
            "source_split": str(image["source_split"]),
        }
        for image_index, image in enumerate(images, start=1)
    ]
    image_keys = [image["image_key"] for image in query_images]
    sample_id = _sample_id(row, official_split, source_row_index, image_keys)
    category_id = str(images[0]["category_id"] or "")
    category_key = str(images[0]["category_key"] or "")
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "question_id": clean_text(row.get("question_id") or row.get("id")),
        "official_split": official_split,
        "source_row_index": source_row_index,
        "dataset_name": SUPPORTED_DATASET,
        "question": clean_text(row.get("question")),
        "question_original": clean_text(row.get("question_original")),
        "question_type": clean_text(row.get("question_type")),
        "answer": clean_text(row.get("answer")),
        "answers": parse_answers(row),
        "multi_answer": clean_text(row.get("multi_answer")),
        "evidence": clean_text(row.get("evidence")),
        "evidence_section_id": clean_text(row.get("evidence_section_id")),
        "evidence_section_title": clean_text(row.get("evidence_section_title")),
        "wikipedia_title": clean_text(row.get("wikipedia_title")),
        "wikipedia_url": clean_text(row.get("wikipedia_url")),
        "wiki_pairs": row_wiki_pairs(row),
        "query_images": query_images,
        "image_keys": image_keys,
        "image_count": len(query_images),
        "dataset_category_id": category_id,
        "category_key": category_key,
    }


def _dataset_distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize parent questions and their ordered query images."""

    row_list = list(rows)
    image_count_histogram: Counter[str] = Counter()
    by_question_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    query_image_count = 0
    single_image_questions = 0
    multi_image_questions = 0
    resolvable = 0
    unresolvable = 0
    unresolvable_reasons: Counter[str] = Counter()
    has_resolvability = False

    for row in row_list:
        image_count = int(row.get("image_count") or len(row.get("query_images") or []))
        query_image_count += image_count
        image_count_histogram[str(image_count)] += 1
        single_image_questions += int(image_count == 1)
        multi_image_questions += int(image_count > 1)
        question_type = clean_text(row.get("question_type")) or "<missing>"
        category_key = clean_text(row.get("category_key")) or "<missing>"
        for bucket in (by_question_type[question_type], by_category[category_key]):
            bucket["questions"] += 1
            bucket["query_images"] += image_count

        if "retrieval_resolvable" not in row:
            continue
        has_resolvability = True
        if bool(row["retrieval_resolvable"]):
            resolvable += 1
            by_question_type[question_type]["retrieval_resolvable"] += 1
            by_category[category_key]["retrieval_resolvable"] += 1
            continue
        unresolvable += 1
        by_question_type[question_type]["unresolvable"] += 1
        by_category[category_key]["unresolvable"] += 1
        if not bool(row.get("vision_resolvable")):
            unresolvable_reasons["no_visual_positive"] += 1
            by_question_type[question_type]["reason:no_visual_positive"] += 1
            by_category[category_key]["reason:no_visual_positive"] += 1
        if not bool(row.get("text_resolvable")):
            unresolvable_reasons["missing_text_evidence"] += 1
            by_question_type[question_type]["reason:missing_text_evidence"] += 1
            by_category[category_key]["reason:missing_text_evidence"] += 1

    result: dict[str, Any] = {
        "rl_questions": len(row_list),
        "query_images": query_image_count,
        "single_image_questions": single_image_questions,
        "multi_image_questions": multi_image_questions,
        "image_count_histogram": dict(
            sorted(image_count_histogram.items(), key=lambda item: int(item[0]))
        ),
        "by_question_type": {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(by_question_type.items())
        },
        "by_category": {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(by_category.items())
        },
    }
    if has_resolvability:
        result["retrieval_resolvable"] = resolvable
        result["retrieval_unresolvable"] = unresolvable
        result["unresolvable_reasons"] = dict(sorted(unresolvable_reasons.items()))
    return result


def _require_schema_v2(value: Mapping[str, Any], label: str) -> None:
    if value.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible {label} schema_version={value.get('schema_version')!r}; "
            "this pipeline requires schema_version=2. Rerun catalog, split, corpus, and sft."
        )


def _validate_logical_v2(rows: Iterable[Mapping[str, Any]], label: str) -> None:
    ambiguous_scalar_fields = {
        "dataset_image_id",
        "image_key",
        "image",
        "source_file_name",
        "source_split",
        "all_image_ids",
    }
    required_image_fields = {
        "image_index",
        "dataset_image_id",
        "image_key",
        "image",
        "source_file_name",
        "source_split",
    }
    for row in rows:
        raw_sample_id = clean_text(row.get("sample_id"))
        sample_id = raw_sample_id or "<missing>"
        _require_schema_v2(row, f"{label} logical sample {sample_id}")
        if not raw_sample_id:
            raise ValueError(f"Logical {label} sample is missing sample_id.")
        present_ambiguous = sorted(ambiguous_scalar_fields.intersection(row))
        if present_ambiguous:
            raise ValueError(
                f"Logical sample {sample_id} contains obsolete scalar image fields "
                f"{present_ambiguous}. Rerun split, corpus, and sft."
            )
        query_images = row.get("query_images")
        if not isinstance(query_images, list) or not query_images:
            raise ValueError(f"Logical sample {sample_id} has no query_images.")
        expected_indices = list(range(1, len(query_images) + 1))
        indices: list[Any] = []
        seen_image_keys: set[str] = set()
        for image in query_images:
            if not isinstance(image, dict):
                raise ValueError(f"Logical sample {sample_id} has a non-object query image.")
            if set(image) != required_image_fields:
                raise ValueError(
                    f"Logical sample {sample_id} query image fields do not match schema v2."
                )
            indices.append(image.get("image_index"))
            required_strings = (
                clean_text(image.get("dataset_image_id")),
                clean_text(image.get("image_key")),
                clean_text(image.get("image")),
                clean_text(image.get("source_file_name")),
                clean_text(image.get("source_split")),
            )
            if not all(required_strings):
                raise ValueError(
                    f"Logical sample {sample_id} has an incomplete query image."
                )
            if clean_text(image.get("source_split")) != clean_text(row.get("official_split")):
                raise ValueError(
                    f"Logical sample {sample_id} contains a query image from the wrong "
                    "official iNaturalist split."
                )
            image_key = clean_text(image.get("image_key"))
            if image_key in seen_image_keys:
                raise ValueError(
                    f"Logical sample {sample_id} contains duplicate query image {image_key}."
                )
            seen_image_keys.add(image_key)
        if indices != expected_indices:
            raise ValueError(
                f"Logical sample {sample_id} has non-contiguous 1-based image_index values."
            )
        image_keys = [clean_text(image.get("image_key")) for image in query_images]
        if row.get("image_keys") != image_keys:
            raise ValueError(f"Logical sample {sample_id} has inconsistent image_keys.")
        if row.get("image_count") != len(query_images):
            raise ValueError(f"Logical sample {sample_id} has inconsistent image_count.")
        if not clean_text(row.get("category_key")) or not clean_text(
            row.get("dataset_category_id")
        ):
            raise ValueError(
                f"Logical sample {sample_id} is missing its shared category identity."
            )
        source_row_index = row.get("source_row_index")
        if (
            isinstance(source_row_index, bool)
            or not isinstance(source_row_index, int)
            or source_row_index < 0
        ):
            raise ValueError(
                f"Logical sample {sample_id} has an invalid source_row_index."
            )
        expected_sample_id = _sample_id(
            row,
            clean_text(row.get("official_split")),
            source_row_index,
            image_keys,
        )
        if sample_id != expected_sample_id:
            raise ValueError(
                f"Logical sample {sample_id} no longer matches its stable identity. "
                "Rerun split, corpus, and sft."
            )


def build_split(sources: PipelineSources) -> dict[str, Any]:
    if not sources.catalog_path.is_file():
        raise FileNotFoundError("catalog.sqlite is missing; run the catalog stage first.")
    catalog_manifest_path = sources.output_dir / "catalog_manifest.json"
    if not catalog_manifest_path.is_file():
        raise FileNotFoundError(
            "catalog_manifest.json is missing; rerun catalog, split, corpus, and sft."
        )
    catalog_manifest = json.loads(catalog_manifest_path.read_text(encoding="utf-8"))
    _require_schema_v2(catalog_manifest, "catalog manifest")
    split_inputs = {
        "evqa_train_csv": _source_fingerprint(sources.evqa_train, "evqa.train_csv"),
        "evqa_val_csv": _source_fingerprint(sources.evqa_val, "evqa.val_csv"),
        "inat_catalog": {
            "path": str(sources.catalog_path),
            "sha256": sha256_file(sources.catalog_path),
        },
    }
    connection = sqlite3.connect(sources.catalog_path)
    logical_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    missing_queries: list[dict[str, Any]] = []
    category_mismatches: list[dict[str, Any]] = []
    dataset_counts: Counter[str] = Counter()
    gldv2_categories: set[str] = set()
    sample_ids: set[str] = set()
    try:
        for official_split, source, output_split in (
            ("train", sources.evqa_train, "train"),
            ("val", sources.evqa_val, "test"),
        ):
            for row_index, row in enumerate(_load_csv(source.path)):
                dataset_name = clean_text(row.get("dataset_name")).lower()
                dataset_counts[dataset_name or "<missing>"] += 1
                if dataset_name != SUPPORTED_DATASET:
                    if dataset_name in GLDV2_NAMES:
                        category = clean_text(row.get("dataset_category_id"))
                        if category:
                            gldv2_categories.add(category)
                    continue
                image_ids = deduplicate(split_delimited(row.get("dataset_image_ids")))
                if not image_ids:
                    missing_queries.append(
                        {
                            "official_split": official_split,
                            "source_row_index": row_index,
                            "reason": "no_dataset_image_ids",
                        }
                    )
                    continue
                resolved_images: list[dict[str, Any]] = []
                row_has_errors = False
                for image_id in image_ids:
                    image = _lookup_query_image(connection, image_id, official_split)
                    if image is None:
                        other_split = "val" if official_split == "train" else "train"
                        exists_elsewhere = _lookup_query_image(connection, image_id, other_split) is not None
                        missing_queries.append(
                            {
                                "official_split": official_split,
                                "source_row_index": row_index,
                                "image_id": image_id,
                                "reason": "found_only_in_wrong_split" if exists_elsewhere else "missing_catalog_image",
                            }
                        )
                        row_has_errors = True
                        continue
                    declared_category = clean_text(row.get("dataset_category_id"))
                    actual_category = clean_text(image.get("category_id"))
                    if not actual_category:
                        missing_queries.append(
                            {
                                "official_split": official_split,
                                "source_row_index": row_index,
                                "image_id": image_id,
                                "reason": "missing_catalog_category",
                            }
                        )
                        row_has_errors = True
                        continue
                    if declared_category and declared_category != actual_category:
                        category_mismatches.append(
                            {
                                "official_split": official_split,
                                "source_row_index": row_index,
                                "image_id": image_id,
                                "declared": declared_category,
                                "catalog": actual_category,
                                "reason": "declared_category_mismatch",
                            }
                        )
                        row_has_errors = True
                        continue
                    resolved_images.append(image)
                actual_categories = {
                    clean_text(image.get("category_id")) for image in resolved_images
                }
                if len(actual_categories) > 1:
                    category_mismatches.append(
                        {
                            "official_split": official_split,
                            "source_row_index": row_index,
                            "image_ids": [str(image["image_id"]) for image in resolved_images],
                            "catalog_categories": sorted(actual_categories),
                            "reason": "inconsistent_query_categories",
                        }
                    )
                    row_has_errors = True
                if row_has_errors:
                    continue
                sample = _logical_sample(
                    row,
                    official_split=official_split,
                    source_row_index=row_index,
                    images=resolved_images,
                )
                if sample["sample_id"] in sample_ids:
                    raise ValueError(f"Duplicate stable sample ID: {sample['sample_id']}")
                sample_ids.add(sample["sample_id"])
                logical_by_split[output_split].append(sample)
    finally:
        connection.close()

    preflight = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "split",
        "created_at": utc_now(),
        "missing_queries": missing_queries,
        "category_mismatches": category_mismatches,
        "dataset_rows": dict(sorted(dataset_counts.items())),
        "gldv2_rows_skipped": sum(dataset_counts.get(name, 0) for name in GLDV2_NAMES),
        "gldv2_categories_skipped": sorted(gldv2_categories),
        "inputs": split_inputs,
    }
    if missing_queries or category_mismatches:
        atomic_write_json(sources.output_dir / "split_preflight_report.json", preflight)
        raise RuntimeError(
            "EVQA query preflight failed. No split artifacts were written; see "
            f"{sources.output_dir / 'split_preflight_report.json'}."
        )

    for rows in logical_by_split.values():
        rows.sort(key=lambda item: item["sample_id"])
    heldout_keys = sorted(
        {
            image["image_key"]
            for rows in logical_by_split.values()
            for row in rows
            for image in row["query_images"]
        }
    )
    heldout_paths = sorted(
        {
            normalize_local_path(image["image"])
            for rows in logical_by_split.values()
            for row in rows
            for image in row["query_images"]
        }
    )
    relevant_categories = sorted(
        {row["dataset_category_id"] for rows in logical_by_split.values() for row in rows}
    )
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".split-stage-", dir=sources.output_dir))
    try:
        final_paths = {
            "logical_train": sources.output_dir / "logical_train.jsonl",
            "logical_test": sources.output_dir / "logical_test.jsonl",
            "heldout_manifest": sources.output_dir / "heldout_manifest.json",
            # Publish this commit marker last.
            "split_report": sources.output_dir / "split_report.json",
        }
        staged_paths = {
            name: staging_dir / final.name for name, final in final_paths.items()
        }
        atomic_write_jsonl(staged_paths["logical_train"], logical_by_split["train"])
        atomic_write_jsonl(staged_paths["logical_test"], logical_by_split["test"])
        heldout_manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "created_at": utc_now(),
            "image_keys": heldout_keys,
            "normalized_paths": heldout_paths,
            "image_keys_sha256": stable_digest(heldout_keys),
            "normalized_paths_sha256": stable_digest(heldout_paths),
            "relevant_category_ids": relevant_categories,
        }
        atomic_write_json(staged_paths["heldout_manifest"], heldout_manifest)
        distribution = {
            "train": _dataset_distribution(logical_by_split["train"]),
            "test": _dataset_distribution(logical_by_split["test"]),
            "overall": _dataset_distribution(
                logical_by_split["train"] + logical_by_split["test"]
            ),
        }
        split_artifacts = {
            name: _artifact_record(
                staged_paths[name],
                jsonl=name in {"logical_train", "logical_test"},
            )
            for name in ("logical_train", "logical_test", "heldout_manifest")
        }
        for name, record in split_artifacts.items():
            record["path"] = str(final_paths[name])
        report = {
            **preflight,
            "train_samples": len(logical_by_split["train"]),
            "test_samples": len(logical_by_split["test"]),
            "distribution": distribution,
            "heldout_images": len(heldout_keys),
            "relevant_categories": len(relevant_categories),
            "logical_train_path": str(final_paths["logical_train"]),
            "logical_test_path": str(final_paths["logical_test"]),
            "heldout_manifest_path": str(final_paths["heldout_manifest"]),
            "artifacts": split_artifacts,
        }
        atomic_write_json(staged_paths["split_report"], report)
        _publish_artifact_set(staged_paths, final_paths)
        return report
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _catalog_rows_for_categories(
    catalog_path: Path,
    category_ids: set[str],
) -> Iterator[dict[str, Any]]:
    connection = sqlite3.connect(catalog_path)
    try:
        cursor = connection.execute(
            """
            SELECT image_key, image_id, source_split, file_name, normalized_path,
                   category_id, category_key, category_json
            FROM images WHERE category_id IS NOT NULL
            ORDER BY source_split, image_id
            """
        )
        keys = (
            "image_key",
            "image_id",
            "source_split",
            "file_name",
            "normalized_path",
            "category_id",
            "category_key",
            "category_json",
        )
        for raw in cursor:
            row = dict(zip(keys, raw))
            if row["category_id"] not in category_ids:
                continue
            row["category"] = json.loads(row.pop("category_json") or "{}")
            yield row
    finally:
        connection.close()


def _valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _safe_archive_name(name: str) -> str:
    normalized = clean_text(name).replace("\\", "/")
    # Remove only explicit current-directory prefixes.  str.lstrip("./") is
    # unsafe here because it also turns "../secret" into "secret".
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.rstrip("/")
    if not normalized or "\x00" in normalized or normalized.startswith("/"):
        raise ValueError(f"Unsafe archive member name: {name!r}")
    raw_parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"Unsafe archive member name: {name!r}")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or ":" in raw_parts[0]:
        raise ValueError(f"Unsafe archive member name: {name!r}")
    return path.as_posix()


def _safe_image_destination(image_root: Path, member_name: str) -> tuple[str, str]:
    safe_name = _safe_archive_name(member_name)
    root = image_root.resolve(strict=False)
    target = (root / PurePosixPath(safe_name)).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Image member resolves outside configured image_root: {member_name!r}"
        ) from exc
    return safe_name, normalize_local_path(target)


def _extract_indexed_images(
    archive_path: Path,
    image_root: Path,
    source_split: str,
    connection: sqlite3.Connection,
) -> int:
    """Extract selected members using a disk-backed lookup table."""

    extracted = 0
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() and member.name in {".", "./"}:
                continue
            normalized = _safe_archive_name(member.name)
            if not member.isfile():
                continue
            found = connection.execute(
                """
                SELECT target_path FROM members
                WHERE source_split = ? AND file_name = ? AND needs_extract = 1
                """,
                (source_split, normalized),
            ).fetchone()
            if not found:
                continue
            _, expected_target = _safe_image_destination(
                image_root, normalized
            )
            if normalize_local_path(found[0]) != expected_target:
                raise ValueError(
                    f"Catalog target path does not match archive member {normalized!r}."
                )
            target = Path(expected_target)
            raw = archive.extractfile(member)
            if raw is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as output:
                    shutil.copyfileobj(raw, output, length=1 << 20)
                os.replace(temp_name, target)
            finally:
                raw.close()
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            connection.execute(
                "UPDATE members SET needs_extract = 0 WHERE source_split = ? AND file_name = ?",
                (source_split, normalized),
            )
            extracted += 1
    connection.commit()
    return extracted


def _materialize_images(
    sources: PipelineSources,
    query_rows: list[dict[str, Any]],
    candidate_path: Path,
    staging_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    member_db = staging_dir / "image_members.sqlite"
    connection = sqlite3.connect(member_db)
    connection.execute(
        """
        CREATE TABLE members (
            source_split TEXT NOT NULL,
            file_name TEXT NOT NULL,
            target_path TEXT NOT NULL,
            kind TEXT NOT NULL,
            needs_extract INTEGER NOT NULL,
            PRIMARY KEY(source_split, file_name)
        )
        """
    )

    def add_member(row: Mapping[str, Any], kind: str) -> None:
        split = str(row["source_split"])
        name = _safe_archive_name(
            str(row["source_file_name"] if "source_file_name" in row else row["file_name"])
        )
        _, expected_target = _safe_image_destination(
            source_for_split(sources, split).image_root, name
        )
        supplied_target = str(row["image"] if "image" in row else row["normalized_path"])
        if normalize_local_path(supplied_target) != expected_target:
            raise ValueError(
                f"Catalog/query target path does not match safe image destination for {name!r}."
            )
        target = Path(expected_target)
        connection.execute(
            "INSERT OR IGNORE INTO members VALUES (?, ?, ?, ?, ?)",
            (split, name, str(target), kind, int(not _valid_image(target))),
        )

    flattened_queries = [
        (row, image)
        for row in query_rows
        for image in row["query_images"]
    ]
    for _, image in flattened_queries:
        add_member(image, "query")
    for index, row in enumerate(iter_jsonl_rows(candidate_path), start=1):
        add_member(row, "retrieval")
        if index % 10000 == 0:
            connection.commit()
    connection.commit()

    for split in ("train", "val"):
        for kind in ("query", "retrieval"):
            names = connection.execute(
                "SELECT file_name FROM members WHERE source_split = ? AND kind = ? ORDER BY file_name",
                (split, kind),
            )
            path = staging_dir / f"inat_{split}_{kind}_members.txt"
            with path.open("w", encoding="utf-8") as stream:
                for (name,) in names:
                    stream.write(name + "\n")
        union_names = connection.execute(
            "SELECT file_name FROM members WHERE source_split = ? ORDER BY file_name", (split,)
        )
        with (staging_dir / f"inat_{split}_union_members.txt").open("w", encoding="utf-8") as stream:
            for (name,) in union_names:
                stream.write(name + "\n")

    extracted_by_split: dict[str, int] = {}
    for split in ("train", "val"):
        source = source_for_split(sources, split)
        needed = int(
            connection.execute(
                "SELECT COUNT(*) FROM members WHERE source_split = ? AND needs_extract = 1",
                (split,),
            ).fetchone()[0]
        )
        if needed and source.archive:
            extracted_by_split[split] = _extract_indexed_images(
                source.archive.path, source.image_root, split, connection
            )
        else:
            extracted_by_split[split] = 0

    missing_queries: list[dict[str, Any]] = []
    for parent, image in flattened_queries:
        query_path = Path(image["image"])
        if not _valid_image(query_path):
            missing_queries.append(
                {
                    "sample_id": parent["sample_id"],
                    "image_index": image["image_index"],
                    "image_key": image["image_key"],
                    "path": image["image"],
                    "source_split": image["source_split"],
                    "reason": (
                        "missing_query_archive_member_or_file"
                        if not query_path.is_file()
                        else "corrupt_query_pixel"
                    ),
                }
            )
    if missing_queries:
        connection.close()
        raise RuntimeError(canonical_json({"missing_query_images": missing_queries}))

    usable_path = staging_dir / "usable_candidates.jsonl"
    dropped_path = staging_dir / "dropped_candidates.jsonl"
    usable_count = 0
    dropped_count = 0
    dropped_preview: list[dict[str, Any]] = []
    dropped_by_category: Counter[str] = Counter()
    with usable_path.open("w", encoding="utf-8") as usable_stream, dropped_path.open(
        "w", encoding="utf-8"
    ) as dropped_stream:
        for row in iter_jsonl_rows(candidate_path):
            if _valid_image(Path(row["normalized_path"])):
                usable_stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                usable_count += 1
                continue
            dropped = {
                "image_key": row["image_key"],
                "category_id": row["category_id"],
                "path": row["normalized_path"],
                "source_split": row["source_split"],
                "reason": "missing_or_corrupt_candidate_image",
            }
            dropped_stream.write(json.dumps(dropped, ensure_ascii=False, separators=(",", ":")) + "\n")
            dropped_count += 1
            dropped_by_category[str(row["category_id"])] += 1
            if len(dropped_preview) < 100:
                dropped_preview.append(dropped)
    connection.close()
    return usable_path, {
        "extracted_by_split": extracted_by_split,
        "query_images": len(flattened_queries),
        "usable_candidates": usable_count,
        "dropped_candidate_count": dropped_count,
        "dropped_candidates": dropped_preview,
        "dropped_candidates_path": str(sources.output_dir / "dropped_candidates.jsonl"),
        "dropped_by_category": dict(sorted(dropped_by_category.items())),
    }


@contextmanager
def _open_kb_json(source: FileSource) -> Iterator[TextIO]:
    if not source.path.is_file():
        raise FileNotFoundError(f"Configured wikipedia.kb does not exist: {source.path}")
    if zipfile.is_zipfile(source.path):
        with zipfile.ZipFile(source.path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".json")]
            if len(names) != 1:
                raise ValueError("Wikipedia KB zip must contain exactly one JSON file.")
            with archive.open(names[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8") as stream:
                    yield stream
    else:
        with source.path.open("r", encoding="utf-8") as stream:
            yield stream


def _iter_selected_kb_pages(
    stream: TextIO,
    selected_urls: set[str],
) -> Iterator[tuple[str, dict[str, Any]]]:
    reader = JsonCharReader(stream)
    if _skip_ws(reader) != "{":
        raise ValueError("Wikipedia KB JSON must be a top-level object.")
    while True:
        char = _skip_ws(reader)
        if char == "}":
            return
        if char != '"':
            raise ValueError(f"Expected Wikipedia URL key, got {char!r}.")
        url = json.loads(_read_raw_string(reader))
        if _skip_ws(reader) != ":":
            raise ValueError("Expected ':' after Wikipedia URL.")
        first = _skip_ws(reader)
        if normalize_url(url) in selected_urls:
            page = json.loads(_collect_json_value(reader, first))
            if isinstance(page, dict):
                yield url, page
        else:
            _skip_json_value(reader, first)
        delimiter = _skip_ws(reader)
        if delimiter == "}":
            return
        if delimiter != ",":
            raise ValueError(f"Expected ',' or '}}' in KB JSON, got {delimiter!r}.")


def _build_text_corpus(
    query_rows: list[dict[str, Any]],
    kb_source: FileSource,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, Any]]:
    selected_urls = {
        pair["normalized_url"]
        for row in query_rows
        for pair in row.get("wiki_pairs", [])
        if pair.get("normalized_url")
    }
    sections_by_url: dict[str, set[str]] = defaultdict(set)
    text_rows: list[dict[str, Any]] = []
    found_urls: set[str] = set()
    with _open_kb_json(kb_source) as stream:
        for original_url, page in _iter_selected_kb_pages(stream, selected_urls):
            normalized_url = normalize_url(original_url)
            found_urls.add(normalized_url)
            title = clean_text(page.get("title")) or original_url.rsplit("/", 1)[-1]
            section_texts = page.get("section_texts") or []
            section_titles = page.get("section_titles") or []
            for section_id, text in enumerate(section_texts):
                text = clean_text(text)
                if not text:
                    continue
                section_title = clean_text(
                    section_titles[section_id] if section_id < len(section_titles) else ""
                )
                display_title = title if not section_title else f"{title} :: {section_title}"
                text_rows.append(
                    {
                        "id": f"{normalized_url}#section-{section_id}",
                        "url": normalized_url,
                        "source_url": original_url,
                        "title": title,
                        "section_id": section_id,
                        "section_title": section_title,
                        "contents": f'"{display_title}"\n{text}',
                    }
                )
                sections_by_url[normalized_url].add(str(section_id))
    text_rows.sort(key=lambda row: row["id"])
    return text_rows, sections_by_url, {
        "requested_wiki_urls": len(selected_urls),
        "found_wiki_urls": len(found_urls),
        "missing_wiki_urls": sorted(selected_urls - found_urls),
        "text_sections": len(text_rows),
    }


def _is_text_resolvable(row: Mapping[str, Any], sections_by_url: Mapping[str, set[str]]) -> bool:
    urls = [pair.get("normalized_url", "") for pair in row.get("wiki_pairs", [])]
    if not urls or any(not sections_by_url.get(url) for url in urls):
        return False
    required_ids = split_delimited(row.get("evidence_section_id"))
    if not required_ids:
        return True
    if len(required_ids) == len(urls):
        return all(section_id in sections_by_url[url] for url, section_id in zip(urls, required_ids))
    if len(urls) == 1:
        return all(section_id in sections_by_url[urls[0]] for section_id in required_ids)
    # Ambiguous multi-hop attribution is not considered reliably retrievable.
    return False


def _vision_corpus_row(row: Mapping[str, Any]) -> dict[str, Any]:
    category = row.get("category") or {}
    scientific_name = clean_text(category.get("name"))
    common_name = clean_text(category.get("common_name"))
    title = common_name or scientific_name or str(row["category_id"])
    taxonomy = " > ".join(
        clean_text(category.get(key))
        for key in ("kingdom", "phylum", "class", "order", "family", "genus", "specific_epithet")
        if clean_text(category.get(key))
    )
    caption_parts = [f"Entity: {title}."]
    if scientific_name:
        caption_parts.append(f"Scientific name: {scientific_name}.")
    if common_name:
        caption_parts.append(f"Common name: {common_name}.")
    if taxonomy:
        caption_parts.append(f"Taxonomy: {taxonomy}.")
    caption = " ".join(caption_parts)
    return {
        "id": row["image_key"],
        "image_key": row["image_key"],
        "category_key": row["category_key"],
        "dataset_name": SUPPORTED_DATASET,
        "image_id": row["image_id"],
        "category_id": row["category_id"],
        "source_split": row["source_split"],
        "image": row["normalized_path"],
        "source_file_name": row["file_name"],
        "title": title,
        "caption": caption,
        "contents": f'"{title}"\n{caption}',
    }


def _rl_sample(
    logical: Mapping[str, Any],
    *,
    positive_count: int,
    text_resolvable: bool,
) -> dict[str, Any]:
    vision_resolvable = positive_count > 0
    retrieval_resolvable = vision_resolvable and text_resolvable
    question = clean_text(logical.get("question"))
    answers = list(logical.get("answers") or [])
    extra_info = {
        **dict(logical),
        "positive_candidate_count": positive_count,
        "vision_resolvable": vision_resolvable,
        "text_resolvable": text_resolvable,
        "retrieval_resolvable": retrieval_resolvable,
    }
    query_images = [dict(image) for image in logical["query_images"]]
    image_block = "\n".join(
        f"Image {image['image_index']}:\n<image>" for image in query_images
    )
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "data_source": "dual_search",
        "sample_id": logical["sample_id"],
        "category_key": logical["category_key"],
        "dataset_category_id": logical["dataset_category_id"],
        "query_images": query_images,
        "image_keys": list(logical["image_keys"]),
        "image_count": int(logical["image_count"]),
        "question": question,
        "answer": logical.get("answer", ""),
        "question_type": logical.get("question_type", ""),
        "retrieval_resolvable": retrieval_resolvable,
        "positive_candidate_count": positive_count,
        "vision_resolvable": vision_resolvable,
        "text_resolvable": text_resolvable,
        "prompt": [
            {
                "role": "user",
                "content": DEFAULT_PROMPT_TEMPLATE.format(
                    image_block=image_block,
                    question=question,
                ),
            }
        ],
        "images": [{"image": image["image"]} for image in query_images],
        "ability": "vision-search",
        "reward_model": {"style": "rule", "ground_truth": {"target": answers}},
        "extra_info": extra_info,
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to create an empty RL dataset: {path.name}")
    for row in rows:
        _require_schema_v2(row, f"RL sample {row.get('sample_id', '<missing>')}")
        if not row["reward_model"]["ground_truth"]["target"]:
            raise ValueError(f"Sample {row['sample_id']} has no answer target.")
        if row.get("image_count") != len(row.get("query_images") or []):
            raise ValueError(f"Sample {row['sample_id']} has inconsistent image_count.")
        if row.get("image_keys") != [
            image.get("image_key") for image in row.get("query_images") or []
        ]:
            raise ValueError(f"Sample {row['sample_id']} has inconsistent image_keys.")
        prompt_text = "".join(message.get("content", "") for message in row["prompt"])
        if prompt_text.count("<image>") != len(row["images"]):
            raise ValueError(f"Sample {row['sample_id']} has mismatched image placeholders.")
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path, index=False)


def _artifact_record(path: Path, *, jsonl: bool = False) -> dict[str, Any]:
    result = {"path": str(path), "sha256": sha256_file(path)}
    if jsonl:
        result.update(corpus_fingerprint(path))
    return result


def _require_artifact_matches(
    expected: Mapping[str, Any],
    path: Path,
    *,
    label: str,
    jsonl: bool = False,
) -> None:
    required = {"sha256"}
    if jsonl:
        required.update({"fingerprint_schema_version", "id_order_sha256", "row_count"})
    missing = sorted(required.difference(expected))
    if missing:
        raise ValueError(
            f"Split report has no complete fingerprint for {label}: missing {missing}. "
            "Rerun split and corpus."
        )
    actual = _artifact_record(path, jsonl=jsonl)
    mismatched = sorted(key for key in required if actual.get(key) != expected.get(key))
    if mismatched:
        raise ValueError(
            f"{label} changed after split construction (fingerprint mismatch: {mismatched}). "
            "Rerun split and corpus."
        )


def _build_corpus_impl(sources: PipelineSources) -> dict[str, Any]:
    logical_train_path = sources.output_dir / "logical_train.jsonl"
    logical_test_path = sources.output_dir / "logical_test.jsonl"
    heldout_path = sources.output_dir / "heldout_manifest.json"
    split_report_path = sources.output_dir / "split_report.json"
    catalog_manifest_path = sources.output_dir / "catalog_manifest.json"
    for path in (
        sources.catalog_path,
        logical_train_path,
        logical_test_path,
        heldout_path,
        split_report_path,
        catalog_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required split artifact is missing: {path}")
    consumed_inputs = _consumed_input_fingerprints(sources)
    catalog_manifest = json.loads(catalog_manifest_path.read_text(encoding="utf-8"))
    _require_schema_v2(catalog_manifest, "catalog manifest")
    expected_metadata = catalog_manifest.get("input_sha256") or {}
    for old_key, new_key in (
        ("inaturalist.train.metadata", "inaturalist_train_metadata"),
        ("inaturalist.val.metadata", "inaturalist_val_metadata"),
    ):
        if expected_metadata.get(old_key) != consumed_inputs[new_key]["sha256"]:
            raise ValueError(
                f"iNaturalist metadata changed after catalog construction: {new_key}. "
                "Rerun catalog and split."
            )
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    _require_schema_v2(heldout, "heldout manifest")
    _require_schema_v2(split_report, "split report")
    split_artifacts = split_report.get("artifacts")
    if not isinstance(split_artifacts, Mapping):
        raise ValueError(
            "Split report has no schema v2 artifact fingerprints. Rerun split and corpus."
        )
    logical_train = read_jsonl(logical_train_path)
    logical_test = read_jsonl(logical_test_path)
    _validate_logical_v2(logical_train, "train")
    _validate_logical_v2(logical_test, "test")
    for name, path, is_jsonl in (
        ("logical_train", logical_train_path, True),
        ("logical_test", logical_test_path, True),
        ("heldout_manifest", heldout_path, False),
    ):
        expected = split_artifacts.get(name)
        if not isinstance(expected, Mapping):
            raise ValueError(
                f"Split report has no fingerprint for {name}. Rerun split and corpus."
            )
        _require_artifact_matches(
            expected,
            path,
            label=name,
            jsonl=is_jsonl,
        )
    query_rows = logical_train + logical_test
    split_inputs = split_report.get("inputs") or {}
    for key in ("evqa_train_csv", "evqa_val_csv"):
        if (split_inputs.get(key) or {}).get("sha256") != consumed_inputs[key]["sha256"]:
            raise ValueError(f"{key} changed after split construction. Rerun split.")
    current_catalog_sha256 = sha256_file(sources.catalog_path)
    if (split_inputs.get("inat_catalog") or {}).get("sha256") != current_catalog_sha256:
        raise ValueError("iNaturalist catalog changed after split construction. Rerun split.")
    expected_heldout_keys = sorted(
        {
            str(image["image_key"])
            for row in query_rows
            for image in row["query_images"]
        }
    )
    expected_heldout_paths = sorted(
        {
            normalize_local_path(image["image"])
            for row in query_rows
            for image in row["query_images"]
        }
    )
    expected_categories = sorted(
        {str(row["dataset_category_id"]) for row in query_rows}
    )
    if (
        heldout.get("image_keys") != expected_heldout_keys
        or heldout.get("normalized_paths") != expected_heldout_paths
        or heldout.get("relevant_category_ids") != expected_categories
        or heldout.get("image_keys_sha256") != stable_digest(expected_heldout_keys)
        or heldout.get("normalized_paths_sha256") != stable_digest(expected_heldout_paths)
    ):
        raise ValueError(
            "Heldout manifest does not exactly match the ordered query_images in "
            "logical_train/logical_test. Rerun split and corpus."
        )
    heldout_keys = set(expected_heldout_keys)
    heldout_paths = set(expected_heldout_paths)
    relevant_categories = set(expected_categories)

    sources.output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".corpus-stage-", dir=sources.output_dir))
    try:
        candidate_path = staging_dir / "candidate_catalog.jsonl"
        candidate_count = 0
        with candidate_path.open("w", encoding="utf-8") as candidate_stream:
            for row in _catalog_rows_for_categories(sources.catalog_path, relevant_categories):
                if row["image_key"] in heldout_keys:
                    continue
                if normalize_local_path(row["normalized_path"]) in heldout_paths:
                    continue
                candidate_stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                candidate_count += 1
        usable_candidate_path, image_report = _materialize_images(
            sources, query_rows, candidate_path, staging_dir
        )

        positive_counts: Counter[str] = Counter()

        def iter_vision_rows() -> Iterator[dict[str, Any]]:
            for candidate in iter_jsonl_rows(usable_candidate_path):
                positive_counts[str(candidate["category_id"])] += 1
                yield _vision_corpus_row(candidate)

        text_rows, sections_by_url, text_report = _build_text_corpus(
            query_rows, sources.wikipedia_kb
        )

        staged_paths = {
            "vision_corpus": staging_dir / "vision_corpus.jsonl",
            "text_corpus": staging_dir / "text_corpus.jsonl",
            "train": staging_dir / "train.parquet",
            "test": staging_dir / "test.parquet",
        }
        if (staging_dir / "dropped_candidates.jsonl").exists():
            staged_paths["dropped_candidates"] = staging_dir / "dropped_candidates.jsonl"
        vision_row_count = atomic_write_jsonl(
            staged_paths["vision_corpus"], iter_vision_rows()
        )
        atomic_write_jsonl(staged_paths["text_corpus"], text_rows)

        def build_rl_rows(logical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                _rl_sample(
                    row,
                    positive_count=int(positive_counts.get(row["dataset_category_id"], 0)),
                    text_resolvable=_is_text_resolvable(row, sections_by_url),
                )
                for row in logical_rows
            ]

        train_rows = build_rl_rows(logical_train)
        test_rows = build_rl_rows(logical_test)
        _write_parquet(train_rows, staged_paths["train"])
        _write_parquet(test_rows, staged_paths["test"])
        distribution = {
            "train": _dataset_distribution(train_rows),
            "test": _dataset_distribution(test_rows),
            "overall": _dataset_distribution(train_rows + test_rows),
        }

        reason_counts: Counter[str] = Counter()
        unresolvable_categories: Counter[str] = Counter()
        by_question_type: dict[str, Counter[str]] = defaultdict(Counter)
        by_category: dict[str, Counter[str]] = defaultdict(Counter)
        for row in train_rows + test_rows:
            question_type = clean_text(row.get("question_type")) or "<missing>"
            category_key = clean_text(row.get("category_key")) or "<missing>"
            by_question_type[question_type]["total"] += 1
            by_category[category_key]["total"] += 1
            if row["retrieval_resolvable"]:
                by_question_type[question_type]["retrieval_resolvable"] += 1
                by_category[category_key]["retrieval_resolvable"] += 1
                continue
            by_question_type[question_type]["unresolvable"] += 1
            by_category[category_key]["unresolvable"] += 1
            if not row["vision_resolvable"]:
                reason_counts["no_visual_positive"] += 1
                by_question_type[question_type]["reason:no_visual_positive"] += 1
                by_category[category_key]["reason:no_visual_positive"] += 1
            if not row["text_resolvable"]:
                reason_counts["missing_text_evidence"] += 1
                by_question_type[question_type]["reason:missing_text_evidence"] += 1
                by_category[category_key]["reason:missing_text_evidence"] += 1
            unresolvable_categories[row["category_key"]] += 1

        final_paths = {
            "vision_corpus": sources.output_dir / "vision_corpus.jsonl",
            "text_corpus": sources.output_dir / "text_corpus.jsonl",
            "train": sources.output_dir / "train.parquet",
            "test": sources.output_dir / "test.parquet",
        }
        if "dropped_candidates" in staged_paths:
            final_paths["dropped_candidates"] = sources.output_dir / "dropped_candidates.jsonl"
        artifact_records = {
            name: _artifact_record(
                staged,
                jsonl=name in {"vision_corpus", "text_corpus", "dropped_candidates"},
            )
            for name, staged in staged_paths.items()
        }
        for name, final in final_paths.items():
            artifact_records[name]["path"] = str(final)

        report = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "stage": "corpus",
            "created_at": utc_now(),
            "samples": {"train": len(train_rows), "test": len(test_rows)},
            "distribution": distribution,
            "vision_candidates_before_validation": candidate_count,
            "vision_corpus_rows": vision_row_count,
            "heldout_image_count": len(heldout_keys),
            "heldout_leaks": 0,
            "text": text_report,
            "image_materialization": image_report,
            "retrieval_resolvable": {
                "all": sum(row["retrieval_resolvable"] for row in train_rows + test_rows),
                "train": sum(row["retrieval_resolvable"] for row in train_rows),
                "test": sum(row["retrieval_resolvable"] for row in test_rows),
                "unresolvable": len(train_rows) + len(test_rows)
                - sum(row["retrieval_resolvable"] for row in train_rows + test_rows),
                "reasons": dict(sorted(reason_counts.items())),
                "unresolvable_categories": dict(sorted(unresolvable_categories.items())),
                "by_question_type": {
                    key: dict(sorted(counts.items()))
                    for key, counts in sorted(by_question_type.items())
                },
                "by_category": {
                    key: dict(sorted(counts.items()))
                    for key, counts in sorted(by_category.items())
                },
            },
            "gldv2": {
                "supported": False,
                "policy": "skipped_and_reported",
                "rows_skipped": int(split_report.get("gldv2_rows_skipped", 0)),
                "categories_skipped": split_report.get("gldv2_categories_skipped", []),
            },
        }
        manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "created_at": utc_now(),
            "sources_config_path": str(sources.config_path),
            "sources_config_sha256": sha256_file(sources.config_path),
            "seed": sources.seed,
            "inputs": {
                **consumed_inputs,
                "inat_catalog": {
                    "path": str(sources.catalog_path),
                    "sha256": current_catalog_sha256,
                    "catalog_manifest_path": str(catalog_manifest_path),
                    "catalog_manifest_sha256": sha256_file(catalog_manifest_path),
                },
            },
            "heldout": {
                "manifest_path": str(heldout_path),
                "manifest_sha256": sha256_file(heldout_path),
                "heldout_image_keys": sorted(heldout_keys),
                "image_keys_sha256": heldout["image_keys_sha256"],
                "normalized_paths_sha256": heldout["normalized_paths_sha256"],
                "count": len(heldout_keys),
            },
            "artifacts": artifact_records,
            "report": report,
        }
        atomic_write_json(staging_dir / "build_report.json", report)
        atomic_write_json(staging_dir / "build_manifest.json", manifest)
        for member_path in staging_dir.glob("inat_*_members.txt"):
            final_paths[member_path.name] = sources.output_dir / member_path.name
            staged_paths[member_path.name] = member_path
        staged_paths["build_report.json"] = staging_dir / "build_report.json"
        staged_paths["build_manifest.json"] = staging_dir / "build_manifest.json"
        final_paths["build_report.json"] = sources.output_dir / "build_report.json"
        final_paths["build_manifest.json"] = sources.output_dir / "build_manifest.json"
        _publish_artifact_set(staged_paths, final_paths)
        return manifest
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def build_corpus(sources: PipelineSources) -> dict[str, Any]:
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _build_corpus_impl(sources)
    except Exception as exc:
        atomic_write_json(
            sources.output_dir / "corpus_preflight_report.json",
            {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "stage": "corpus",
                "status": "failed",
                "created_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "partial_parquet_written": False,
                "expected_outputs": [
                    str(sources.output_dir / "train.parquet"),
                    str(sources.output_dir / "test.parquet"),
                ],
            },
        )
        raise


def build_indexes(sources: PipelineSources) -> dict[str, Any]:
    index_config = sources.raw.get("indexes") or {}
    vision_config = index_config.get("vision") or {}
    text_config = index_config.get("text") or {}
    vision_model = clean_text(vision_config.get("model_path"))
    text_model = clean_text(text_config.get("model_path"))
    if not vision_model or not text_model:
        raise ValueError("indexes.vision.model_path and indexes.text.model_path are required.")
    if not bool(index_config.get("allow_remote_models", False)):
        for label, model in (("vision", vision_model), ("text", text_model)):
            if not Path(model).expanduser().exists():
                raise FileNotFoundError(
                    f"Configured {label} model is not local: {model}. The pipeline never downloads models; "
                    "set a local path or explicitly set indexes.allow_remote_models=true."
                )

    from dual_search.search.index_builder import HybridTextIndexBuilder
    from dual_search.search.vision_index_builder import VisionIndexBuilder

    vision_corpus = sources.output_dir / "vision_corpus.jsonl"
    text_corpus = sources.output_dir / "text_corpus.jsonl"
    vision_dir = sources.output_dir / "indexes" / "vision"
    text_dir = sources.output_dir / "indexes" / "text"
    vision_builder = VisionIndexBuilder(
        model_path=vision_model,
        corpus_path=str(vision_corpus),
        save_dir=str(vision_dir),
        batch_size=int(vision_config.get("batch_size", 32)),
        faiss_type=vision_config.get("faiss_type", "Flat"),
        save_embedding=bool(vision_config.get("save_embedding", False)),
        faiss_gpu=bool(vision_config.get("faiss_gpu", False)),
        device=vision_config.get("device"),
        normalize_embeddings=not bool(vision_config.get("no_normalize", False)),
        truncate_dim=vision_config.get("truncate_dim"),
    )
    vision_builder.build_index()
    text_builder = HybridTextIndexBuilder(
        corpus_path=str(text_corpus),
        save_dir=str(text_dir),
        model_path=text_model,
        batch_size=int(text_config.get("batch_size", 16)),
        max_length=int(text_config.get("max_length", 8192)),
        device=text_config.get("device"),
        use_fp16=not bool(text_config.get("no_fp16", False)),
        faiss_type=text_config.get("faiss_type", "Flat"),
        faiss_gpu=bool(text_config.get("faiss_gpu", False)),
        save_embedding=bool(text_config.get("save_embedding", False)),
    )
    text_builder.build()
    report = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "index",
        "created_at": utc_now(),
        "vision_index_dir": str(vision_dir),
        "text_index_dir": str(text_dir),
    }
    atomic_write_json(sources.output_dir / "index_report.json", report)
    return report


def build_sft(sources: PipelineSources) -> dict[str, Any]:
    """Run the separately maintained SFT builder from the shared sources config."""

    from dual_search.data.sft_builder import (
        SFTBuilderConfig,
        TeacherConfig,
        VLLMTeacherClient,
        build_sft_files,
    )

    config = sources.raw.get("sft") or {}
    base_url = clean_text(config.get("teacher_base_url"))
    model = clean_text(config.get("teacher_model"))
    if not base_url or not model:
        raise ValueError("sft.teacher_base_url and sft.teacher_model are required for the SFT stage.")
    api_key_env = clean_text(config.get("teacher_api_key_env")) or "VLLM_API_KEY"
    teacher = VLLMTeacherClient(
        TeacherConfig(
            base_url=base_url,
            model=model,
            api_key=os.getenv(api_key_env),
            timeout_seconds=float(config.get("teacher_timeout", 120.0)),
            max_retries=int(config.get("teacher_max_retries", 2)),
            retry_backoff_seconds=float(config.get("teacher_retry_backoff", 1.0)),
            temperature=float(config.get("teacher_temperature", 0.0)),
            max_tokens=int(config.get("teacher_max_tokens", 768)),
        )
    )
    observation_tokenizer = None
    observation_tokenizer_path = clean_text(config.get("observation_tokenizer_path"))
    if observation_tokenizer_path:
        tokenizer_path = Path(observation_tokenizer_path).expanduser().resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(
                "sft.observation_tokenizer_path must be an existing local model directory: "
                f"{tokenizer_path}"
            )
        from transformers import AutoTokenizer

        observation_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=bool(config.get("observation_tokenizer_trust_remote_code", True)),
        )
    result = build_sft_files(
        train_parquet=sources.output_dir / "train.parquet",
        vision_corpus_path=sources.output_dir / "vision_corpus.jsonl",
        text_corpus_path=sources.output_dir / "text_corpus.jsonl",
        manifest_path=sources.output_dir / "build_manifest.json",
        output_dir=sources.output_dir,
        teacher=teacher,
        config=SFTBuilderConfig(
            sample_fraction=float(config.get("sample_fraction", 0.05)),
            validation_fraction=float(config.get("validation_fraction", 0.10)),
            seed=int(config.get("seed", sources.seed)),
            oracle_top_k=int(config.get("oracle_top_k", 3)),
            max_tool_response_tokens=int(config.get("max_tool_response_tokens", 500)),
            fallback_wrapper_token_reserve=int(config.get("fallback_wrapper_token_reserve", 32)),
        ),
        observation_tokenizer=observation_tokenizer,
    )
    return result.report


def run_stage(stage: str, sources: PipelineSources) -> dict[str, Any]:
    if stage == "catalog":
        return build_catalog(sources)
    if stage == "split":
        return build_split(sources)
    if stage == "corpus":
        return build_corpus(sources)
    if stage == "sft":
        return build_sft(sources)
    if stage == "index":
        return build_indexes(sources)
    if stage == "all":
        reports = {
            "catalog": build_catalog(sources),
            "split": build_split(sources),
            "corpus": build_corpus(sources),
            "sft": build_sft(sources),
            "index": build_indexes(sources),
        }
        return {"schema_version": PIPELINE_SCHEMA_VERSION, "stages": reports}
    raise ValueError(f"Unknown pipeline stage: {stage}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leak-free EVQA/iNaturalist corpora and datasets from local sources."
    )
    parser.add_argument("--config", type=Path, required=True, help="Local sources JSON file.")
    parser.add_argument("stage", choices=("catalog", "split", "corpus", "sft", "index", "all"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_stage(args.stage, load_sources(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
