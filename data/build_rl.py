"""Build the fixed 11K EVQA/iNaturalist RL dataset from local files.

This standalone CPU-only entrypoint never downloads data or runs a model.
Intermediate artifacts live under output_dir/.build_cache.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_fp_FINGERPRINT_SCHEMA_VERSION = 1

_fp_DEFAULT_ID_KEYS = ('id', 'sample_id', 'image_key')

def _fp_canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def _fp_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def _fp_sha256_file(path: str | Path, chunk_size: int=1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def _fp_stable_digest(value: Any) -> str:
    return _fp_sha256_bytes(_fp_canonical_json(value).encode('utf-8'))

def _fp_normalize_local_path(path: str | Path) -> str:
    """Return the canonical path spelling used for leak checks.

    ``strict=False`` is deliberate: catalog creation happens before images are
    necessarily extracted from their archives.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    return os.path.normcase(str(Path(expanded).resolve(strict=False)))

def _fp_iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8') as stream:
        for (line_no, line) in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'JSONL row {line_no} in {path} is not an object.')
            yield value

def _fp_row_identity(row: Mapping[str, Any], id_keys: Sequence[str]=_fp_DEFAULT_ID_KEYS) -> str:
    for key in id_keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f'Corpus row has no stable identity in keys {tuple(id_keys)}: {row!r}')

def _fp_id_order_sha256(rows: Iterable[Mapping[str, Any]], id_keys: Sequence[str]=_fp_DEFAULT_ID_KEYS) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    seen: set[str] = set()
    for row in rows:
        identity = _fp_row_identity(row, id_keys=id_keys)
        if identity in seen:
            raise ValueError(f'Duplicate corpus identity: {identity}')
        seen.add(identity)
        digest.update(identity.encode('utf-8'))
        digest.update(b'\n')
        count += 1
    return (digest.hexdigest(), count)

def _fp_corpus_fingerprint(path: str | Path, *, id_keys: Sequence[str]=_fp_DEFAULT_ID_KEYS) -> dict[str, Any]:
    path = Path(path)
    (order_digest, count) = _fp_id_order_sha256(_fp_iter_jsonl(path), id_keys=id_keys)
    return {'fingerprint_schema_version': _fp_FINGERPRINT_SCHEMA_VERSION, 'sha256': _fp_sha256_file(path), 'id_order_sha256': order_digest, 'row_count': count}

_ep_PIPELINE_SCHEMA_VERSION = 2

_ep_SUPPORTED_DATASET = 'inaturalist'

_ep_DEFAULT_PROMPT_TEMPLATE = '{image_block}\nImages are numbered from 1 in the order shown above.\nAnswer the question about the images. You may use the available vision_search and search tools when external evidence is needed, but call at most one tool in each assistant turn. Reason inside <think>...</think> and return the final answer inside <answer>...</answer>.\nQuestion: {question}'

def _ep_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ep_normalize_url(url: Any) -> str:
    value = _ep_clean_text(url)
    if value.startswith('http://'):
        value = 'https://' + value[len('http://'):]
    return value.rstrip('/')

def _ep_clean_text(value: Any) -> str:
    if value is None:
        return ''
    value = str(value).strip()
    return '' if value.lower() == 'nan' else value

def _ep_split_delimited(value: Any, delimiter: str='|') -> list[str]:
    return [part.strip() for part in _ep_clean_text(value).split(delimiter) if part.strip()]

def _ep_deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = _ep_clean_text(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result

def _ep_parse_answers(row: Mapping[str, Any]) -> list[str]:
    answers = _ep_split_delimited(row.get('answer'), '|')
    multi_answer = _ep_clean_text(row.get('multi_answer'))
    if multi_answer and multi_answer.lower() not in {'0', '1', 'false', 'true'}:
        answers.extend(_ep_split_delimited(multi_answer, '|'))
    return _ep_deduplicate((' && '.join(_ep_split_delimited(answer, '&&')) for answer in answers))

def _ep_row_wiki_pairs(row: Mapping[str, Any]) -> list[dict[str, str]]:
    titles = _ep_split_delimited(row.get('wikipedia_title'))
    urls = _ep_split_delimited(row.get('wikipedia_url'))
    pairs: list[dict[str, str]] = []
    for (index, url) in enumerate(urls):
        title = titles[index] if index < len(titles) else titles[0] if titles else ''
        pairs.append({'title': title, 'url': _ep_clean_text(url), 'normalized_url': _ep_normalize_url(url)})
    return pairs

@dataclass(frozen=True)
class _ep_FileSource:
    path: Path
    sha256: str | None = None

    def verify(self, label: str) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f'Configured {label} does not exist: {self.path}')
        if self.sha256:
            actual = _fp_sha256_file(self.path)
            if actual.lower() != self.sha256.lower():
                raise ValueError(f'Checksum mismatch for {label}: expected {self.sha256}, got {actual}.')

def _ep_source_fingerprint(source: _ep_FileSource, label: str) -> dict[str, Any]:
    if not source.path.is_file():
        raise FileNotFoundError(f'Configured {label} does not exist: {source.path}')
    actual = _fp_sha256_file(source.path)
    if source.sha256 and actual.lower() != source.sha256.lower():
        raise ValueError(f'Checksum mismatch for {label}: expected {source.sha256}, got {actual}.')
    return {'path': str(source.path), 'sha256': actual, 'configured_sha256': source.sha256}

def _ep_consumed_input_fingerprints(sources: 'PipelineSources') -> dict[str, Any]:
    inputs: dict[str, Any] = {'evqa_train_csv': _ep_source_fingerprint(sources.evqa_train, 'evqa.train_csv'), 'evqa_val_csv': _ep_source_fingerprint(sources.evqa_val, 'evqa.val_csv'), 'inaturalist_train_metadata': _ep_source_fingerprint(sources.inat_train.metadata, 'inaturalist.train.metadata'), 'inaturalist_val_metadata': _ep_source_fingerprint(sources.inat_val.metadata, 'inaturalist.val.metadata'), 'wikipedia_kb': _ep_source_fingerprint(sources.wikipedia_kb, 'wikipedia.kb')}
    if sources.inat_train.archive is not None:
        inputs['inaturalist_train_archive'] = _ep_source_fingerprint(sources.inat_train.archive, 'inaturalist.train.archive')
    if sources.inat_val.archive is not None:
        inputs['inaturalist_val_archive'] = _ep_source_fingerprint(sources.inat_val.archive, 'inaturalist.val.archive')
    return inputs

@dataclass(frozen=True)
class _ep_InatSplitSource:
    metadata: _ep_FileSource
    image_root: Path
    archive: _ep_FileSource | None = None

@dataclass(frozen=True)
class _ep_PipelineSources:
    config_path: Path
    output_dir: Path
    evqa_train: _ep_FileSource
    evqa_val: _ep_FileSource
    inat_train: _ep_InatSplitSource
    inat_val: _ep_InatSplitSource
    wikipedia_kb: _ep_FileSource
    seed: int
    raw: dict[str, Any]

    @property
    def catalog_path(self) -> Path:
        return self.output_dir / 'catalog.sqlite'

def _ep_resolve_config_path(base: Path, value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded.resolve(strict=False) if expanded.is_absolute() else (base / expanded).resolve(strict=False)

def _ep_file_source(value: Any, *, base: Path, label: str, required: bool=True) -> _ep_FileSource | None:
    if value in (None, ''):
        if required:
            raise ValueError(f'Missing required sources config entry: {label}')
        return None
    if isinstance(value, str):
        return _ep_FileSource(_ep_resolve_config_path(base, value))
    if not isinstance(value, dict) or not value.get('path'):
        raise ValueError(f'{label} must be a path string or an object with a path field.')
    return _ep_FileSource(_ep_resolve_config_path(base, str(value['path'])), _ep_clean_text(value.get('sha256')) or None)

def _ep_load_sources(config_path: str | Path) -> _ep_PipelineSources:
    config_path = Path(config_path).resolve(strict=True)
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Sources config must be a JSON object.')
    base = config_path.parent
    evqa = raw.get('evqa') or {}
    inat = raw.get('inaturalist') or {}
    wikipedia = raw.get('wikipedia') or {}

    def inat_source(split: str) -> _ep_InatSplitSource:
        block = inat.get(split) or {}
        metadata = _ep_file_source(block.get('metadata'), base=base, label=f'inaturalist.{split}.metadata')
        archive = _ep_file_source(block.get('archive'), base=base, label=f'inaturalist.{split}.archive', required=False)
        image_root_value = block.get('image_root')
        if not image_root_value:
            raise ValueError(f'Missing required sources config entry: inaturalist.{split}.image_root')
        return _ep_InatSplitSource(metadata=metadata, archive=archive, image_root=_ep_resolve_config_path(base, str(image_root_value)))
    output_value = raw.get('output_dir')
    if not output_value:
        raise ValueError('Missing required sources config entry: output_dir')
    kb_value = wikipedia.get('kb') or wikipedia.get('kb_json') or wikipedia.get('kb_zip')
    return _ep_PipelineSources(config_path=config_path, output_dir=_ep_resolve_config_path(base, str(output_value)), evqa_train=_ep_file_source(evqa.get('train_csv'), base=base, label='evqa.train_csv'), evqa_val=_ep_file_source(evqa.get('val_csv'), base=base, label='evqa.val_csv'), inat_train=inat_source('train'), inat_val=inat_source('val'), wikipedia_kb=_ep_file_source(kb_value, base=base, label='wikipedia.kb'), seed=int(raw.get('seed', 42)), raw=raw)

def _ep_source_for_split(sources: _ep_PipelineSources, split: str) -> _ep_InatSplitSource:
    if split == 'train':
        return sources.inat_train
    if split == 'val':
        return sources.inat_val
    raise ValueError(f'Unsupported iNaturalist source split: {split}')

def _ep_atomic_replace_writer(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (fd, temp_name) = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        write_fn(temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def _ep_atomic_write_json(path: Path, value: Any) -> None:
    _ep_atomic_replace_writer(path, lambda temp: temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'))

def _ep_atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0

    def write(temp: Path) -> None:
        nonlocal count
        with temp.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                count += 1
    _ep_atomic_replace_writer(path, write)
    return count

def _ep_publish_artifact_set(staged_paths: Mapping[str, Path], final_paths: Mapping[str, Path]) -> None:
    """Publish a related artifact set with best-effort transactional rollback.

    POSIX has no atomic rename for multiple files. Move any previous generation
    aside first, publish every staged member, and restore the complete previous
    generation if one replacement fails. The manifest/report must be ordered
    last by callers so consumers never receive a commit marker for an
    incomplete generation.
    """
    if list(staged_paths) != list(final_paths):
        raise ValueError('staged and final artifact sets must have the same ordered keys')
    for (name, staged) in staged_paths.items():
        if not staged.is_file():
            raise FileNotFoundError(f'staged artifact {name!r} is missing: {staged}')
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for (name, final) in final_paths.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            if not final.exists():
                continue
            (descriptor, backup_name) = tempfile.mkstemp(prefix=f'.{final.name}.', suffix='.backup', dir=final.parent)
            os.close(descriptor)
            backup = Path(backup_name)
            backup.unlink()
            os.replace(final, backup)
            backups[name] = backup
        for (name, staged) in staged_paths.items():
            os.replace(staged, final_paths[name])
            published.append(name)
    except Exception as publish_error:
        rollback_errors: list[str] = []
        for name in reversed(published):
            try:
                final_paths[name].unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f'remove {final_paths[name]}: {exc}')
        for (name, backup) in reversed(list(backups.items())):
            try:
                if backup.exists():
                    os.replace(backup, final_paths[name])
            except OSError as exc:
                rollback_errors.append(f'restore {final_paths[name]}: {exc}')
        if rollback_errors:
            raise RuntimeError('Artifact publication failed and rollback was incomplete: ' + '; '.join(rollback_errors)) from publish_error
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)

def _ep_read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as stream:
        for (line_no, line) in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'Expected object at {path}:{line_no}')
            rows.append(value)
    return rows

class _ep_JsonCharReader:
    """Small streaming JSON reader used for multi-gigabyte top-level objects."""

    def __init__(self, stream: TextIO, chunk_size: int=1 << 20):
        self.stream = stream
        self.chunk_size = chunk_size
        self.buffer = ''
        self.position = 0
        self.pending: list[str] = []

    def read_char(self) -> str:
        if self.pending:
            return self.pending.pop()
        if self.position >= len(self.buffer):
            self.buffer = self.stream.read(self.chunk_size)
            self.position = 0
            if not self.buffer:
                return ''
        char = self.buffer[self.position]
        self.position += 1
        return char

    def unread(self, char: str) -> None:
        if char:
            self.pending.append(char)

def _ep_skip_ws(reader: _ep_JsonCharReader) -> str:
    while True:
        char = reader.read_char()
        if not char or not char.isspace():
            return char

def _ep_read_raw_string(reader: _ep_JsonCharReader) -> str:
    output = ['"']
    escaped = False
    while True:
        char = reader.read_char()
        if not char:
            raise ValueError('Unexpected EOF in JSON string.')
        output.append(char)
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '"':
            return ''.join(output)

def _ep_collect_json_value(reader: _ep_JsonCharReader, first: str) -> str:
    if first == '"':
        return _ep_read_raw_string(reader)
    output = [first]
    if first not in '[{':
        while True:
            char = reader.read_char()
            if not char or char in ',}]':
                reader.unread(char)
                return ''.join(output).strip()
            output.append(char)
    stack = [first]
    while stack:
        char = reader.read_char()
        if not char:
            raise ValueError('Unexpected EOF in JSON value.')
        output.append(char)
        if char == '"':
            raw_tail = _ep_read_raw_string(reader)
            output.append(raw_tail[1:])
        elif char in '[{':
            stack.append(char)
        elif char == '}' and stack[-1] == '{':
            stack.pop()
        elif char == ']' and stack[-1] == '[':
            stack.pop()
    return ''.join(output)

def _ep_skip_json_value(reader: _ep_JsonCharReader, first: str) -> None:
    _ep_collect_json_value(reader, first)

def _ep_iter_array_values(reader: _ep_JsonCharReader) -> Iterator[dict[str, Any]]:
    char = _ep_skip_ws(reader)
    if char == ']':
        return
    while True:
        value = json.loads(_ep_collect_json_value(reader, char))
        if not isinstance(value, dict):
            raise ValueError('Expected objects inside metadata array.')
        yield value
        delimiter = _ep_skip_ws(reader)
        if delimiter == ']':
            return
        if delimiter != ',':
            raise ValueError(f"Expected ',' or ']' in JSON array, got {delimiter!r}.")
        char = _ep_skip_ws(reader)

def _ep_iter_top_level_arrays(stream: TextIO, selected_keys: set[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    reader = _ep_JsonCharReader(stream)
    if _ep_skip_ws(reader) != '{':
        raise ValueError('Metadata JSON must be a top-level object.')
    while True:
        char = _ep_skip_ws(reader)
        if char == '}':
            return
        if char != '"':
            raise ValueError(f'Expected top-level JSON key, got {char!r}.')
        key = json.loads(_ep_read_raw_string(reader))
        if _ep_skip_ws(reader) != ':':
            raise ValueError("Expected ':' after top-level JSON key.")
        first = _ep_skip_ws(reader)
        if key in selected_keys:
            if first != '[':
                raise ValueError(f'Expected top-level {key!r} to be an array.')
            for value in _ep_iter_array_values(reader):
                yield (key, value)
        else:
            _ep_skip_json_value(reader, first)
        delimiter = _ep_skip_ws(reader)
        if delimiter == '}':
            return
        if delimiter != ',':
            raise ValueError(f"Expected ',' or '}}', got {delimiter!r}.")

def _ep_create_catalog_schema(connection: sqlite3.Connection) -> None:
    connection.executescript('\n        CREATE TABLE categories (\n            category_id TEXT PRIMARY KEY,\n            category_json TEXT NOT NULL\n        );\n        CREATE TABLE image_categories (\n            image_key TEXT PRIMARY KEY,\n            category_id TEXT NOT NULL\n        );\n        CREATE TABLE images (\n            image_key TEXT PRIMARY KEY,\n            image_id TEXT NOT NULL,\n            source_split TEXT NOT NULL,\n            file_name TEXT NOT NULL,\n            normalized_path TEXT NOT NULL,\n            category_id TEXT,\n            category_key TEXT,\n            category_json TEXT\n        );\n        CREATE UNIQUE INDEX images_split_id ON images(source_split, image_id);\n        CREATE INDEX images_category ON images(category_id);\n        CREATE INDEX images_normalized_path ON images(normalized_path);\n        ')

def _ep_stream_metadata_into_catalog(connection: sqlite3.Connection, source_split: str, source: _ep_InatSplitSource) -> dict[str, int]:
    counters: Counter[str] = Counter()
    with source.metadata.path.open('r', encoding='utf-8') as stream:
        for (key, item) in _ep_iter_top_level_arrays(stream, {'images', 'annotations', 'categories'}):
            if key == 'categories':
                category_id = _ep_clean_text(item.get('id'))
                if not category_id:
                    raise ValueError(f'Invalid {source_split} category without id: {item!r}')
                category_json = _fp_canonical_json(item)
                existing = connection.execute('SELECT category_json FROM categories WHERE category_id = ?', (category_id,)).fetchone()
                if existing and existing[0] != category_json:
                    raise ValueError(f'Conflicting iNaturalist category metadata for {category_id}.')
                connection.execute('INSERT OR IGNORE INTO categories(category_id, category_json) VALUES (?, ?)', (category_id, category_json))
            elif key == 'annotations':
                image_id = _ep_clean_text(item.get('image_id'))
                category_id = _ep_clean_text(item.get('category_id'))
                if not image_id or not category_id:
                    raise ValueError(f'Invalid {source_split} annotation without image/category id: {item!r}')
                image_key = f'inaturalist:{image_id}'
                existing = connection.execute('SELECT category_id FROM image_categories WHERE image_key = ?', (image_key,)).fetchone()
                if existing and existing[0] != category_id:
                    raise ValueError(f'Conflicting iNaturalist categories for {image_key}.')
                connection.execute('INSERT OR REPLACE INTO image_categories(image_key, category_id) VALUES (?, ?)', (image_key, category_id))
            else:
                image_id = _ep_clean_text(item.get('id'))
                file_name = _ep_clean_text(item.get('file_name'))
                if not image_id or not file_name:
                    raise ValueError(f'Invalid {source_split} image without id/file_name: {item!r}')
                (file_name, normalized_path) = _ep_safe_image_destination(source.image_root, file_name)
                image_key = f'inaturalist:{image_id}'
                try:
                    connection.execute('\n                        INSERT INTO images(\n                            image_key, image_id, source_split, file_name, normalized_path\n                        ) VALUES (?, ?, ?, ?, ?)\n                        ', (image_key, image_id, source_split, file_name, normalized_path))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f'Duplicate iNaturalist image identity: {image_key}') from exc
            counters[key] += 1
            if sum(counters.values()) % 10000 == 0:
                connection.commit()
    connection.commit()
    return dict(counters)

def _ep_build_catalog_impl(sources: _ep_PipelineSources) -> dict[str, Any]:
    catalog_inputs = {'inaturalist_train_metadata': _ep_source_fingerprint(sources.inat_train.metadata, 'inaturalist.train.metadata'), 'inaturalist_val_metadata': _ep_source_fingerprint(sources.inat_val.metadata, 'inaturalist.val.metadata')}
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    (fd, temp_name) = tempfile.mkstemp(prefix='.catalog.', suffix='.sqlite', dir=sources.output_dir)
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink()
    try:
        connection = sqlite3.connect(temp_path)
        try:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=NORMAL')
            _ep_create_catalog_schema(connection)
            split_reports = {'train': _ep_stream_metadata_into_catalog(connection, 'train', sources.inat_train), 'val': _ep_stream_metadata_into_catalog(connection, 'val', sources.inat_val)}
            connection.execute('\n                UPDATE images\n                SET category_id = (\n                        SELECT category_id FROM image_categories\n                        WHERE image_categories.image_key = images.image_key\n                    )\n                ')
            connection.execute("UPDATE images SET category_key = 'inaturalist:' || category_id WHERE category_id IS NOT NULL")
            connection.execute('\n                UPDATE images\n                SET category_json = (\n                    SELECT category_json FROM categories\n                    WHERE categories.category_id = images.category_id\n                )\n                ')
            connection.commit()
            image_count = int(connection.execute('SELECT COUNT(*) FROM images').fetchone()[0])
            uncategorized = int(connection.execute('SELECT COUNT(*) FROM images WHERE category_id IS NULL').fetchone()[0])
            category_count = int(connection.execute('SELECT COUNT(*) FROM categories').fetchone()[0])
            orphan_annotations = int(connection.execute('\n                    SELECT COUNT(*) FROM image_categories\n                    LEFT JOIN images USING(image_key)\n                    WHERE images.image_key IS NULL\n                    ').fetchone()[0])
            unknown_categories = int(connection.execute('SELECT COUNT(*) FROM images WHERE category_id IS NOT NULL AND category_json IS NULL').fetchone()[0])
            if uncategorized or orphan_annotations or unknown_categories:
                raise ValueError(f'iNaturalist metadata is incomplete: uncategorized_images={uncategorized}, orphan_annotations={orphan_annotations}, unknown_categories={unknown_categories}.')
            connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            connection.execute('PRAGMA journal_mode=DELETE')
        finally:
            connection.close()
        os.replace(temp_path, sources.catalog_path)
    finally:
        for suffix in ('', '-wal', '-shm'):
            candidate = Path(str(temp_path) + suffix)
            if candidate.exists():
                candidate.unlink()
    report = {'schema_version': _ep_PIPELINE_SCHEMA_VERSION, 'stage': 'catalog', 'created_at': _ep_utc_now(), 'catalog_path': str(sources.catalog_path), 'catalog_sha256': _fp_sha256_file(sources.catalog_path), 'images': image_count, 'categories': category_count, 'uncategorized_images': uncategorized, 'splits': split_reports, 'input_sha256': {'inaturalist.train.metadata': catalog_inputs['inaturalist_train_metadata']['sha256'], 'inaturalist.val.metadata': catalog_inputs['inaturalist_val_metadata']['sha256']}, 'inputs': catalog_inputs}
    _ep_atomic_write_json(sources.output_dir / 'catalog_manifest.json', report)
    return report

def _ep_build_catalog(sources: _ep_PipelineSources) -> dict[str, Any]:
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _ep_build_catalog_impl(sources)
    except Exception as exc:
        _ep_atomic_write_json(sources.output_dir / 'catalog_preflight_report.json', {'schema_version': _ep_PIPELINE_SCHEMA_VERSION, 'stage': 'catalog', 'status': 'failed', 'created_at': _ep_utc_now(), 'error_type': type(exc).__name__, 'error': str(exc), 'inputs': {'inaturalist_train_metadata': str(sources.inat_train.metadata.path), 'inaturalist_val_metadata': str(sources.inat_val.metadata.path)}, 'partial_catalog_written': False})
        raise

def _ep_lookup_query_image(connection: sqlite3.Connection, image_id: str, expected_split: str) -> dict[str, Any] | None:
    row = connection.execute('\n        SELECT image_key, image_id, source_split, file_name, normalized_path,\n               category_id, category_key, category_json\n        FROM images WHERE source_split = ? AND image_id = ?\n        ', (expected_split, image_id)).fetchone()
    if not row:
        return None
    keys = ('image_key', 'image_id', 'source_split', 'file_name', 'normalized_path', 'category_id', 'category_key', 'category_json')
    value = dict(zip(keys, row))
    value['category'] = json.loads(value.pop('category_json') or '{}')
    return value

def _ep_sample_id(row: Mapping[str, Any], official_split: str, source_row_index: int, image_keys: list[str]) -> str:
    question_id = _ep_clean_text(row.get('question_id') or row.get('id'))
    identity = {'official_split': official_split, 'question_id': question_id or f'source-row:{source_row_index}', 'question': _ep_clean_text(row.get('question')), 'question_original': _ep_clean_text(row.get('question_original')), 'question_type': _ep_clean_text(row.get('question_type')), 'answer': _ep_clean_text(row.get('answer')), 'multi_answer': _ep_clean_text(row.get('multi_answer')), 'evidence': _ep_clean_text(row.get('evidence')), 'evidence_section_id': _ep_clean_text(row.get('evidence_section_id')), 'evidence_section_title': _ep_clean_text(row.get('evidence_section_title')), 'wikipedia_url': [pair['normalized_url'] for pair in _ep_row_wiki_pairs(row)], 'dataset_category_id': _ep_clean_text(row.get('dataset_category_id')), 'image_keys': image_keys}
    return f'evqa:{_fp_stable_digest(identity)[:24]}'

def _ep_logical_sample(row: Mapping[str, Any], *, official_split: str, source_row_index: int, images: list[Mapping[str, Any]]) -> dict[str, Any]:
    query_images = [{'image_index': image_index, 'dataset_image_id': str(image['image_id']), 'image_key': str(image['image_key']), 'image': str(image['normalized_path']), 'source_file_name': str(image['file_name']), 'source_split': str(image['source_split'])} for (image_index, image) in enumerate(images, start=1)]
    image_keys = [image['image_key'] for image in query_images]
    sample_id = _ep_sample_id(row, official_split, source_row_index, image_keys)
    category_id = str(images[0]['category_id'] or '')
    category_key = str(images[0]['category_key'] or '')
    return {'schema_version': _ep_PIPELINE_SCHEMA_VERSION, 'sample_id': sample_id, 'question_id': _ep_clean_text(row.get('question_id') or row.get('id')), 'official_split': official_split, 'source_row_index': source_row_index, 'dataset_name': _ep_SUPPORTED_DATASET, 'question': _ep_clean_text(row.get('question')), 'question_original': _ep_clean_text(row.get('question_original')), 'question_type': _ep_clean_text(row.get('question_type')), 'answer': _ep_clean_text(row.get('answer')), 'answers': _ep_parse_answers(row), 'multi_answer': _ep_clean_text(row.get('multi_answer')), 'evidence': _ep_clean_text(row.get('evidence')), 'evidence_section_id': _ep_clean_text(row.get('evidence_section_id')), 'evidence_section_title': _ep_clean_text(row.get('evidence_section_title')), 'wikipedia_title': _ep_clean_text(row.get('wikipedia_title')), 'wikipedia_url': _ep_clean_text(row.get('wikipedia_url')), 'wiki_pairs': _ep_row_wiki_pairs(row), 'query_images': query_images, 'image_keys': image_keys, 'image_count': len(query_images), 'dataset_category_id': category_id, 'category_key': category_key}

def _ep_require_schema_v2(value: Mapping[str, Any], label: str) -> None:
    if value.get('schema_version') != _ep_PIPELINE_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible {label} schema_version={value.get('schema_version')!r}; "
            "rerun data/build_rl.py."
        )

def _ep_catalog_rows_for_categories(catalog_path: Path, category_ids: set[str]) -> Iterator[dict[str, Any]]:
    if not category_ids:
        return
    connection = sqlite3.connect(catalog_path)
    try:
        keys = ('image_key', 'image_id', 'source_split', 'file_name', 'normalized_path', 'category_id', 'category_key', 'category_json')
        ordered_category_ids = sorted((str(value) for value in category_ids))
        for offset in range(0, len(ordered_category_ids), 500):
            chunk = ordered_category_ids[offset:offset + 500]
            placeholders = ','.join(('?' for _ in chunk))
            cursor = connection.execute(f'\n                SELECT image_key, image_id, source_split, file_name,\n                       normalized_path, category_id, category_key, category_json\n                FROM images\n                WHERE category_id IN ({placeholders})\n                ORDER BY category_id, source_split, image_id\n                ', chunk)
            for raw in cursor:
                row = dict(zip(keys, raw))
                row['category'] = json.loads(row.pop('category_json') or '{}')
                yield row
    finally:
        connection.close()

def _ep_valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False

def _ep_safe_archive_name(name: str) -> str:
    normalized = _ep_clean_text(name).replace('\\', '/')
    while normalized.startswith('./'):
        normalized = normalized[2:]
    normalized = normalized.rstrip('/')
    if not normalized or '\x00' in normalized or normalized.startswith('/'):
        raise ValueError(f'Unsafe archive member name: {name!r}')
    raw_parts = normalized.split('/')
    if any((part in {'', '.', '..'} for part in raw_parts)):
        raise ValueError(f'Unsafe archive member name: {name!r}')
    path = PurePosixPath(normalized)
    if path.is_absolute() or '..' in path.parts or ':' in raw_parts[0]:
        raise ValueError(f'Unsafe archive member name: {name!r}')
    return path.as_posix()

def _ep_safe_image_destination(image_root: Path, member_name: str) -> tuple[str, str]:
    safe_name = _ep_safe_archive_name(member_name)
    root = image_root.resolve(strict=False)
    target = (root / PurePosixPath(safe_name)).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Image member resolves outside configured image_root: {member_name!r}') from exc
    return (safe_name, _fp_normalize_local_path(target))

def _ep_extract_indexed_images(archive_path: Path, image_root: Path, source_split: str, connection: sqlite3.Connection) -> int:
    """Extract selected members using a disk-backed lookup table."""
    extracted = 0
    with tarfile.open(archive_path, mode='r:*') as archive:
        for member in archive:
            if not member.isfile() and member.name in {'.', './'}:
                continue
            normalized = _ep_safe_archive_name(member.name)
            if not member.isfile():
                continue
            found = connection.execute('\n                SELECT target_path FROM members\n                WHERE source_split = ? AND file_name = ? AND needs_extract = 1\n                ', (source_split, normalized)).fetchone()
            if not found:
                continue
            (_, expected_target) = _ep_safe_image_destination(image_root, normalized)
            if _fp_normalize_local_path(found[0]) != expected_target:
                raise ValueError(f'Catalog target path does not match archive member {normalized!r}.')
            target = Path(expected_target)
            raw = archive.extractfile(member)
            if raw is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            (fd, temp_name) = tempfile.mkstemp(prefix=f'.{target.name}.', dir=target.parent)
            try:
                with os.fdopen(fd, 'wb') as output:
                    shutil.copyfileobj(raw, output, length=1 << 20)
                os.replace(temp_name, target)
            finally:
                raw.close()
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            connection.execute('UPDATE members SET needs_extract = 0 WHERE source_split = ? AND file_name = ?', (source_split, normalized))
            extracted += 1
    connection.commit()
    return extracted

@contextmanager
def _ep_open_kb_json(source: _ep_FileSource) -> Iterator[TextIO]:
    if not source.path.is_file():
        raise FileNotFoundError(f'Configured wikipedia.kb does not exist: {source.path}')
    if zipfile.is_zipfile(source.path):
        with zipfile.ZipFile(source.path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith('.json')]
            if len(names) != 1:
                raise ValueError('Wikipedia KB zip must contain exactly one JSON file.')
            with archive.open(names[0]) as raw:
                with io.TextIOWrapper(raw, encoding='utf-8') as stream:
                    yield stream
    else:
        with source.path.open('r', encoding='utf-8') as stream:
            yield stream

def _ep_iter_selected_kb_pages(stream: TextIO, selected_urls: set[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    reader = _ep_JsonCharReader(stream)
    if _ep_skip_ws(reader) != '{':
        raise ValueError('Wikipedia KB JSON must be a top-level object.')
    while True:
        char = _ep_skip_ws(reader)
        if char == '}':
            return
        if char != '"':
            raise ValueError(f'Expected Wikipedia URL key, got {char!r}.')
        url = json.loads(_ep_read_raw_string(reader))
        if _ep_skip_ws(reader) != ':':
            raise ValueError("Expected ':' after Wikipedia URL.")
        first = _ep_skip_ws(reader)
        if _ep_normalize_url(url) in selected_urls:
            page = json.loads(_ep_collect_json_value(reader, first))
            if isinstance(page, dict):
                yield (url, page)
        else:
            _ep_skip_json_value(reader, first)
        delimiter = _ep_skip_ws(reader)
        if delimiter == '}':
            return
        if delimiter != ',':
            raise ValueError(f"Expected ',' or '}}' in KB JSON, got {delimiter!r}.")

def _ep_build_text_corpus(query_rows: list[dict[str, Any]], kb_source: _ep_FileSource) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, Any]]:
    selected_urls = {pair['normalized_url'] for row in query_rows for pair in row.get('wiki_pairs', []) if pair.get('normalized_url')}
    sections_by_url: dict[str, set[str]] = defaultdict(set)
    text_rows: list[dict[str, Any]] = []
    found_urls: set[str] = set()
    with _ep_open_kb_json(kb_source) as stream:
        for (original_url, page) in _ep_iter_selected_kb_pages(stream, selected_urls):
            normalized_url = _ep_normalize_url(original_url)
            found_urls.add(normalized_url)
            title = _ep_clean_text(page.get('title')) or original_url.rsplit('/', 1)[-1]
            section_texts = page.get('section_texts') or []
            section_titles = page.get('section_titles') or []
            for (section_id, text) in enumerate(section_texts):
                text = _ep_clean_text(text)
                if not text:
                    continue
                section_title = _ep_clean_text(section_titles[section_id] if section_id < len(section_titles) else '')
                display_title = title if not section_title else f'{title} :: {section_title}'
                text_rows.append({'id': f'{normalized_url}#section-{section_id}', 'url': normalized_url, 'source_url': original_url, 'title': title, 'section_id': section_id, 'section_title': section_title, 'contents': f'"{display_title}"\n{text}'})
                sections_by_url[normalized_url].add(str(section_id))
    text_rows.sort(key=lambda row: row['id'])
    return (text_rows, sections_by_url, {'requested_wiki_urls': len(selected_urls), 'found_wiki_urls': len(found_urls), 'missing_wiki_urls': sorted(selected_urls - found_urls), 'text_sections': len(text_rows)})

def _ep_is_text_resolvable(row: Mapping[str, Any], sections_by_url: Mapping[str, set[str]]) -> bool:
    urls = [pair.get('normalized_url', '') for pair in row.get('wiki_pairs', [])]
    if not urls or any((not sections_by_url.get(url) for url in urls)):
        return False
    required_ids = _ep_split_delimited(row.get('evidence_section_id'))
    if not required_ids:
        return True
    if len(required_ids) == len(urls):
        return all((section_id in sections_by_url[url] for (url, section_id) in zip(urls, required_ids)))
    if len(urls) == 1:
        return all((section_id in sections_by_url[urls[0]] for section_id in required_ids))
    return False

def _ep_vision_corpus_row(row: Mapping[str, Any]) -> dict[str, Any]:
    category = row.get('category') or {}
    scientific_name = _ep_clean_text(category.get('name'))
    common_name = _ep_clean_text(category.get('common_name'))
    title = common_name or scientific_name or str(row['category_id'])
    taxonomy = ' > '.join((_ep_clean_text(category.get(key)) for key in ('kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'specific_epithet') if _ep_clean_text(category.get(key))))
    caption_parts = [f'Entity: {title}.']
    if scientific_name:
        caption_parts.append(f'Scientific name: {scientific_name}.')
    if common_name:
        caption_parts.append(f'Common name: {common_name}.')
    if taxonomy:
        caption_parts.append(f'Taxonomy: {taxonomy}.')
    caption = ' '.join(caption_parts)
    return {'id': row['image_key'], 'image_key': row['image_key'], 'category_key': row['category_key'], 'dataset_name': _ep_SUPPORTED_DATASET, 'image_id': row['image_id'], 'category_id': row['category_id'], 'source_split': row['source_split'], 'image': row['normalized_path'], 'source_file_name': row['file_name'], 'title': title, 'caption': caption, 'contents': f'"{title}"\n{caption}'}

def _ep_rl_sample(logical: Mapping[str, Any], *, positive_count: int, text_resolvable: bool) -> dict[str, Any]:
    vision_resolvable = positive_count > 0
    retrieval_resolvable = vision_resolvable and text_resolvable
    question = _ep_clean_text(logical.get('question'))
    answers = list(logical.get('answers') or [])
    extra_info = {**dict(logical), 'positive_candidate_count': positive_count, 'vision_resolvable': vision_resolvable, 'text_resolvable': text_resolvable, 'retrieval_resolvable': retrieval_resolvable}
    query_images = [dict(image) for image in logical['query_images']]
    image_block = '\n'.join((f"Image {image['image_index']}:\n<image>" for image in query_images))
    return {'schema_version': _ep_PIPELINE_SCHEMA_VERSION, 'data_source': 'dual_search', 'sample_id': logical['sample_id'], 'category_key': logical['category_key'], 'dataset_category_id': logical['dataset_category_id'], 'query_images': query_images, 'image_keys': list(logical['image_keys']), 'image_count': int(logical['image_count']), 'question': question, 'answer': logical.get('answer', ''), 'question_type': logical.get('question_type', ''), 'retrieval_resolvable': retrieval_resolvable, 'positive_candidate_count': positive_count, 'vision_resolvable': vision_resolvable, 'text_resolvable': text_resolvable, 'prompt': [{'role': 'user', 'content': _ep_DEFAULT_PROMPT_TEMPLATE.format(image_block=image_block, question=question)}], 'images': [{'image': image['image']} for image in query_images], 'ability': 'vision-search', 'reward_model': {'style': 'rule', 'ground_truth': {'target': answers}}, 'extra_info': extra_info}

def _ep_write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f'Refusing to create an empty RL dataset: {path.name}')
    for row in rows:
        _ep_require_schema_v2(row, f"RL sample {row.get('sample_id', '<missing>')}")
        if not row['reward_model']['ground_truth']['target']:
            raise ValueError(f"Sample {row['sample_id']} has no answer target.")
        if row.get('image_count') != len(row.get('query_images') or []):
            raise ValueError(f"Sample {row['sample_id']} has inconsistent image_count.")
        if row.get('image_keys') != [image.get('image_key') for image in row.get('query_images') or []]:
            raise ValueError(f"Sample {row['sample_id']} has inconsistent image_keys.")
        prompt_text = ''.join((message.get('content', '') for message in row['prompt']))
        if prompt_text.count('<image>') != len(row['images']):
            raise ValueError(f"Sample {row['sample_id']} has mismatched image placeholders.")
    import pandas as pd
    pd.DataFrame(rows).to_parquet(path, index=False)

def _ep_artifact_record(path: Path, *, jsonl: bool=False) -> dict[str, Any]:
    result = {'path': str(path), 'sha256': _fp_sha256_file(path)}
    if jsonl:
        result.update(_fp_corpus_fingerprint(path))
    return result

_rp_DEFAULT_PRIMARY_COUNTS = {'automatic': 10000, '2_hop': 1000}

_rp_DEFAULT_IMAGE_COUNT_WEIGHTS = {'1': 0.3, '2': 0.25, '3': 0.25, '4': 0.1, '5': 0.1}

_rp_DEFAULT_SUPERCATEGORIES = ('Plants', 'Insects', 'Birds')

_rp_QUESTION_TYPE_ALIASES = {'automatic': 'automatic', '2_hop': '2_hop', '2-hop': '2_hop', '2hop': '2_hop'}

_rp_TAXON_ALIASES = {'plant': 'Plants', 'plants': 'Plants', 'plantae': 'Plants', 'insect': 'Insects', 'insects': 'Insects', 'insecta': 'Insects', 'bird': 'Birds', 'birds': 'Birds', 'aves': 'Birds'}

@dataclass(frozen=True)
class _rp_PoolConfig:
    primary_counts: dict[str, int]
    image_count_weights: dict[int, float]
    supercategories: tuple[str, ...]
    reserve_fraction: float
    seed: int
    min_positive_candidates: int
    corpus_images_per_category: int

    @classmethod
    def from_sources(cls, sources: _ep_PipelineSources) -> 'PoolConfig':
        raw = sources.raw.get('rl_pool') or {}
        if not isinstance(raw, Mapping):
            raise ValueError('rl_pool must be a JSON object.')
        counts_value = raw.get('primary_counts', _rp_DEFAULT_PRIMARY_COUNTS)
        if not isinstance(counts_value, Mapping) or not counts_value:
            raise ValueError('rl_pool.primary_counts must be a non-empty object.')
        parsed_counts: dict[str, int] = {}
        for (raw_type, raw_count) in counts_value.items():
            question_type = _rp_canonical_question_type(raw_type)
            if not question_type:
                raise ValueError(f'rl_pool.primary_counts supports only automatic and 2_hop; got {raw_type!r}.')
            if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
                raise ValueError(f'rl_pool.primary_counts[{raw_type!r}] must be a non-negative integer.')
            if question_type in parsed_counts:
                raise ValueError(f'Duplicate canonical question type {question_type!r}.')
            parsed_counts[question_type] = raw_count
        primary_counts = {question_type: parsed_counts[question_type] for question_type in ('automatic', '2_hop') if question_type in parsed_counts}
        if sum(primary_counts.values()) <= 0:
            raise ValueError('rl_pool.primary_counts must request at least one row.')
        weights_value = raw.get('image_count_weights', _rp_DEFAULT_IMAGE_COUNT_WEIGHTS)
        if not isinstance(weights_value, Mapping) or not weights_value:
            raise ValueError('rl_pool.image_count_weights must be a non-empty object.')
        image_weights: dict[int, float] = {}
        for (raw_count, raw_weight) in weights_value.items():
            try:
                image_count = int(raw_count)
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError('rl_pool.image_count_weights keys must be image counts 1..5 and values must be numbers.') from exc
            if image_count not in range(1, 6):
                raise ValueError('rl_pool.image_count_weights only supports image counts 1..5.')
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('rl_pool.image_count_weights values must be finite and >= 0.')
            image_weights[image_count] = weight
        for image_count in range(1, 6):
            image_weights.setdefault(image_count, 0.0)
        if sum(image_weights.values()) <= 0:
            raise ValueError('rl_pool.image_count_weights must contain a positive weight.')
        categories_value = raw.get('supercategories', _rp_DEFAULT_SUPERCATEGORIES)
        if not isinstance(categories_value, Sequence) or isinstance(categories_value, (str, bytes)):
            raise ValueError('rl_pool.supercategories must be a list.')
        categories: list[str] = []
        for value in categories_value:
            canonical = _rp_TAXON_ALIASES.get(_ep_clean_text(value).lower())
            if canonical not in _rp_DEFAULT_SUPERCATEGORIES:
                raise ValueError('rl_pool.supercategories supports only Plants, Insects, and Birds.')
            if canonical in categories:
                raise ValueError(f'Duplicate rl_pool supercategory {canonical!r}.')
            categories.append(canonical)
        if not categories:
            raise ValueError('rl_pool.supercategories must not be empty.')
        reserve_fraction = float(raw.get('reserve_fraction', 0.25))
        if not math.isfinite(reserve_fraction) or not 0 <= reserve_fraction <= 1:
            raise ValueError('rl_pool.reserve_fraction must be finite and within 0..1.')
        seed = int(raw.get('seed', sources.seed))
        min_candidates = raw.get('min_positive_candidates', raw.get('future_min_positive_candidates', 8))
        max_candidates = raw.get('corpus_images_per_category', raw.get('future_corpus_max_per_category', 32))
        if isinstance(min_candidates, bool) or not isinstance(min_candidates, int) or min_candidates <= 0:
            raise ValueError('rl_pool.min_positive_candidates must be an integer > 0.')
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
            raise ValueError('rl_pool.corpus_images_per_category must be an integer > 0.')
        if min_candidates > max_candidates:
            raise ValueError('rl_pool.min_positive_candidates cannot exceed corpus_images_per_category.')
        return cls(primary_counts=primary_counts, image_count_weights=dict(sorted(image_weights.items())), supercategories=tuple(categories), reserve_fraction=reserve_fraction, seed=seed, min_positive_candidates=min_candidates, corpus_images_per_category=max_candidates)

    def as_dict(self) -> dict[str, Any]:
        return {'primary_counts': dict(self.primary_counts), 'image_count_weights': {str(key): value for (key, value) in self.image_count_weights.items()}, 'supercategories': list(self.supercategories), 'reserve_fraction': self.reserve_fraction, 'seed': self.seed, 'min_positive_candidates': self.min_positive_candidates, 'corpus_images_per_category': self.corpus_images_per_category, 'image_selection': 'csv_prefix'}

@dataclass(frozen=True)
class _rp_Candidate:
    row: dict[str, Any]
    source_row_index: int
    images: tuple[dict[str, Any], ...]
    parent: dict[str, Any]
    question_type: str
    coarse_taxon: str
    dedupe_key: str

class _rp_PoolCapacityError(RuntimeError):
    """Raised when a configured quota cannot be filled after allowed fallback."""

def _rp_initialize_candidate_spool(connection: sqlite3.Connection) -> None:
    """Create the disk-backed candidate spool used by the pool stage.

    Candidate payloads are intentionally kept as canonical JSON in SQLite.
    The real EVQA CSV can contain far more eligible rows than the 13,750 rows
    needed by the default train+reserve selection, so retaining every expanded
    Python object would make peak memory proportional to the input size.
    """
    connection.executescript('\n        PRAGMA journal_mode = OFF;\n        PRAGMA synchronous = OFF;\n        PRAGMA temp_store = FILE;\n        CREATE TABLE candidates (\n            dedupe_key TEXT PRIMARY KEY,\n            source_row_index INTEGER NOT NULL,\n            question_type TEXT NOT NULL,\n            coarse_taxon TEXT NOT NULL,\n            category_id TEXT NOT NULL,\n            parent_id TEXT NOT NULL,\n            tiebreak TEXT NOT NULL,\n            train_rank TEXT NOT NULL,\n            reserve_rank TEXT NOT NULL,\n            payload_json TEXT NOT NULL,\n            evidence_ok INTEGER NOT NULL DEFAULT 0\n        ) WITHOUT ROWID;\n        CREATE INDEX candidates_evidence_cell_train\n            ON candidates(\n                evidence_ok, question_type, coarse_taxon,\n                train_rank, tiebreak, source_row_index\n            );\n        CREATE INDEX candidates_evidence_cell_reserve\n            ON candidates(\n                evidence_ok, question_type, coarse_taxon,\n                reserve_rank, tiebreak, source_row_index\n            );\n        CREATE INDEX candidates_evidence_category\n            ON candidates(evidence_ok, category_id);\n        CREATE TABLE duplicate_rejections (\n            dedupe_key TEXT NOT NULL,\n            source_row_index INTEGER NOT NULL,\n            tiebreak TEXT NOT NULL\n        );\n        CREATE INDEX duplicate_rejections_order\n            ON duplicate_rejections(tiebreak, source_row_index);\n        CREATE TEMP TABLE banned_categories (\n            category_id TEXT PRIMARY KEY\n        ) WITHOUT ROWID;\n        CREATE TEMP TABLE used_parents (\n            parent_id TEXT PRIMARY KEY\n        ) WITHOUT ROWID;\n        ')

def _rp_candidate_payload(candidate: _rp_Candidate) -> str:
    return json.dumps({'row': candidate.row, 'source_row_index': candidate.source_row_index, 'images': list(candidate.images), 'parent': candidate.parent, 'question_type': candidate.question_type, 'coarse_taxon': candidate.coarse_taxon, 'dedupe_key': candidate.dedupe_key}, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def _rp_candidate_from_payload(payload_json: str) -> _rp_Candidate:
    value = json.loads(payload_json)
    return _rp_Candidate(row=dict(value['row']), source_row_index=int(value['source_row_index']), images=tuple((dict(image) for image in value['images'])), parent=dict(value['parent']), question_type=str(value['question_type']), coarse_taxon=str(value['coarse_taxon']), dedupe_key=str(value['dedupe_key']))

def _rp_iter_spooled_candidates(connection: sqlite3.Connection, *, evidence_ok: bool | None=None) -> Iterable[_rp_Candidate]:
    query = 'SELECT payload_json FROM candidates'
    parameters: tuple[Any, ...] = ()
    if evidence_ok is not None:
        query += ' WHERE evidence_ok = ?'
        parameters = (int(evidence_ok),)
    query += ' ORDER BY tiebreak, source_row_index'
    cursor = connection.execute(query, parameters)
    for (payload_json,) in cursor:
        yield _rp_candidate_from_payload(str(payload_json))

def _rp_set_banned_categories(connection: sqlite3.Connection, category_ids: Iterable[str]) -> None:
    connection.execute('DELETE FROM banned_categories')
    connection.executemany('INSERT INTO banned_categories(category_id) VALUES (?)', ((category_id,) for category_id in sorted(set(category_ids))))

def _rp_iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    """Stream EVQA rows instead of materializing the full CSV in memory."""
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            yield dict(row)

def _rp_canonical_question_type(value: Any) -> str:
    return _rp_QUESTION_TYPE_ALIASES.get(_ep_clean_text(value).lower(), '')

def _rp_coarse_taxon(category: Mapping[str, Any]) -> str:
    for field in ('coarse_taxon', 'supercategory', 'iconic_taxon_name'):
        canonical = _rp_TAXON_ALIASES.get(_ep_clean_text(category.get(field)).lower())
        if canonical:
            return canonical
    kingdom = _ep_clean_text(category.get('kingdom')).lower()
    class_name = _ep_clean_text(category.get('class')).lower()
    if kingdom == 'plantae':
        return 'Plants'
    if class_name == 'insecta':
        return 'Insects'
    if class_name == 'aves':
        return 'Birds'
    return ''

def _rp_apportion(total: int, weights: Mapping[Any, float], order: Sequence[Any]) -> dict[Any, int]:
    """Largest-remainder apportionment with deterministic configured-order ties."""
    if total < 0:
        raise ValueError('Cannot apportion a negative total.')
    weight_sum = sum((float(weights[key]) for key in order))
    if weight_sum <= 0:
        raise ValueError('Cannot apportion with zero total weight.')
    ideals = {key: total * float(weights[key]) / weight_sum for key in order}
    result = {key: math.floor(ideals[key]) for key in order}
    remaining = total - sum(result.values())
    order_index = {key: index for (index, key) in enumerate(order)}
    ranked = sorted(order, key=lambda key: (-(ideals[key] - result[key]), order_index[key]))
    for key in ranked[:remaining]:
        result[key] += 1
    return result

def _rp_balanced_category_totals(total: int, categories: Sequence[str], *, offset: int) -> dict[str, int]:
    (base, remainder) = divmod(total, len(categories))
    result = {category: base for category in categories}
    for index in range(remainder):
        result[categories[(offset + index) % len(categories)]] += 1
    return result

def _rp_quota_matrix(image_totals: Mapping[int, int], category_totals: Mapping[str, int], image_order: Sequence[int], category_order: Sequence[str]) -> dict[int, dict[str, int]]:
    """Build an integer matrix with exact image-count and category margins."""
    remaining_categories = dict(category_totals)
    remaining_total = sum(remaining_categories.values())
    matrix: dict[int, dict[str, int]] = {}
    for (row_index, image_count) in enumerate(image_order):
        row_total = image_totals[image_count]
        if row_index == len(image_order) - 1:
            if sum(remaining_categories.values()) != row_total:
                raise AssertionError('Quota matrix margins became inconsistent.')
            matrix[image_count] = dict(remaining_categories)
            break
        row_weights = {category: float(remaining_categories[category]) for category in category_order}
        if row_total:
            row = _rp_apportion(row_total, row_weights, category_order)
        else:
            row = {category: 0 for category in category_order}
        overflow = 0
        for category in category_order:
            if row[category] > remaining_categories[category]:
                overflow += row[category] - remaining_categories[category]
                row[category] = remaining_categories[category]
        while overflow:
            receivers = [category for category in category_order if row[category] < remaining_categories[category]]
            if not receivers:
                raise AssertionError('Unable to repair quota matrix overflow.')
            receiver = max(receivers, key=lambda category: (remaining_categories[category] - row[category], -category_order.index(category)))
            row[receiver] += 1
            overflow -= 1
        matrix[image_count] = row
        for category in category_order:
            remaining_categories[category] -= row[category]
        remaining_total -= row_total
        if remaining_total != sum(remaining_categories.values()):
            raise AssertionError('Quota matrix did not preserve its margins.')
    return matrix

def _rp_adjust_category_capacity(desired: Mapping[str, int], available: Mapping[str, int], category_order: Sequence[str]) -> dict[str, int]:
    adjusted = {category: min(desired[category], available.get(category, 0)) for category in category_order}
    deficit = sum(desired.values()) - sum(adjusted.values())
    while deficit:
        donors = [category for category in category_order if adjusted[category] < available.get(category, 0)]
        if not donors:
            raise _rp_PoolCapacityError(f'Configured pool quota exceeds eligible candidate capacity: requested={sum(desired.values())}, available={sum(available.values())}.')
        donor = max(donors, key=lambda category: (available.get(category, 0) - adjusted[category], -category_order.index(category)))
        adjusted[donor] += 1
        deficit -= 1
    return adjusted

def _rp_matrix_fallbacks(*, partition: str, question_type: str, initial: Mapping[int, Mapping[str, int]], actual: Mapping[int, Mapping[str, int]], image_order: Sequence[int], category_order: Sequence[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for image_count in image_order:
        losses: list[list[Any]] = []
        gains: list[list[Any]] = []
        for category in category_order:
            delta = actual[image_count][category] - initial[image_count][category]
            if delta < 0:
                losses.append([category, -delta])
            elif delta > 0:
                gains.append([category, delta])
        loss_index = gain_index = 0
        while loss_index < len(losses) and gain_index < len(gains):
            moved = min(losses[loss_index][1], gains[gain_index][1])
            events.append({'partition': partition, 'question_type': question_type, 'image_count': image_count, 'from_coarse_taxon': losses[loss_index][0], 'to_coarse_taxon': gains[gain_index][0], 'count': moved})
            losses[loss_index][1] -= moved
            gains[gain_index][1] -= moved
            if losses[loss_index][1] == 0:
                loss_index += 1
            if gains[gain_index][1] == 0:
                gain_index += 1
    return events

def _rp_candidate_rank(candidate: _rp_Candidate, *, seed: int, partition: str) -> str:
    return _fp_stable_digest({'seed': seed, 'partition': partition, 'parent_sample_id': candidate.parent['sample_id'], 'source_image_keys': candidate.parent['image_keys']})

def _rp_dedupe_identity(candidate: _rp_Candidate) -> str:
    row = candidate.row
    return _fp_stable_digest({'dataset_category_id': candidate.parent['dataset_category_id'], 'question': _ep_clean_text(row.get('question')), 'answers': _ep_parse_answers(row), 'evidence': _ep_clean_text(row.get('evidence'))})

def _rp_candidate_tiebreak(candidate: _rp_Candidate) -> str:
    return _fp_stable_digest({'dedupe_key': candidate.dedupe_key, 'source_image_keys': candidate.parent['image_keys'], 'question_id': candidate.parent.get('question_id', '')})

def _rp_record_rejection(reasons: Counter[str], preview: list[dict[str, Any]], reason: str, source_row_index: int, **details: Any) -> None:
    reasons[reason] += 1
    if len(preview) < 200:
        preview.append({'source_row_index': source_row_index, 'reason': reason, **details})

def _rp_spool_train_candidates(sources: _ep_PipelineSources, config: _rp_PoolConfig, catalog_connection: sqlite3.Connection, spool_connection: sqlite3.Connection) -> tuple[Counter[str], list[dict[str, Any]], dict[str, int]]:
    rejections: Counter[str] = Counter()
    rejection_preview: list[dict[str, Any]] = []
    input_counts: Counter[str] = Counter()
    requested_types = set(config.primary_counts)
    for (row_index, raw_row) in enumerate(_rp_iter_csv_rows(sources.evqa_train.path)):
        row: dict[str, Any] = dict(raw_row)
        dataset_name = _ep_clean_text(row.get('dataset_name')).lower()
        input_counts[f"dataset:{dataset_name or '<missing>'}"] += 1
        if dataset_name != _ep_SUPPORTED_DATASET:
            continue
        question_type = _rp_canonical_question_type(row.get('question_type'))
        input_counts[f"question_type:{question_type or _ep_clean_text(row.get('question_type')) or '<missing>'}"] += 1
        if question_type not in requested_types:
            continue
        if not _ep_clean_text(row.get('question')):
            _rp_record_rejection(rejections, rejection_preview, 'missing_question', row_index)
            continue
        if not _ep_parse_answers(row):
            _rp_record_rejection(rejections, rejection_preview, 'missing_answer', row_index)
            continue
        if not _ep_clean_text(row.get('evidence')):
            _rp_record_rejection(rejections, rejection_preview, 'missing_evidence', row_index)
            continue
        raw_image_ids = _ep_split_delimited(row.get('dataset_image_ids'))
        image_ids = _ep_deduplicate(raw_image_ids)
        if len(raw_image_ids) != 5 or len(image_ids) != 5:
            _rp_record_rejection(rejections, rejection_preview, 'requires_five_unique_query_images', row_index, raw_image_count=len(raw_image_ids), unique_image_count=len(image_ids))
            continue
        images: list[dict[str, Any]] = []
        resolution_error = False
        for image_id in image_ids:
            image = _ep_lookup_query_image(catalog_connection, image_id, 'train')
            if image is None:
                other_split = _ep_lookup_query_image(catalog_connection, image_id, 'val')
                _rp_record_rejection(rejections, rejection_preview, 'query_image_wrong_split' if other_split else 'missing_catalog_image', row_index, image_id=image_id)
                resolution_error = True
                break
            images.append(image)
        if resolution_error:
            continue
        actual_categories = {_ep_clean_text(image.get('category_id')) for image in images if image}
        declared_category = _ep_clean_text(row.get('dataset_category_id'))
        if len(actual_categories) != 1 or not next(iter(actual_categories), ''):
            _rp_record_rejection(rejections, rejection_preview, 'inconsistent_query_categories', row_index, categories=sorted(actual_categories))
            continue
        actual_category = next(iter(actual_categories))
        if declared_category and declared_category != actual_category:
            _rp_record_rejection(rejections, rejection_preview, 'declared_category_mismatch', row_index, declared=declared_category, catalog=actual_category)
            continue
        category = images[0].get('category') or {}
        taxonomy_fields = ('supercategory', 'iconic_taxon_name', 'kingdom', 'class')
        if not _ep_clean_text(category.get('name')) or not any((_ep_clean_text(category.get(field)) for field in taxonomy_fields)):
            _rp_record_rejection(rejections, rejection_preview, 'missing_taxonomy', row_index, category_id=actual_category)
            continue
        coarse_taxon = _rp_coarse_taxon(category)
        if coarse_taxon not in config.supercategories:
            _rp_record_rejection(rejections, rejection_preview, 'non_target_coarse_taxon', row_index, coarse_taxon=coarse_taxon or '<missing>')
            continue
        row['question_type'] = question_type
        parent = _ep_logical_sample(row, official_split='train', source_row_index=row_index, images=images)
        candidate = _rp_Candidate(row=row, source_row_index=row_index, images=tuple(images), parent=parent, question_type=question_type, coarse_taxon=coarse_taxon, dedupe_key='')
        candidate = _rp_Candidate(**{**candidate.__dict__, 'dedupe_key': _rp_dedupe_identity(candidate)})
        tiebreak = _rp_candidate_tiebreak(candidate)
        existing = spool_connection.execute('\n            SELECT tiebreak, source_row_index\n            FROM candidates\n            WHERE dedupe_key = ?\n            ', (candidate.dedupe_key,)).fetchone()
        if existing is not None:
            (existing_tiebreak, existing_source_row_index) = existing
            new_identity = (tiebreak, candidate.source_row_index)
            existing_identity = (str(existing_tiebreak), int(existing_source_row_index))
            if new_identity < existing_identity:
                spool_connection.execute('\n                    INSERT INTO duplicate_rejections(\n                        dedupe_key, source_row_index, tiebreak\n                    )\n                    VALUES (?, ?, ?)\n                    ', (candidate.dedupe_key, int(existing_source_row_index), str(existing_tiebreak)))
                spool_connection.execute('\n                    UPDATE candidates\n                    SET source_row_index = ?,\n                        question_type = ?,\n                        coarse_taxon = ?,\n                        category_id = ?,\n                        parent_id = ?,\n                        tiebreak = ?,\n                        train_rank = ?,\n                        reserve_rank = ?,\n                        payload_json = ?,\n                        evidence_ok = 0\n                    WHERE dedupe_key = ?\n                    ', (candidate.source_row_index, candidate.question_type, candidate.coarse_taxon, _ep_clean_text(candidate.parent['dataset_category_id']), _ep_clean_text(candidate.parent['sample_id']), tiebreak, _rp_candidate_rank(candidate, seed=config.seed, partition='train'), _rp_candidate_rank(candidate, seed=config.seed, partition='reserve'), _rp_candidate_payload(candidate), candidate.dedupe_key))
            else:
                spool_connection.execute('\n                    INSERT INTO duplicate_rejections(\n                        dedupe_key, source_row_index, tiebreak\n                    )\n                    VALUES (?, ?, ?)\n                    ', (candidate.dedupe_key, candidate.source_row_index, tiebreak))
            continue
        spool_connection.execute('\n            INSERT INTO candidates(\n                dedupe_key,\n                source_row_index,\n                question_type,\n                coarse_taxon,\n                category_id,\n                parent_id,\n                tiebreak,\n                train_rank,\n                reserve_rank,\n                payload_json\n            )\n            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', (candidate.dedupe_key, candidate.source_row_index, candidate.question_type, candidate.coarse_taxon, _ep_clean_text(candidate.parent['dataset_category_id']), _ep_clean_text(candidate.parent['sample_id']), tiebreak, _rp_candidate_rank(candidate, seed=config.seed, partition='train'), _rp_candidate_rank(candidate, seed=config.seed, partition='reserve'), _rp_candidate_payload(candidate)))
    duplicate_count = int(spool_connection.execute('SELECT COUNT(*) FROM duplicate_rejections').fetchone()[0])
    if duplicate_count:
        rejections['duplicate_question_evidence'] += duplicate_count
    remaining_preview = max(0, 200 - len(rejection_preview))
    if remaining_preview:
        for (source_row_index, retained_source_row_index) in spool_connection.execute('\n            SELECT d.source_row_index, c.source_row_index\n            FROM duplicate_rejections AS d\n            JOIN candidates AS c\n                ON c.dedupe_key = d.dedupe_key\n            ORDER BY d.tiebreak, d.source_row_index\n            LIMIT ?\n            ', (remaining_preview,)):
            rejection_preview.append({'source_row_index': int(source_row_index), 'reason': 'duplicate_question_evidence', 'retained_source_row_index': int(retained_source_row_index)})
    spool_connection.commit()
    return (rejections, rejection_preview, dict(input_counts))

def _rp_kb_section_index(sources: _ep_PipelineSources, candidates: Iterable[_rp_Candidate]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    requested_urls = {pair['normalized_url'] for candidate in candidates for pair in candidate.parent.get('wiki_pairs', []) if pair.get('normalized_url')}
    sections_by_url: dict[str, set[str]] = defaultdict(set)
    found_urls: set[str] = set()
    with _ep_open_kb_json(sources.wikipedia_kb) as stream:
        for (original_url, page) in _ep_iter_selected_kb_pages(stream, requested_urls):
            normalized_url = _ep_normalize_url(original_url)
            found_urls.add(normalized_url)
            section_texts = page.get('section_texts') or []
            for (section_id, text) in enumerate(section_texts):
                if _ep_clean_text(text):
                    sections_by_url[normalized_url].add(str(section_id))
    return (dict(sections_by_url), {'requested_wiki_urls': len(requested_urls), 'found_wiki_urls': len(found_urls), 'missing_wiki_urls': sorted(requested_urls - found_urls), 'usable_sections': sum((len(value) for value in sections_by_url.values()))})

def _rp_evidence_rejection(candidate: _rp_Candidate, sections_by_url: Mapping[str, set[str]]) -> str:
    pairs = [pair for pair in candidate.parent.get('wiki_pairs', []) if pair.get('normalized_url')]
    section_ids = _ep_split_delimited(candidate.row.get('evidence_section_id'))
    expected_pages = 2 if candidate.question_type == '2_hop' else 1
    if len(pairs) != expected_pages:
        return f'{candidate.question_type}_requires_{expected_pages}_wiki_pages'
    if candidate.question_type == '2_hop':
        normalized_urls = [pair['normalized_url'] for pair in pairs]
        if any((not url for url in normalized_urls)):
            return '2_hop_requires_two_nonempty_wiki_pages'
        if len(set(normalized_urls)) != 2:
            return '2_hop_requires_two_distinct_wiki_pages'
    if not section_ids:
        return 'missing_evidence_section_id'
    if candidate.question_type == '2_hop' and len(section_ids) != 2:
        return '2_hop_requires_two_mapped_evidence_sections'
    if candidate.question_type == 'automatic' and len(pairs) == 1:
        if any((section_id not in sections_by_url.get(pairs[0]['normalized_url'], set()) for section_id in section_ids)):
            return 'missing_kb_evidence_section'
        return ''
    if any((section_id not in sections_by_url.get(pair['normalized_url'], set()) for (pair, section_id) in zip(pairs, section_ids))):
        return 'missing_kb_evidence_section'
    return ''

def _rp_select_partition(*, spool_connection: sqlite3.Connection, counts: Mapping[str, int], config: _rp_PoolConfig, partition: str, used_parent_ids: set[str]) -> tuple[list[tuple[_rp_Candidate, int]], dict[str, Any]]:
    image_order = list(config.image_count_weights)
    category_order = list(config.supercategories)
    selected: list[tuple[_rp_Candidate, int]] = []
    fallback_events: list[dict[str, Any]] = []
    requested_cells: dict[str, dict[str, dict[str, int]]] = {}
    actual_cells: dict[str, dict[str, dict[str, int]]] = {}
    type_order = list(config.primary_counts)
    spool_connection.execute('DELETE FROM used_parents')
    spool_connection.executemany('INSERT INTO used_parents(parent_id) VALUES (?)', ((parent_id,) for parent_id in sorted(used_parent_ids)))
    rank_column = 'train_rank' if partition == 'train' else 'reserve_rank'
    for (type_index, question_type) in enumerate(type_order):
        total = counts.get(question_type, 0)
        image_totals = _rp_apportion(total, config.image_count_weights, image_order)
        desired_categories = _rp_balanced_category_totals(total, category_order, offset=type_index)
        initial_matrix = _rp_quota_matrix(image_totals, desired_categories, image_order, category_order)
        available_counts = {category: 0 for category in category_order}
        for (coarse_taxon, count) in spool_connection.execute('\n            SELECT c.coarse_taxon, COUNT(*)\n            FROM candidates AS c\n            LEFT JOIN banned_categories AS b\n                ON b.category_id = c.category_id\n            LEFT JOIN used_parents AS u\n                ON u.parent_id = c.parent_id\n            WHERE c.evidence_ok = 1\n              AND c.question_type = ?\n              AND b.category_id IS NULL\n              AND u.parent_id IS NULL\n            GROUP BY c.coarse_taxon\n            ', (question_type,)):
            available_counts[str(coarse_taxon)] = int(count)
        if sum(available_counts.values()) < total:
            raise _rp_PoolCapacityError(f'Insufficient {question_type} candidates for {partition}: requested={total}, available={sum(available_counts.values())}, by_coarse_taxon={available_counts}.')
        adjusted_categories = _rp_adjust_category_capacity(desired_categories, available_counts, category_order)
        final_matrix = _rp_quota_matrix(image_totals, adjusted_categories, image_order, category_order)
        fallback_events.extend(_rp_matrix_fallbacks(partition=partition, question_type=question_type, initial=initial_matrix, actual=final_matrix, image_order=image_order, category_order=category_order))
        requested_cells[question_type] = {str(image_count): dict(initial_matrix[image_count]) for image_count in image_order}
        actual_cells[question_type] = {str(image_count): dict(final_matrix[image_count]) for image_count in image_order}
        for image_count in image_order:
            for category in category_order:
                amount = final_matrix[image_count][category]
                if not amount:
                    continue
                chosen: list[_rp_Candidate] = []
                cursor = spool_connection.execute(f'\n                    SELECT c.payload_json\n                    FROM candidates AS c\n                    LEFT JOIN banned_categories AS b\n                        ON b.category_id = c.category_id\n                    LEFT JOIN used_parents AS u\n                        ON u.parent_id = c.parent_id\n                    WHERE c.evidence_ok = 1\n                      AND c.question_type = ?\n                      AND c.coarse_taxon = ?\n                      AND b.category_id IS NULL\n                      AND u.parent_id IS NULL\n                    ORDER BY c.{rank_column}, c.tiebreak, c.source_row_index\n                    ', (question_type, category))
                for (payload_json,) in cursor:
                    candidate = _rp_candidate_from_payload(str(payload_json))
                    parent_id = _ep_clean_text(candidate.parent['sample_id'])
                    if parent_id in used_parent_ids:
                        continue
                    chosen.append(candidate)
                    used_parent_ids.add(parent_id)
                    if len(chosen) == amount:
                        break
                cursor.close()
                spool_connection.executemany('INSERT OR IGNORE INTO used_parents(parent_id) VALUES (?)', ((_ep_clean_text(candidate.parent['sample_id']),) for candidate in chosen))
                if len(chosen) != amount:
                    raise AssertionError('Capacity-adjusted quota selected too few candidates.')
                selected.extend(((candidate, image_count) for candidate in chosen))
    return (selected, {'requested_cells': requested_cells, 'actual_cells': actual_cells, 'fallbacks': fallback_events})

def _rp_logical_pool_row(candidate: _rp_Candidate, *, image_count: int, partition: str) -> dict[str, Any]:
    row = _ep_logical_sample(candidate.row, official_split='train', source_row_index=candidate.source_row_index, images=list(candidate.images[:image_count]))
    source_query_images = [dict(value) for value in candidate.parent['query_images']]
    row.update({'parent_sample_id': candidate.parent['sample_id'], 'source_query_images': source_query_images, 'source_image_keys': list(candidate.parent['image_keys']), 'coarse_taxon': candidate.coarse_taxon, 'pool_partition': partition, 'metadata_candidate_count': 0})
    return row

def _rp_official_val_exclusions(sources: _ep_PipelineSources, connection: sqlite3.Connection) -> tuple[set[str], set[str], Counter[str], list[dict[str, Any]]]:
    keys: set[str] = set()
    paths: set[str] = set()
    issues: Counter[str] = Counter()
    issue_preview: list[dict[str, Any]] = []
    for (row_index, row) in enumerate(_rp_iter_csv_rows(sources.evqa_val.path)):
        if _ep_clean_text(row.get('dataset_name')).lower() != _ep_SUPPORTED_DATASET:
            continue
        image_ids = _ep_deduplicate(_ep_split_delimited(row.get('dataset_image_ids')))
        if not image_ids:
            issues['no_dataset_image_ids'] += 1
            continue
        for image_id in image_ids:
            image_key = f'inaturalist:{image_id}'
            keys.add(image_key)
            image = _ep_lookup_query_image(connection, image_id, 'val')
            if image is None:
                reason = 'found_only_in_wrong_split' if _ep_lookup_query_image(connection, image_id, 'train') else 'missing_catalog_image'
                issues[reason] += 1
                if len(issue_preview) < 100:
                    issue_preview.append({'source_row_index': row_index, 'image_id': image_id, 'image_key': image_key, 'reason': reason})
                continue
            paths.add(_fp_normalize_local_path(image['normalized_path']))
    return (keys, paths, issues, issue_preview)

def _rp_metadata_candidate_counts(connection: sqlite3.Connection, category_ids: set[str], excluded_keys: set[str], excluded_paths: set[str]) -> dict[str, int]:
    """Count unique usable catalog candidates without scanning unrelated species.

    A duplicate catalog row at the same normalized path is only one physical
    candidate.  Query exclusions apply to both canonical image identity and
    normalized path, matching the later corpus-stage leak checks.
    """
    ordered_category_ids = sorted(category_ids)
    counts: dict[str, int] = {}
    chunk_size = 500
    for offset in range(0, len(ordered_category_ids), chunk_size):
        chunk = ordered_category_ids[offset:offset + chunk_size]
        identities: dict[str, set[str]] = {category_id: set() for category_id in chunk}
        placeholders = ','.join(('?' for _ in chunk))
        cursor = connection.execute(f'\n            SELECT image_key, category_id, normalized_path\n            FROM images\n            WHERE category_id IN ({placeholders})\n            ORDER BY category_id, source_split, image_id\n            ', chunk)
        for (image_key, category_id, normalized_path) in cursor:
            image_key = str(image_key)
            category_id = str(category_id)
            normalized_path = _fp_normalize_local_path(str(normalized_path))
            if image_key in excluded_keys or normalized_path in excluded_paths:
                continue
            identity = f'path:{normalized_path}' if normalized_path else f'key:{image_key}'
            identities[category_id].add(identity)
        counts.update({category_id: len(identities[category_id]) for category_id in chunk})
    return counts

def _rp_distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    by_type: Counter[str] = Counter()
    by_image_count: Counter[str] = Counter()
    by_taxon: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_type_images: Counter[str] = Counter()
    by_category_images: Counter[str] = Counter()
    cells: Counter[str] = Counter()
    query_image_count = 0
    single_image_questions = 0
    multi_image_questions = 0
    for row in values:
        question_type = _ep_clean_text(row.get('question_type'))
        image_count_value = int(row.get('image_count') or 0)
        image_count = str(image_count_value)
        taxon = _ep_clean_text(row.get('coarse_taxon'))
        category_id = _ep_clean_text(row.get('dataset_category_id'))
        by_type[question_type] += 1
        by_image_count[image_count] += 1
        by_taxon[taxon] += 1
        by_category[category_id] += 1
        by_type_images[question_type] += image_count_value
        by_category_images[category_id] += image_count_value
        query_image_count += image_count_value
        single_image_questions += int(image_count_value == 1)
        multi_image_questions += int(image_count_value > 1)
        cells[f'{question_type}|{image_count}|{taxon}'] += 1
    return {'total': len(values), 'query_images': query_image_count, 'single_image_questions': single_image_questions, 'multi_image_questions': multi_image_questions, 'by_question_type': dict(sorted(by_type.items())), 'query_images_by_question_type': dict(sorted(by_type_images.items())), 'by_image_count': dict(sorted(by_image_count.items(), key=lambda item: int(item[0]))), 'by_coarse_taxon': dict(sorted(by_taxon.items())), 'by_dataset_category_id': dict(sorted(by_category.items())), 'query_images_by_dataset_category_id': dict(sorted(by_category_images.items())), 'cells': dict(sorted(cells.items())), 'unique_parent_samples': len({_ep_clean_text(row.get('parent_sample_id')) for row in values}), 'unique_source_images': len({image_key for row in values for image_key in row.get('source_image_keys', [])})}

def _rp_pool_counts(config: _rp_PoolConfig) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    primary_counts = dict(config.primary_counts)
    reserve_counts = {question_type: math.floor(count * config.reserve_fraction + 0.5) for (question_type, count) in primary_counts.items()}
    image_order = list(config.image_count_weights)
    primary_image_quotas = {question_type: {str(image_count): count for (image_count, count) in _rp_apportion(total, config.image_count_weights, image_order).items()} for (question_type, total) in primary_counts.items()}
    reserve_image_quotas = {question_type: {str(image_count): count for (image_count, count) in _rp_apportion(total, config.image_count_weights, image_order).items()} for (question_type, total) in reserve_counts.items()}
    return (primary_counts, reserve_counts, primary_image_quotas, reserve_image_quotas)

def _rp_validated_catalog_inputs(sources: _ep_PipelineSources, catalog_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate that the catalog and both metadata files are one generation."""
    catalog_manifest = json.loads(catalog_manifest_path.read_text(encoding='utf-8'))
    _ep_require_schema_v2(catalog_manifest, 'catalog manifest')
    expected_catalog_sha256 = _ep_clean_text(catalog_manifest.get('catalog_sha256'))
    if not expected_catalog_sha256:
        raise ValueError('catalog_manifest.json has no catalog_sha256; rerun the catalog stage before pool.')
    actual_catalog_sha256 = _fp_sha256_file(sources.catalog_path)
    if expected_catalog_sha256.lower() != actual_catalog_sha256.lower():
        raise ValueError('catalog.sqlite changed after catalog construction (catalog_sha256 mismatch); rerun catalog before pool.')
    metadata_inputs = {'inaturalist_train_metadata': _ep_source_fingerprint(sources.inat_train.metadata, 'inaturalist.train.metadata'), 'inaturalist_val_metadata': _ep_source_fingerprint(sources.inat_val.metadata, 'inaturalist.val.metadata')}
    expected_metadata = catalog_manifest.get('input_sha256')
    if not isinstance(expected_metadata, Mapping):
        raise ValueError('catalog_manifest.json has no input_sha256 mapping; rerun the catalog stage before pool.')
    for (manifest_key, input_key) in (('inaturalist.train.metadata', 'inaturalist_train_metadata'), ('inaturalist.val.metadata', 'inaturalist_val_metadata')):
        expected_sha256 = _ep_clean_text(expected_metadata.get(manifest_key))
        if not expected_sha256:
            raise ValueError(f'catalog_manifest.json is missing the metadata fingerprint {manifest_key!r}; rerun the catalog stage before pool.')
        actual_sha256 = metadata_inputs[input_key]['sha256']
        if expected_sha256.lower() != actual_sha256.lower():
            raise ValueError(f'iNaturalist metadata changed after catalog construction: {input_key}. Rerun catalog before pool.')
    return (catalog_manifest, {**metadata_inputs, 'catalog_manifest': {'path': str(catalog_manifest_path), 'sha256': _fp_sha256_file(catalog_manifest_path)}, 'inat_catalog': {'path': str(sources.catalog_path), 'sha256': actual_catalog_sha256, 'manifest_catalog_sha256': expected_catalog_sha256}})

def _rp_pool_source_exclusions(rows: Iterable[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    values = list(rows)
    keys = {_ep_clean_text(image_key) for row in values for image_key in row.get('source_image_keys', []) if _ep_clean_text(image_key)}
    paths = {_fp_normalize_local_path(_ep_clean_text(image.get('image'))) for row in values for image in row.get('source_query_images', []) if _ep_clean_text(image.get('image'))}
    return (keys, paths)

def _rp_build_pool_impl(sources: _ep_PipelineSources) -> dict[str, Any]:
    config = _rp_PoolConfig.from_sources(sources)
    if not sources.catalog_path.is_file():
        raise FileNotFoundError('catalog.sqlite is missing; run the catalog stage first.')
    catalog_manifest_path = sources.output_dir / 'catalog_manifest.json'
    if not catalog_manifest_path.is_file():
        raise FileNotFoundError('catalog_manifest.json is missing; rerun the catalog stage before pool.')
    (_, catalog_inputs) = _rp_validated_catalog_inputs(sources, catalog_manifest_path)
    source_inputs = {'evqa_train_csv': _ep_source_fingerprint(sources.evqa_train, 'evqa.train_csv'), 'evqa_val_csv': _ep_source_fingerprint(sources.evqa_val, 'evqa.val_csv'), 'wikipedia_kb': _ep_source_fingerprint(sources.wikipedia_kb, 'wikipedia.kb'), **catalog_inputs}
    (primary_counts, reserve_counts, primary_image_quotas, reserve_image_quotas) = _rp_pool_counts(config)
    spool_dir = Path(tempfile.mkdtemp(prefix='.pool-candidate-spool-', dir=sources.output_dir))
    spool_path = spool_dir / 'candidates.sqlite'
    connection: sqlite3.Connection | None = None
    spool_connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(sources.catalog_path)
        spool_connection = sqlite3.connect(spool_path)
        _rp_initialize_candidate_spool(spool_connection)
        (rejections, rejection_preview, input_counts) = _rp_spool_train_candidates(sources, config, connection, spool_connection)
        candidate_count = int(spool_connection.execute('SELECT COUNT(*) FROM candidates').fetchone()[0])
        (sections_by_url, kb_report) = _rp_kb_section_index(sources, _rp_iter_spooled_candidates(spool_connection))
        spool_connection.execute('\n            CREATE TEMP TABLE valid_evidence (\n                dedupe_key TEXT PRIMARY KEY\n            ) WITHOUT ROWID\n            ')
        for candidate in _rp_iter_spooled_candidates(spool_connection):
            reason = _rp_evidence_rejection(candidate, sections_by_url)
            if reason:
                _rp_record_rejection(rejections, rejection_preview, reason, candidate.source_row_index, parent_sample_id=candidate.parent['sample_id'])
                continue
            spool_connection.execute('INSERT INTO valid_evidence(dedupe_key) VALUES (?)', (candidate.dedupe_key,))
        spool_connection.execute('\n            UPDATE candidates\n            SET evidence_ok = 1\n            WHERE dedupe_key IN (SELECT dedupe_key FROM valid_evidence)\n            ')
        spool_connection.execute('DROP TABLE valid_evidence')
        spool_connection.commit()
        evidence_candidate_count = int(spool_connection.execute('SELECT COUNT(*) FROM candidates WHERE evidence_ok = 1').fetchone()[0])
        (val_keys, val_paths, val_issues, val_issue_preview) = _rp_official_val_exclusions(sources, connection)
        if val_issues:
            raise RuntimeError('Official EVQA val query preflight failed; the complete heldout set cannot be proven. ' + _fp_canonical_json({'issues': dict(sorted(val_issues.items())), 'preview': val_issue_preview}))
        evidence_category_ids = {_ep_clean_text(category_id) for (category_id,) in spool_connection.execute('\n                SELECT DISTINCT category_id\n                FROM candidates\n                WHERE evidence_ok = 1\n                ')}
        val_only_candidate_counts = _rp_metadata_candidate_counts(connection, evidence_category_ids, val_keys, val_paths)
        excluded_category_ids = {category_id for (category_id, count) in val_only_candidate_counts.items() if count < config.min_positive_candidates}
        reachability_iterations: list[dict[str, Any]] = [{'iteration': 0, 'exclusion_scope': 'official_val_only', 'newly_excluded_dataset_category_ids': sorted(excluded_category_ids), 'candidate_counts': val_only_candidate_counts}]
        for iteration in range(1, len(evidence_category_ids) + 2):
            _rp_set_banned_categories(spool_connection, excluded_category_ids)
            used_parent_ids: set[str] = set()
            (primary_selection, primary_selection_report) = _rp_select_partition(spool_connection=spool_connection, counts=primary_counts, config=config, partition='train', used_parent_ids=used_parent_ids)
            (reserve_selection, reserve_selection_report) = _rp_select_partition(spool_connection=spool_connection, counts=reserve_counts, config=config, partition='reserve', used_parent_ids=used_parent_ids)
            train_rows = [_rp_logical_pool_row(candidate, image_count=image_count, partition='train') for (candidate, image_count) in primary_selection]
            reserve_rows = [_rp_logical_pool_row(candidate, image_count=image_count, partition='reserve') for (candidate, image_count) in reserve_selection]
            (pool_source_keys, pool_source_paths) = _rp_pool_source_exclusions(train_rows + reserve_rows)
            excluded_keys = pool_source_keys | val_keys
            excluded_paths = pool_source_paths | val_paths
            selected_category_ids = {_ep_clean_text(row['dataset_category_id']) for row in train_rows + reserve_rows}
            candidate_counts = _rp_metadata_candidate_counts(connection, selected_category_ids, excluded_keys, excluded_paths)
            newly_excluded_category_ids = {category_id for (category_id, count) in candidate_counts.items() if count < config.min_positive_candidates} - excluded_category_ids
            reachability_iterations.append({'iteration': iteration, 'exclusion_scope': 'tentative_pool_source_union_official_val', 'selected_dataset_category_ids': sorted(selected_category_ids), 'newly_excluded_dataset_category_ids': sorted(newly_excluded_category_ids), 'candidate_counts': candidate_counts})
            if not newly_excluded_category_ids:
                break
            excluded_category_ids.update(newly_excluded_category_ids)
        else:
            raise AssertionError('Metadata candidate reachability did not converge.')
        if any((count < config.min_positive_candidates for count in candidate_counts.values())):
            raise AssertionError('Final pool contains a category below min_positive_candidates.')
        _rp_set_banned_categories(spool_connection, excluded_category_ids)
        reachable_candidate_count = int(spool_connection.execute('\n                SELECT COUNT(*)\n                FROM candidates AS c\n                LEFT JOIN banned_categories AS b\n                    ON b.category_id = c.category_id\n                WHERE c.evidence_ok = 1\n                  AND b.category_id IS NULL\n                ').fetchone()[0])
        for (source_row_index, parent_id, category_id) in spool_connection.execute('\n            SELECT c.source_row_index, c.parent_id, c.category_id\n            FROM candidates AS c\n            JOIN banned_categories AS b\n                ON b.category_id = c.category_id\n            WHERE c.evidence_ok = 1\n            ORDER BY c.source_row_index\n            '):
            _rp_record_rejection(rejections, rejection_preview, 'insufficient_metadata_positive_candidates', int(source_row_index), parent_sample_id=str(parent_id), dataset_category_id=str(category_id), minimum=config.min_positive_candidates)
        for row in train_rows + reserve_rows:
            row['metadata_candidate_count'] = candidate_counts.get(_ep_clean_text(row['dataset_category_id']), 0)
    finally:
        if spool_connection is not None:
            spool_connection.close()
        if connection is not None:
            connection.close()
        shutil.rmtree(spool_dir, ignore_errors=True)
    train_rows.sort(key=lambda row: row['sample_id'])
    reserve_rows.sort(key=lambda row: row['sample_id'])
    if len(train_rows) != sum(primary_counts.values()):
        raise AssertionError('Primary pool row count does not match configured quota.')
    if len(reserve_rows) != sum(reserve_counts.values()):
        raise AssertionError('Reserve pool row count does not match configured quota.')
    train_parent_ids = {row['parent_sample_id'] for row in train_rows}
    reserve_parent_ids = {row['parent_sample_id'] for row in reserve_rows}
    if train_parent_ids & reserve_parent_ids:
        raise AssertionError('Primary and reserve pools overlap by parent_sample_id.')
    inputs = source_inputs
    below_minimum = {category_id: count for (category_id, count) in candidate_counts.items() if count < config.min_positive_candidates}
    exclusion_manifest = {'schema_version': _ep_PIPELINE_SCHEMA_VERSION, 'stage': 'pool', 'created_at': _ep_utc_now(), 'pool_source_image_keys': sorted(pool_source_keys), 'official_val_image_keys': sorted(val_keys), 'image_keys': sorted(excluded_keys), 'pool_source_normalized_paths': sorted(pool_source_paths), 'official_val_normalized_paths': sorted(val_paths), 'normalized_paths': sorted(excluded_paths), 'image_keys_sha256': _fp_stable_digest(sorted(excluded_keys)), 'normalized_paths_sha256': _fp_stable_digest(sorted(excluded_paths)), 'official_val_catalog_issues': dict(sorted(val_issues.items())), 'official_val_catalog_issue_preview': val_issue_preview}
    report: dict[str, Any] = {'schema_version': _ep_PIPELINE_SCHEMA_VERSION, 'stage': 'pool', 'status': 'complete', 'created_at': _ep_utc_now(), 'config': config.as_dict(), 'requested': {'train': {'total': sum(primary_counts.values()), 'by_question_type': primary_counts, 'by_question_type_and_image_count': primary_image_quotas}, 'reserve': {'total': sum(reserve_counts.values()), 'by_question_type': reserve_counts, 'by_question_type_and_image_count': reserve_image_quotas}}, 'actual': {'train': _rp_distribution(train_rows), 'reserve': _rp_distribution(reserve_rows)}, 'selection': {'train': primary_selection_report, 'reserve': reserve_selection_report, 'fallbacks': primary_selection_report['fallbacks'] + reserve_selection_report['fallbacks'], 'metadata_reachability_iterations': reachability_iterations}, 'candidates': {'after_metadata_and_dedup': candidate_count, 'after_kb_preflight': evidence_candidate_count, 'after_metadata_reachability': reachable_candidate_count, 'input_counts': input_counts, 'rejections': dict(sorted(rejections.items())), 'rejection_preview': rejection_preview}, 'kb': kb_report, 'exclusion': {'pool_source_images': len(pool_source_keys), 'official_val_images': len(val_keys), 'union_images': len(excluded_keys), 'union_paths': len(excluded_paths), 'official_val_catalog_issues': dict(sorted(val_issues.items()))}, 'metadata_candidates': {'by_dataset_category_id': candidate_counts, 'minimum_required_later': config.min_positive_candidates, 'below_minimum': below_minimum, 'official_val_only_by_dataset_category_id': val_only_candidate_counts, 'excluded_dataset_category_ids': sorted(excluded_category_ids)}, 'overlap': {'train_reserve_parent_samples': len(train_parent_ids & reserve_parent_ids)}, 'inputs': inputs}
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix='.pool-stage-', dir=sources.output_dir))
    try:
        final_paths = {'logical_train': sources.output_dir / 'pool_logical_train.jsonl', 'logical_reserve': sources.output_dir / 'pool_logical_reserve.jsonl', 'exclusion_manifest': sources.output_dir / 'pool_exclusion_manifest.json', 'report': sources.output_dir / 'pool_report.json', 'manifest': sources.output_dir / 'pool_manifest.json'}
        staged_paths = {name: staging_dir / final_path.name for (name, final_path) in final_paths.items()}
        _ep_atomic_write_jsonl(staged_paths['logical_train'], train_rows)
        _ep_atomic_write_jsonl(staged_paths['logical_reserve'], reserve_rows)
        _ep_atomic_write_json(staged_paths['exclusion_manifest'], exclusion_manifest)
        report_artifacts = {name: _ep_artifact_record(staged_paths[name], jsonl=name in {'logical_train', 'logical_reserve'}) for name in ('logical_train', 'logical_reserve', 'exclusion_manifest')}
        for (name, record) in report_artifacts.items():
            record['path'] = str(final_paths[name])
        report['artifacts'] = report_artifacts
        _ep_atomic_write_json(staged_paths['report'], report)
        manifest_artifacts = {name: _ep_artifact_record(staged_paths[name], jsonl=name in {'logical_train', 'logical_reserve'}) for name in ('logical_train', 'logical_reserve', 'exclusion_manifest', 'report')}
        for (name, record) in manifest_artifacts.items():
            record['path'] = str(final_paths[name])
        manifest = {'schema_version': _ep_PIPELINE_SCHEMA_VERSION, 'stage': 'pool', 'status': 'complete', 'created_at': _ep_utc_now(), 'config': config.as_dict(), 'inputs': inputs, 'train_sample_id_order_sha256': _fp_stable_digest([row['sample_id'] for row in train_rows]), 'reserve_sample_id_order_sha256': _fp_stable_digest([row['sample_id'] for row in reserve_rows]), 'train_parent_id_order_sha256': _fp_stable_digest([row['parent_sample_id'] for row in train_rows]), 'reserve_parent_id_order_sha256': _fp_stable_digest([row['parent_sample_id'] for row in reserve_rows]), 'artifacts': manifest_artifacts}
        _ep_atomic_write_json(staged_paths['manifest'], manifest)
        _ep_publish_artifact_set(staged_paths, final_paths)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    (sources.output_dir / 'pool_preflight_report.json').unlink(missing_ok=True)
    return report

def _rp_build_pool(sources: _ep_PipelineSources) -> dict[str, Any]:
    """Build and atomically publish the configured CPU-only RL logical pool."""
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _rp_build_pool_impl(sources)
    except Exception as exc:
        _ep_atomic_write_json(sources.output_dir / 'pool_preflight_report.json', {'schema_version': _ep_PIPELINE_SCHEMA_VERSION, 'stage': 'pool', 'status': 'failed', 'created_at': _ep_utc_now(), 'error_type': type(exc).__name__, 'error': str(exc), 'partial_pool_artifacts_written': False})
        raise

_rc_SCHEMA_VERSION = 2

_rc_PRIMARY_FILE = 'pool_logical_train.jsonl'

_rc_RESERVE_FILE = 'pool_logical_reserve.jsonl'

_rc_EXCLUSION_FILE = 'pool_exclusion_manifest.json'

_rc_POOL_MANIFEST_FILE = 'pool_manifest.json'

@dataclass(frozen=True)
class _rc_PoolCorpusConfig:
    min_candidates: int
    max_candidates: int
    candidate_scan_limit: int
    allowed_taxa: tuple[str, ...]

def _rc_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{label} must be a positive integer.')
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be a positive integer.') from exc
    if parsed <= 0:
        raise ValueError(f'{label} must be a positive integer.')
    return parsed

def _rc_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{label} must be a non-negative integer.')
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be a non-negative integer.') from exc
    if parsed < 0:
        raise ValueError(f'{label} must be a non-negative integer.')
    return parsed

def _rc_config(sources: Any) -> _rc_PoolCorpusConfig:
    raw = getattr(sources, 'raw', {}) or {}
    block = raw.get('rl_pool') or {}
    if not isinstance(block, Mapping):
        raise ValueError('rl_pool must be a JSON object.')
    min_candidates = _rc_positive_int(block.get('min_positive_candidates', block.get('future_min_positive_candidates', 8)), label='rl_pool.min_positive_candidates')
    max_candidates = _rc_positive_int(block.get('corpus_images_per_category', block.get('future_corpus_max_per_category', 32)), label='rl_pool.corpus_images_per_category')
    if min_candidates > max_candidates:
        raise ValueError('rl_pool.min_positive_candidates cannot exceed rl_pool.corpus_images_per_category.')
    scan_limit = _rc_positive_int(block.get('candidate_scan_limit', max(max_candidates * 2, min_candidates)), label='rl_pool.candidate_scan_limit')
    raw_taxa = block.get('supercategories', block.get('coarse_taxa', _rp_DEFAULT_SUPERCATEGORIES))
    if not isinstance(raw_taxa, Sequence) or isinstance(raw_taxa, (str, bytes)):
        raise ValueError('rl_pool.supercategories must be a list.')
    canonical_taxa: list[str] = []
    for value in raw_taxa:
        canonical = _rp_TAXON_ALIASES.get(_ep_clean_text(value).lower())
        if canonical not in _rp_DEFAULT_SUPERCATEGORIES:
            raise ValueError('rl_pool.supercategories supports only Plants, Insects, and Birds.')
        if canonical in canonical_taxa:
            raise ValueError(f'Duplicate canonical rl_pool supercategory {canonical!r}.')
        canonical_taxa.append(canonical)
    if not canonical_taxa:
        raise ValueError('rl_pool.supercategories must not be empty.')
    return _rc_PoolCorpusConfig(min_candidates=min_candidates, max_candidates=max_candidates, candidate_scan_limit=scan_limit, allowed_taxa=tuple(canonical_taxa))

def _rc_require_schema_v2(value: Mapping[str, Any], *, label: str) -> None:
    if value.get('schema_version') != _rc_SCHEMA_VERSION:
        raise ValueError(f"Incompatible {label} schema_version={value.get('schema_version')!r}; rerun catalog and pool.")

def _rc_required_artifact_sha(manifest: Mapping[str, Any], artifact_name: str) -> str:
    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, Mapping):
        raise ValueError('Pool manifest has no artifact fingerprint table; rerun pool.')
    record = artifacts.get(artifact_name)
    if not isinstance(record, Mapping):
        raise ValueError(f'Pool manifest is missing artifact record {artifact_name!r}; rerun pool.')
    sha256 = _ep_clean_text(record.get('sha256'))
    if not sha256:
        raise ValueError(f'Pool manifest artifact {artifact_name!r} has no sha256; rerun pool.')
    return sha256

def _rc_verify_pool_artifact(path: Path, manifest: Mapping[str, Any], artifact_name: str) -> None:
    expected = _rc_required_artifact_sha(manifest, artifact_name)
    if _fp_sha256_file(path) != expected:
        raise ValueError(f'{path.name} changed after pool selection; rerun pool and pool_corpus.')

def _rc_verify_manifest_input(pool_manifest: Mapping[str, Any], input_name: str, path: Path) -> None:
    inputs = pool_manifest.get('inputs')
    if not isinstance(inputs, Mapping):
        raise ValueError('Pool manifest has no input fingerprints; rerun pool.')
    record = inputs.get(input_name)
    if not isinstance(record, Mapping):
        raise ValueError(f'Pool manifest is missing input fingerprint {input_name!r}; rerun pool.')
    expected = _ep_clean_text(record.get('sha256'))
    if not expected:
        raise ValueError(f'Pool manifest input {input_name!r} has no sha256; rerun pool.')
    if _fp_sha256_file(path) != expected:
        raise ValueError(f'{path.name} changed after pool selection; rerun catalog and pool.')

def _rc_normalize_image_list(row: Mapping[str, Any], *, sample_id: str, field: str) -> list[dict[str, Any]]:
    raw_images = row.get(field)
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError(f'Pool sample {sample_id} has no {field}.')
    images: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for (position, raw) in enumerate(raw_images, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f'Pool sample {sample_id} has a non-object {field}[{position - 1}].')
        image = dict(raw)
        image_key = _ep_clean_text(image.get('image_key'))
        image_path = _ep_clean_text(image.get('image') or image.get('normalized_path'))
        file_name = _ep_clean_text(image.get('source_file_name') or image.get('file_name'))
        split = _ep_clean_text(image.get('source_split'))
        image_id = _ep_clean_text(image.get('dataset_image_id') or image.get('image_id'))
        if not all((image_key, image_path, file_name, split, image_id)):
            raise ValueError(f'Pool sample {sample_id} has an incomplete {field}[{position - 1}].')
        if image_key in seen_keys:
            raise ValueError(f'Pool sample {sample_id} has duplicate source image {image_key}.')
        seen_keys.add(image_key)
        images.append({'image_index': position, 'dataset_image_id': image_id, 'image_key': image_key, 'image': image_path, 'source_file_name': file_name, 'source_split': split})
    return images

def _rc_validate_logical_rows(rows: Iterable[Mapping[str, Any]], *, partition: str, allowed_taxa: set[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in rows:
        row = dict(raw)
        sample_id = _ep_clean_text(row.get('sample_id'))
        if not sample_id:
            raise ValueError(f'{partition} pool contains a sample without sample_id.')
        if sample_id in seen_ids:
            raise ValueError(f'Duplicate {partition} pool sample_id: {sample_id}')
        seen_ids.add(sample_id)
        _rc_require_schema_v2(row, label=f'{partition} pool sample {sample_id}')
        question_type = _ep_clean_text(row.get('question_type'))
        if not question_type:
            raise ValueError(f'Pool sample {sample_id} has no question_type.')
        image_count = row.get('image_count')
        if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count < 1 or (image_count > 5):
            raise ValueError(f'Pool sample {sample_id} has invalid image_count.')
        query_images = _rc_normalize_image_list(row, sample_id=sample_id, field='query_images')
        if len(query_images) != image_count:
            raise ValueError(f'Pool sample {sample_id} has inconsistent image_count/query_images.')
        if row.get('image_keys') != [image['image_key'] for image in query_images]:
            raise ValueError(f'Pool sample {sample_id} has inconsistent image_keys.')
        if not isinstance(row.get('source_query_images'), list):
            raise ValueError(f'Pool sample {sample_id} has no five-image source_query_images trace.')
        source_field = 'source_query_images'
        source_images = _rc_normalize_image_list(row, sample_id=sample_id, field=source_field)
        source_keys = [image['image_key'] for image in source_images]
        if len(source_images) != 5:
            raise ValueError(f'Pool sample {sample_id} must preserve all five EVQA source images.')
        declared_source_keys = row.get('source_image_keys')
        if not isinstance(declared_source_keys, list) or declared_source_keys != source_keys:
            raise ValueError(f'Pool sample {sample_id} has inconsistent source_image_keys.')
        if not set(row['image_keys']).issubset(source_keys):
            raise ValueError(f'Pool sample {sample_id} query_images are not a subset of source images.')
        if row['image_keys'] != source_keys[:image_count]:
            raise ValueError(f'Pool sample {sample_id} query_images must be the ordered CSV prefix.')
        source_row_index = row.get('source_row_index')
        official_split = _ep_clean_text(row.get('official_split'))
        if isinstance(source_row_index, bool) or not isinstance(source_row_index, int) or source_row_index < 0 or (official_split != 'train'):
            raise ValueError(f'Pool sample {sample_id} has invalid source row/split provenance.')
        expected_sample_id = _ep_sample_id(row, official_split, source_row_index, [image['image_key'] for image in query_images])
        if sample_id != expected_sample_id:
            raise ValueError(f'Pool sample {sample_id} does not match its selected image prefix.')
        category_id = _ep_clean_text(row.get('dataset_category_id'))
        category_key = _ep_clean_text(row.get('category_key'))
        raw_coarse_taxon = _ep_clean_text(row.get('coarse_taxon') or row.get('supercategory'))
        coarse_taxon = _rp_TAXON_ALIASES.get(raw_coarse_taxon.lower(), '')
        if not category_id or not category_key:
            raise ValueError(f'Pool sample {sample_id} has no category identity.')
        if coarse_taxon not in allowed_taxa:
            raise ValueError(f'Pool sample {sample_id} has unsupported coarse_taxon={coarse_taxon!r}.')
        if not _ep_clean_text(row.get('question')):
            raise ValueError(f'Pool sample {sample_id} has no question.')
        parent_sample_id = _ep_clean_text(row.get('parent_sample_id'))
        if not parent_sample_id:
            raise ValueError(f'Pool sample {sample_id} has no parent_sample_id.')
        expected_parent_id = _ep_sample_id(row, official_split, source_row_index, source_keys)
        if parent_sample_id != expected_parent_id:
            raise ValueError(f'Pool sample {sample_id} has inconsistent parent_sample_id.')
        if _ep_clean_text(row.get('pool_partition')) != partition:
            raise ValueError(f'Pool sample {sample_id} has inconsistent pool_partition.')
        answers = row.get('answers')
        if not isinstance(answers, list) or not any((_ep_clean_text(answer) for answer in answers)):
            raise ValueError(f'Pool sample {sample_id} has no answer targets.')
        wiki_pairs = row.get('wiki_pairs')
        if not isinstance(wiki_pairs, list) or not wiki_pairs:
            raise ValueError(f'Pool sample {sample_id} has no Wikipedia evidence.')
        row.update({'query_images': query_images, 'image_keys': [image['image_key'] for image in query_images], 'source_query_images': source_images, 'source_image_keys': source_keys, 'image_count': image_count, 'dataset_category_id': category_id, 'category_key': category_key, 'coarse_taxon': coarse_taxon, 'pool_partition': partition, 'parent_sample_id': parent_sample_id})
        normalized.append(row)
    return normalized

def _rc_load_inputs(sources: Any, cfg: _rc_PoolCorpusConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Path]]:
    output_dir = Path(sources.output_dir)
    paths = {'primary': output_dir / _rc_PRIMARY_FILE, 'reserve': output_dir / _rc_RESERVE_FILE, 'exclusion': output_dir / _rc_EXCLUSION_FILE, 'pool_report': output_dir / 'pool_report.json', 'pool_manifest': output_dir / _rc_POOL_MANIFEST_FILE, 'catalog_manifest': output_dir / 'catalog_manifest.json', 'catalog': Path(sources.catalog_path)}
    for (label, path) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Required pool artifact {label!r} is missing: {path}')
    pool_manifest = json.loads(paths['pool_manifest'].read_text(encoding='utf-8'))
    exclusion = json.loads(paths['exclusion'].read_text(encoding='utf-8'))
    if not isinstance(pool_manifest, dict) or not isinstance(exclusion, dict):
        raise ValueError('Pool manifest and exclusion manifest must be JSON objects.')
    _rc_require_schema_v2(pool_manifest, label='pool manifest')
    _rc_require_schema_v2(exclusion, label='pool exclusion manifest')
    manifest_pool_config = pool_manifest.get('config')
    if not isinstance(manifest_pool_config, Mapping):
        raise ValueError('Pool manifest has no normalized config; rerun pool.')
    current_pool_config = _rp_PoolConfig.from_sources(sources).as_dict()
    if _fp_canonical_json(manifest_pool_config) != _fp_canonical_json(current_pool_config):
        raise ValueError('rl_pool selection config changed after pool construction; rerun pool.')
    _rc_verify_pool_artifact(paths['primary'], pool_manifest, 'logical_train')
    _rc_verify_pool_artifact(paths['reserve'], pool_manifest, 'logical_reserve')
    _rc_verify_pool_artifact(paths['exclusion'], pool_manifest, 'exclusion_manifest')
    _rc_verify_pool_artifact(paths['pool_report'], pool_manifest, 'report')
    _rc_verify_manifest_input(pool_manifest, 'evqa_train_csv', Path(sources.evqa_train.path))
    _rc_verify_manifest_input(pool_manifest, 'evqa_val_csv', Path(sources.evqa_val.path))
    _rc_verify_manifest_input(pool_manifest, 'inaturalist_train_metadata', Path(sources.inat_train.metadata.path))
    _rc_verify_manifest_input(pool_manifest, 'inaturalist_val_metadata', Path(sources.inat_val.metadata.path))
    _rc_verify_manifest_input(pool_manifest, 'catalog_manifest', paths['catalog_manifest'])
    _rc_verify_manifest_input(pool_manifest, 'inat_catalog', paths['catalog'])
    _rc_verify_manifest_input(pool_manifest, 'wikipedia_kb', Path(sources.wikipedia_kb.path))
    primary = _rc_validate_logical_rows(_ep_read_jsonl(paths['primary']), partition='train', allowed_taxa=set(cfg.allowed_taxa))
    reserve = _rc_validate_logical_rows(_ep_read_jsonl(paths['reserve']), partition='reserve', allowed_taxa=set(cfg.allowed_taxa))
    overlap = sorted({row['sample_id'] for row in primary} & {row['sample_id'] for row in reserve})
    if overlap:
        raise ValueError(f'Primary and reserve pools overlap: {overlap[:10]}')
    for (partition, rows, sample_digest_key, parent_digest_key) in (('train', primary, 'train_sample_id_order_sha256', 'train_parent_id_order_sha256'), ('reserve', reserve, 'reserve_sample_id_order_sha256', 'reserve_parent_id_order_sha256')):
        expected_sample_digest = _ep_clean_text(pool_manifest.get(sample_digest_key))
        expected_parent_digest = _ep_clean_text(pool_manifest.get(parent_digest_key))
        if not expected_sample_digest or not expected_parent_digest:
            raise ValueError(f'Pool manifest is missing {partition} order digests; rerun pool.')
        actual_sample_digest = _fp_stable_digest([row['sample_id'] for row in rows])
        actual_parent_digest = _fp_stable_digest([row['parent_sample_id'] for row in rows])
        if expected_sample_digest != actual_sample_digest:
            raise ValueError(f'{partition} pool sample order changed after selection; rerun pool.')
        if expected_parent_digest != actual_parent_digest:
            raise ValueError(f'{partition} pool parent order changed after selection; rerun pool.')
    return (primary, reserve, exclusion, pool_manifest, paths)

def _rc_exclusion_sets(exclusion: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    key_values = exclusion.get('image_keys') or exclusion.get('heldout_image_keys') or []
    path_values = exclusion.get('normalized_paths') or exclusion.get('heldout_normalized_paths') or []
    if not isinstance(key_values, list) or not isinstance(path_values, list):
        raise ValueError('pool_exclusion_manifest image_keys/normalized_paths must be arrays.')
    keys = {_ep_clean_text(value) for value in key_values if _ep_clean_text(value)}
    paths = {_fp_normalize_local_path(value) for value in path_values if _ep_clean_text(value)}
    missing_keys: list[str] = []
    missing_paths: list[str] = []
    for row in rows:
        for image in row['source_query_images']:
            if image['image_key'] not in keys:
                missing_keys.append(image['image_key'])
            normalized = _fp_normalize_local_path(image['image'])
            if normalized not in paths:
                missing_paths.append(normalized)
    if missing_keys or missing_paths:
        raise ValueError(f'Pool exclusion manifest does not cover every primary/reserve source image: missing_keys={sorted(set(missing_keys))[:20]}, missing_paths={sorted(set(missing_paths))[:20]}.')
    expected_key_digest = _ep_clean_text(exclusion.get('image_keys_sha256'))
    expected_path_digest = _ep_clean_text(exclusion.get('normalized_paths_sha256'))
    if not expected_key_digest or not expected_path_digest:
        raise ValueError('Pool exclusion manifest is missing image key/path fingerprints; rerun pool.')
    if expected_key_digest != _fp_stable_digest(sorted(keys)):
        raise ValueError('Pool exclusion image_keys fingerprint is corrupt.')
    if expected_path_digest != _fp_stable_digest(sorted(paths)):
        raise ValueError('Pool exclusion normalized_paths fingerprint is corrupt.')
    return (keys, paths)

def _rc_validate_query_catalog(catalog_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    connection = sqlite3.connect(catalog_path)
    try:
        for row in rows:
            for image in row['query_images']:
                found = connection.execute('\n                    SELECT image_id, source_split, file_name, normalized_path,\n                           category_id, category_key\n                    FROM images WHERE image_key = ?\n                    ', (image['image_key'],)).fetchone()
                expected = (image['dataset_image_id'], image['source_split'], _ep_safe_archive_name(image['source_file_name']), _fp_normalize_local_path(image['image']), row['dataset_category_id'], row['category_key'])
                if found is None:
                    raise ValueError(f"Pool query image is absent from catalog: {image['image_key']}")
                actual = (str(found[0]), str(found[1]), str(found[2]), _fp_normalize_local_path(found[3]), str(found[4]), str(found[5]))
                if actual != expected:
                    raise ValueError(f"Pool query image disagrees with catalog: {image['image_key']}")
    finally:
        connection.close()

def _rc_build_candidate_inventory(sources: Any, category_ids: set[str], excluded_keys: set[str], excluded_paths: set[str], *, inventory_path: Path) -> tuple[Path, dict[str, Any]]:
    """Persist every unique eligible catalog candidate in deterministic order.

    The former implementation retained only the first ``candidate_scan_limit``
    metadata rows per category.  That made the limit a correctness boundary:
    corrupt or absent early files hid valid later candidates.  Keeping the
    inventory in SQLite makes the scan exhaustive without retaining millions
    of catalog objects in Python memory; validation consumes it in bounded
    batches later.
    """
    connection = sqlite3.connect(inventory_path)
    connection.executescript('\n        CREATE TABLE candidates (\n            sequence INTEGER PRIMARY KEY AUTOINCREMENT,\n            category_id TEXT NOT NULL,\n            image_key TEXT NOT NULL UNIQUE,\n            normalized_path TEXT NOT NULL UNIQUE,\n            payload TEXT NOT NULL\n        );\n        CREATE INDEX candidates_category_sequence\n            ON candidates(category_id, sequence);\n        ')
    seen = 0
    excluded = 0
    duplicate_keys = 0
    duplicate_paths = 0
    inserted = 0
    by_category: Counter[str] = Counter()
    try:
        for row in _ep_catalog_rows_for_categories(Path(sources.catalog_path), category_ids):
            seen += 1
            image_key = _ep_clean_text(row.get('image_key'))
            normalized_path = _fp_normalize_local_path(row['normalized_path'])
            if image_key in excluded_keys or normalized_path in excluded_paths:
                excluded += 1
                continue
            existing = connection.execute('\n                SELECT image_key, normalized_path FROM candidates\n                WHERE image_key = ? OR normalized_path = ?\n                LIMIT 1\n                ', (image_key, normalized_path)).fetchone()
            if existing is not None:
                duplicate_keys += int(str(existing[0]) == image_key)
                duplicate_paths += int(_fp_normalize_local_path(existing[1]) == normalized_path)
                continue
            normalized_row = dict(row)
            normalized_row['image_key'] = image_key
            normalized_row['normalized_path'] = normalized_path
            category_id = str(normalized_row['category_id'])
            connection.execute('\n                INSERT INTO candidates(\n                    category_id, image_key, normalized_path, payload\n                ) VALUES (?, ?, ?, ?)\n                ', (category_id, image_key, normalized_path, json.dumps(normalized_row, ensure_ascii=False, separators=(',', ':'), sort_keys=True)))
            inserted += 1
            by_category[category_id] += 1
            if inserted % 10000 == 0:
                connection.commit()
        connection.commit()
    finally:
        connection.close()
    return (inventory_path, {'catalog_rows_in_selected_categories': seen, 'excluded_catalog_rows': excluded, 'unique_candidate_rows': inserted, 'duplicate_image_keys': duplicate_keys, 'duplicate_normalized_paths': duplicate_paths, 'inventory_by_category': dict(sorted(by_category.items()))})

def _rc_next_candidate_batch(connection: sqlite3.Connection, category_ids: Iterable[str], cursors: dict[str, int], *, batch_size: int) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for category_id in sorted(category_ids):
        raw_rows = connection.execute('\n            SELECT sequence, payload FROM candidates\n            WHERE category_id = ? AND sequence > ?\n            ORDER BY sequence\n            LIMIT ?\n            ', (category_id, cursors.get(category_id, 0), batch_size)).fetchall()
        if not raw_rows:
            result[category_id] = []
            continue
        cursors[category_id] = int(raw_rows[-1][0])
        result[category_id] = [json.loads(value) for (_, value) in raw_rows]
    return result

def _rc_register_member(connection: sqlite3.Connection, sources: Any, row: Mapping[str, Any], *, query: bool) -> None:
    split = _ep_clean_text(row.get('source_split'))
    file_name = _ep_safe_archive_name(_ep_clean_text(row.get('source_file_name') if query else row.get('file_name')))
    supplied_path = _ep_clean_text(row.get('image') if query else row.get('normalized_path'))
    (_, expected_path) = _ep_safe_image_destination(_ep_source_for_split(sources, split).image_root, file_name)
    if _fp_normalize_local_path(supplied_path) != expected_path:
        raise ValueError(f'Image path does not match configured image_root for {file_name!r}.')
    existing = connection.execute('\n        SELECT target_path FROM members\n        WHERE source_split = ? AND file_name = ?\n        ', (split, file_name)).fetchone()
    if existing and _fp_normalize_local_path(existing[0]) != expected_path:
        raise ValueError(f'Conflicting materialization target for {file_name!r}.')
    # The configured archives are the canonical pixel source. Re-extract every
    # selected query/candidate member on a corpus rebuild so a valid-looking
    # but locally modified extracted image cannot silently change the dataset
    # under the same archive/build fingerprint.
    connection.execute('\n        INSERT OR IGNORE INTO members(\n            source_split, file_name, target_path, kind, needs_extract\n        ) VALUES (?, ?, ?, ?, ?)\n        ', (split, file_name, expected_path, 'query' if query else 'retrieval', 1))

def _rc_materialize(sources: Any, rows: Sequence[Mapping[str, Any]], candidates: Mapping[str, Sequence[Mapping[str, Any]]], staging_dir: Path) -> dict[str, Any]:
    member_db = staging_dir / 'pool_image_members.sqlite'
    connection = sqlite3.connect(member_db)
    try:
        connection.execute('\n            CREATE TABLE members (\n                source_split TEXT NOT NULL,\n                file_name TEXT NOT NULL,\n                target_path TEXT NOT NULL,\n                kind TEXT NOT NULL,\n                needs_extract INTEGER NOT NULL,\n                PRIMARY KEY(source_split, file_name)\n            )\n            ')
        for row in rows:
            for image in row['query_images']:
                _rc_register_member(connection, sources, image, query=True)
        for category_rows in candidates.values():
            for candidate in category_rows:
                _rc_register_member(connection, sources, candidate, query=False)
        connection.commit()
        extracted_by_split: dict[str, int] = {}
        missing_before_by_split: dict[str, int] = {}
        for split in ('train', 'val'):
            needed = int(connection.execute('\n                    SELECT COUNT(*) FROM members\n                    WHERE source_split = ? AND needs_extract = 1\n                    ', (split,)).fetchone()[0])
            missing_before_by_split[split] = needed
            source = _ep_source_for_split(sources, split)
            if needed and source.archive is not None:
                extracted_by_split[split] = _ep_extract_indexed_images(source.archive.path, source.image_root, split, connection)
            else:
                extracted_by_split[split] = 0
        remaining = int(connection.execute('SELECT COUNT(*) FROM members WHERE needs_extract = 1').fetchone()[0])
        return {'unique_members': int(connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]), 'missing_before_by_split': missing_before_by_split, 'extracted_by_split': extracted_by_split, 'archive_members_not_found': remaining}
    finally:
        connection.close()

def _rc_query_failures(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[str]], Counter[str]]:
    failures: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        reasons: list[str] = []
        for image in row['query_images']:
            path = Path(image['image'])
            if _ep_valid_image(path):
                continue
            reason = 'missing_query_image' if not path.is_file() else 'corrupt_query_image'
            reasons.append(f"{reason}:image_index={image['image_index']}:image_key={image['image_key']}")
            counts[reason] += 1
        if reasons:
            failures[row['sample_id']] = reasons
    return (failures, counts)

def _rc_usable_candidates(candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]], *, max_candidates: int) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], Counter[str]]:
    usable: dict[str, list[dict[str, Any]]] = {}
    dropped: list[dict[str, Any]] = []
    dropped_by_category: Counter[str] = Counter()
    for (category_id, rows) in sorted(candidate_rows.items()):
        selected: list[dict[str, Any]] = []
        for row in rows:
            if _ep_valid_image(Path(row['normalized_path'])):
                if len(selected) < max_candidates:
                    selected.append(dict(row))
                continue
            dropped.append({'image_key': row['image_key'], 'category_id': category_id, 'source_split': row['source_split'], 'path': row['normalized_path'], 'reason': 'missing_candidate_image' if not Path(row['normalized_path']).is_file() else 'corrupt_candidate_image'})
            dropped_by_category[category_id] += 1
        usable[category_id] = selected
    return (usable, dropped, dropped_by_category)

def _rc_materialize_candidate_inventory(sources: Any, rows: Sequence[Mapping[str, Any]], inventory_path: Path, category_ids: set[str], *, batch_size: int, max_candidates: int, staging_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], Counter[str], dict[str, Any], int]:
    """Validate candidates in bounded batches until every category is full.

    ``batch_size`` is a throughput knob, never a hard candidate cap.  If a
    batch contains corrupt files or missing archive members, the next catalog
    batch is materialized and checked until the category reaches
    ``max_candidates`` or its inventory is exhausted.
    """
    selected: dict[str, list[dict[str, Any]]] = {category_id: [] for category_id in category_ids}
    dropped: list[dict[str, Any]] = []
    dropped_by_category: Counter[str] = Counter()
    cursors = {category_id: 0 for category_id in category_ids}
    active = set(category_ids)
    rounds: list[dict[str, Any]] = []
    validated_rows = 0
    first_round = True
    connection = sqlite3.connect(inventory_path)
    try:
        while active or first_round:
            first_round = False
            batch = _rc_next_candidate_batch(connection, active, cursors, batch_size=batch_size)
            nonempty = {category_id: values for (category_id, values) in batch.items() if values}
            round_dir = staging_dir / f'materialize-{len(rounds) + 1:04d}'
            round_dir.mkdir(parents=True, exist_ok=False)
            materialization = _rc_materialize(sources, rows if not rounds else (), nonempty, round_dir)
            batch_size_actual = sum((len(values) for values in nonempty.values()))
            validated_rows += batch_size_actual
            (usable, round_dropped, round_dropped_by_category) = _rc_usable_candidates(nonempty, max_candidates=max_candidates)
            dropped.extend(round_dropped)
            dropped_by_category.update(round_dropped_by_category)
            for (category_id, values) in usable.items():
                remaining = max_candidates - len(selected[category_id])
                if remaining > 0:
                    selected[category_id].extend(values[:remaining])
            exhausted = {category_id for category_id in active if not batch.get(category_id)}
            filled = {category_id for category_id in active if len(selected[category_id]) >= max_candidates}
            active -= exhausted | filled
            rounds.append({'round': len(rounds) + 1, 'candidate_rows': batch_size_actual, 'active_categories_after': len(active), **materialization})
            if not nonempty:
                break
    finally:
        connection.close()
    return (selected, dropped, dropped_by_category, {'rounds': rounds, 'round_count': len(rounds), 'extracted_by_split': {split: sum((int(round_record['extracted_by_split'].get(split, 0)) for round_record in rounds)) for split in ('train', 'val')}, 'archive_members_not_found': sum((int(round_record['archive_members_not_found']) for round_record in rounds))}, validated_rows)

def _rc_bucket(row: Mapping[str, Any]) -> tuple[str, int]:
    return (str(row['question_type']), int(row['image_count']))

def _rc_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_type: Counter[str] = Counter()
    by_image_count: Counter[str] = Counter()
    by_type_image: Counter[str] = Counter()
    by_taxon: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_type_images: Counter[str] = Counter()
    by_category_images: Counter[str] = Counter()
    query_image_count = 0
    single_image_questions = 0
    multi_image_questions = 0
    for row in rows:
        (question_type, image_count) = _rc_bucket(row)
        category_key = str(row['category_key'])
        by_type[question_type] += 1
        by_image_count[str(image_count)] += 1
        by_type_image[f'{question_type}|{image_count}'] += 1
        by_taxon[str(row['coarse_taxon'])] += 1
        by_category[category_key] += 1
        by_type_images[question_type] += image_count
        by_category_images[category_key] += image_count
        query_image_count += image_count
        single_image_questions += int(image_count == 1)
        multi_image_questions += int(image_count > 1)
    return {'total': len(rows), 'query_images': query_image_count, 'single_image_questions': single_image_questions, 'multi_image_questions': multi_image_questions, 'by_question_type': dict(sorted(by_type.items())), 'query_images_by_question_type': dict(sorted(by_type_images.items())), 'by_image_count': dict(sorted(by_image_count.items(), key=lambda item: int(item[0]))), 'by_question_type_and_image_count': dict(sorted(by_type_image.items())), 'by_coarse_taxon': dict(sorted(by_taxon.items())), 'unique_categories': len(by_category), 'by_category': dict(sorted(by_category.items())), 'query_images_by_category': dict(sorted(by_category_images.items()))}

def _rc_choose_final_rows(primary: Sequence[dict[str, Any]], reserve: Sequence[dict[str, Any]], failures: Mapping[str, list[str]], *, allowed_taxa: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available: dict[tuple[str, int, str], deque[dict[str, Any]]] = defaultdict(deque)
    invalid_reserve: Counter[str] = Counter()
    for row in reserve:
        row_failures = failures.get(row['sample_id'], [])
        if row_failures:
            for reason in row_failures:
                invalid_reserve[reason.split(':', 1)[0]] += 1
            continue
        (question_type, image_count) = _rc_bucket(row)
        available[question_type, image_count, row['coarse_taxon']].append(row)
    final: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    shortage: Counter[str] = Counter()
    for primary_row in primary:
        primary_failures = failures.get(primary_row['sample_id'], [])
        if not primary_failures:
            final.append(primary_row)
            continue
        (question_type, image_count) = _rc_bucket(primary_row)
        original_taxon = str(primary_row['coarse_taxon'])
        same_key = (question_type, image_count, original_taxon)
        replacement: dict[str, Any] | None = None
        if available[same_key]:
            replacement = available[same_key].popleft()
        else:
            candidates: list[tuple[int, int, str]] = []
            for (taxon_index, taxon) in enumerate(allowed_taxa):
                if taxon == original_taxon:
                    continue
                remaining = len(available[question_type, image_count, taxon])
                if remaining:
                    candidates.append((-remaining, taxon_index, taxon))
            if candidates:
                (_, _, chosen_taxon) = min(candidates)
                replacement = available[question_type, image_count, chosen_taxon].popleft()
        if replacement is None:
            shortage[f'{question_type}|{image_count}|{original_taxon}'] += 1
            continue
        final.append(replacement)
        replacements.append({'replaced_sample_id': primary_row['sample_id'], 'replacement_sample_id': replacement['sample_id'], 'question_type': question_type, 'image_count': image_count, 'from_coarse_taxon': original_taxon, 'to_coarse_taxon': replacement['coarse_taxon'], 'cross_taxon': replacement['coarse_taxon'] != original_taxon, 'reasons': primary_failures})
    if shortage:
        raise RuntimeError('Reserve pool cannot satisfy final question_type/image_count targets: ' + json.dumps(dict(sorted(shortage.items())), sort_keys=True))
    if len(final) != len(primary):
        raise RuntimeError(f'Final pool has {len(final)} rows; expected {len(primary)}.')
    target_buckets = Counter((_rc_bucket(row) for row in primary))
    final_buckets = Counter((_rc_bucket(row) for row in final))
    if target_buckets != final_buckets:
        raise RuntimeError('Reserve replacement changed question_type/image_count quotas.')
    return (final, {'primary_kept': len(primary) - len(replacements), 'reserve_replacements': len(replacements), 'cross_taxon_replacements': sum((int(item['cross_taxon']) for item in replacements)), 'replacements': replacements, 'invalid_reserve_by_reason': dict(sorted(invalid_reserve.items()))})

def _rc_rl_row(logical: Mapping[str, Any], *, positive_count: int) -> dict[str, Any]:
    row = _ep_rl_sample(logical, positive_count=positive_count, text_resolvable=True)
    row.update({'parent_sample_id': logical['parent_sample_id'], 'source_image_keys': list(logical['source_image_keys']), 'coarse_taxon': logical['coarse_taxon']})
    if not row['vision_resolvable'] or not row['text_resolvable'] or (not row['retrieval_resolvable']):
        raise RuntimeError(f"Final RL row is unexpectedly unresolvable: {row['sample_id']}")
    return row

def _rc_input_record(path: Path) -> dict[str, Any]:
    return {'path': str(path), 'sha256': _fp_sha256_file(path)}

def _rc_archive_input_records(sources: Any) -> dict[str, dict[str, Any]]:
    """Hash configured archives once before selective extraction begins."""
    records: dict[str, dict[str, Any]] = {}
    for (split, source) in (('train', sources.inat_train), ('val', sources.inat_val)):
        if source.archive is None:
            continue
        records[f'inaturalist_{split}_archive'] = _ep_source_fingerprint(source.archive, f'inaturalist.{split}.archive')
    return records

def _rc_pool_seed(pool_manifest: Mapping[str, Any]) -> int:
    config = pool_manifest.get('config')
    if not isinstance(config, Mapping) or 'seed' not in config:
        raise ValueError('Pool manifest config.seed is missing; rerun pool.')
    value = config['seed']
    if isinstance(value, bool):
        raise ValueError('Pool manifest config.seed must be an integer; rerun pool.')
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Pool manifest config.seed must be an integer; rerun pool.') from exc

def _rc_pool_fallback_summary(pool_report: Mapping[str, Any]) -> dict[str, Any]:
    selection = pool_report.get('selection')
    if not isinstance(selection, Mapping):
        raise ValueError('Pool report has no selection summary; rerun pool.')
    events = selection.get('fallbacks')
    if not isinstance(events, list):
        raise ValueError('Pool report selection.fallbacks must be an array.')
    moved = 0
    by_partition: Counter[str] = Counter()
    by_question_type: Counter[str] = Counter()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError('Pool report contains an invalid fallback event.')
        count = _rc_nonnegative_int(event.get('count', 0), label='pool_report.selection.fallbacks[].count')
        moved += count
        by_partition[_ep_clean_text(event.get('partition')) or '<missing>'] += count
        by_question_type[_ep_clean_text(event.get('question_type')) or '<missing>'] += count
    return {'event_count': len(events), 'rows_reallocated': moved, 'by_partition': dict(sorted(by_partition.items())), 'by_question_type': dict(sorted(by_question_type.items())), 'events': [dict(event) for event in events]}

def _rc_conda_environment_record() -> dict[str, Any]:
    prefix_text = _ep_clean_text(os.environ.get('CONDA_PREFIX'))
    conda_version = _ep_clean_text(os.environ.get('CONDA_VERSION'))
    conda_executable = _ep_clean_text(os.environ.get('CONDA_EXE'))
    if not conda_version and conda_executable:
        try:
            completed = subprocess.run([conda_executable, '--version'], check=False, capture_output=True, text=True, timeout=10)
            output = _ep_clean_text(completed.stdout or completed.stderr)
            if completed.returncode == 0 and output:
                conda_version = output.removeprefix('conda ').strip()
        except (OSError, subprocess.SubprocessError):
            pass
    record: dict[str, Any] = {'conda_environment': os.environ.get('CONDA_DEFAULT_ENV', ''), 'conda_prefix': prefix_text, 'conda_version': conda_version or 'unavailable', 'python': platform.python_version(), 'python_executable': sys.executable}
    if prefix_text:
        history_path = Path(prefix_text) / 'conda-meta' / 'history'
        if history_path.is_file():
            record['conda_history'] = {'path': str(history_path), 'sha256': _fp_sha256_file(history_path)}
    return record

def _rc_pixel_digest(records: Iterable[tuple[str, str]]) -> dict[str, Any]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for (raw_key, raw_path) in records:
        image_key = _ep_clean_text(raw_key)
        normalized_path = _fp_normalize_local_path(raw_path)
        key = (image_key, normalized_path)
        if key in unique:
            continue
        path = Path(normalized_path)
        if not _ep_valid_image(path):
            raise RuntimeError(f'Cannot fingerprint missing or corrupt final image {image_key}: {path}')
        unique[key] = {'image_key': image_key, 'normalized_path': normalized_path, 'file_sha256': _fp_sha256_file(path)}
    ordered = [unique[key] for key in sorted(unique)]
    return {'count': len(ordered), 'sha256': _fp_stable_digest(ordered)}

def _rc_build_pool_corpus_impl(sources: Any) -> dict[str, Any]:
    cfg = _rc_config(sources)
    (primary, reserve, exclusion, pool_manifest, input_paths) = _rc_load_inputs(sources, cfg)
    archive_input_records = _rc_archive_input_records(sources)
    pool_report = json.loads(input_paths['pool_report'].read_text(encoding='utf-8'))
    if not isinstance(pool_report, dict):
        raise ValueError('Pool report must be a JSON object.')
    _rc_require_schema_v2(pool_report, label='pool report')
    seed = _rc_pool_seed(pool_manifest)
    pool_fallback_summary = _rc_pool_fallback_summary(pool_report)
    all_rows = primary + reserve
    if not primary:
        raise ValueError('Primary pool is empty.')
    configured_counts = (pool_manifest.get('config') or {}).get('primary_counts') or (getattr(sources, 'raw', {}) or {}).get('rl_pool', {}).get('primary_counts') or {}
    if configured_counts:
        actual_counts = Counter((str(row['question_type']) for row in primary))
        expected_counts: dict[str, int] = {}
        for (key, value) in configured_counts.items():
            count = _rc_nonnegative_int(value, label=f'rl_pool.primary_counts.{key}')
            if count:
                expected_counts[str(key)] = count
        if dict(actual_counts) != expected_counts:
            raise ValueError(f'Primary pool counts {dict(actual_counts)} do not match configured counts {expected_counts}; rerun pool.')
    (excluded_keys, excluded_paths) = _rc_exclusion_sets(exclusion, all_rows)
    _rc_validate_query_catalog(Path(sources.catalog_path), all_rows)
    category_ids = {str(row['dataset_category_id']) for row in all_rows}
    output_dir = Path(sources.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix='.pool-corpus-stage-', dir=output_dir))
    try:
        (inventory_path, candidate_scan_report) = _rc_build_candidate_inventory(sources, category_ids, excluded_keys, excluded_paths, inventory_path=staging_dir / 'candidate_inventory.sqlite')
        (usable_candidates, dropped_candidates, dropped_by_category, materialization, validated_candidate_rows) = _rc_materialize_candidate_inventory(sources, all_rows, inventory_path, category_ids, batch_size=cfg.candidate_scan_limit, max_candidates=cfg.max_candidates, staging_dir=staging_dir)
        candidate_scan_report.update({'candidate_rows_selected_for_validation': validated_candidate_rows, 'candidate_rows_not_needed_after_cap': max(0, int(candidate_scan_report['unique_candidate_rows']) - validated_candidate_rows), 'candidate_rows_beyond_scan_limit': 0})
        (query_failures, query_failure_counts) = _rc_query_failures(all_rows)
        positive_counts = {category_id: len(rows) for (category_id, rows) in usable_candidates.items()}
        (all_text_rows, sections_by_url, text_scan_report) = _ep_build_text_corpus(all_rows, sources.wikipedia_kb)
        failures: dict[str, list[str]] = {sample_id: list(reasons) for (sample_id, reasons) in query_failures.items()}
        failure_counts: Counter[str] = Counter(query_failure_counts)
        for row in all_rows:
            sample_id = row['sample_id']
            count = positive_counts.get(row['dataset_category_id'], 0)
            if count < cfg.min_candidates:
                failures.setdefault(sample_id, []).append(f'insufficient_visual_candidates:found={count}:required={cfg.min_candidates}')
                failure_counts['insufficient_visual_candidates'] += 1
            if not _ep_is_text_resolvable(row, sections_by_url):
                failures.setdefault(sample_id, []).append('missing_text_evidence')
                failure_counts['missing_text_evidence'] += 1
        (final_logical, replacement_report) = _rc_choose_final_rows(primary, reserve, failures, allowed_taxa=cfg.allowed_taxa)
        final_categories = {str(row['dataset_category_id']) for row in final_logical}
        final_vision_candidates = [candidate for category_id in sorted(final_categories) for candidate in usable_candidates.get(category_id, [])]
        final_positive_counts = Counter((str(candidate['category_id']) for candidate in final_vision_candidates))
        for row in final_logical:
            if final_positive_counts[row['dataset_category_id']] < cfg.min_candidates:
                raise RuntimeError(f"Final category {row['dataset_category_id']} lost its minimum visual candidate count.")
        final_urls = {pair.get('normalized_url') for row in final_logical for pair in row.get('wiki_pairs', []) if pair.get('normalized_url')}
        final_text_rows = [row for row in all_text_rows if row.get('url') in final_urls]
        final_sections: dict[str, set[str]] = defaultdict(set)
        for row in final_text_rows:
            final_sections[str(row['url'])].add(str(row['section_id']))
        unresolved_after_trim = [row['sample_id'] for row in final_logical if not _ep_is_text_resolvable(row, final_sections)]
        if unresolved_after_trim:
            raise RuntimeError(f'Text corpus trimming removed required evidence for samples: {unresolved_after_trim[:20]}')
        vision_rows = [_ep_vision_corpus_row(candidate) for candidate in final_vision_candidates]
        vision_leaks = [row['image_key'] for row in vision_rows if row['image_key'] in excluded_keys or _fp_normalize_local_path(row['image']) in excluded_paths]
        if vision_leaks:
            raise RuntimeError(f'Excluded query images leaked into visual corpus: {vision_leaks[:20]}')
        rl_rows = [_rc_rl_row(row, positive_count=int(final_positive_counts[row['dataset_category_id']])) for row in final_logical]
        if any((row['positive_candidate_count'] < cfg.min_candidates or row['positive_candidate_count'] > cfg.max_candidates for row in rl_rows)):
            raise RuntimeError('Final RL rows violate visual positive candidate bounds.')
        query_pixel_digest = _rc_pixel_digest(((image['image_key'], image['image']) for row in final_logical for image in row['query_images']))
        candidate_pixel_digest = _rc_pixel_digest(((candidate['image_key'], candidate['normalized_path']) for candidate in final_vision_candidates))
        staged = {'vision_corpus': staging_dir / 'vision_corpus.jsonl', 'text_corpus': staging_dir / 'text_corpus.jsonl', 'train': staging_dir / 'train.parquet', 'dropped_candidates': staging_dir / 'pool_dropped_candidates.jsonl'}
        _ep_atomic_write_jsonl(staged['vision_corpus'], vision_rows)
        _ep_atomic_write_jsonl(staged['text_corpus'], final_text_rows)
        _ep_atomic_write_jsonl(staged['dropped_candidates'], dropped_candidates)
        _ep_write_parquet(rl_rows, staged['train'])
        final_distribution = _rc_distribution(final_logical)
        target_distribution = _rc_distribution(primary)
        if final_distribution['total'] != target_distribution['total'] or final_distribution['by_question_type'] != target_distribution['by_question_type'] or final_distribution['by_image_count'] != target_distribution['by_image_count'] or (final_distribution['by_question_type_and_image_count'] != target_distribution['by_question_type_and_image_count']):
            raise RuntimeError('Final RL distribution does not match pool targets.')
        report = {'schema_version': _rc_SCHEMA_VERSION, 'stage': 'pool_corpus', 'status': 'ok', 'created_at': _ep_utc_now(), 'config': {'seed': seed, 'min_positive_candidates': cfg.min_candidates, 'corpus_images_per_category': cfg.max_candidates, 'candidate_scan_limit': cfg.candidate_scan_limit, 'supercategories': list(cfg.allowed_taxa)}, 'samples': {'primary': len(primary), 'reserve': len(reserve), 'final': len(final_logical)}, 'target_distribution': target_distribution, 'final_distribution': final_distribution, 'replacement': replacement_report, 'pool_selection_fallback': pool_fallback_summary, 'failure_counts': dict(sorted(failure_counts.items())), 'failed_sample_preview': [{'sample_id': sample_id, 'reasons': reasons} for (sample_id, reasons) in list(sorted(failures.items()))[:100]], 'image_materialization': materialization, 'candidate_scan': candidate_scan_report, 'candidate_validation': {'dropped_candidates': len(dropped_candidates), 'dropped_by_category': dict(sorted(dropped_by_category.items())), 'final_categories': len(final_categories), 'final_vision_corpus_rows': len(vision_rows), 'minimum_per_category': min(final_positive_counts.values(), default=0), 'maximum_per_category': max(final_positive_counts.values(), default=0), 'positive_counts': dict(sorted(final_positive_counts.items()))}, 'heldout': {'image_keys': len(excluded_keys), 'normalized_paths': len(excluded_paths), 'vision_corpus_leaks': 0}, 'text': {**text_scan_report, 'final_wiki_urls': len(final_urls), 'final_text_sections': len(final_text_rows)}, 'pixel_fingerprints': {'query': query_pixel_digest, 'candidate': candidate_pixel_digest}}
        final_paths = {'vision_corpus': output_dir / 'vision_corpus.jsonl', 'text_corpus': output_dir / 'text_corpus.jsonl', 'train': output_dir / 'train.parquet', 'dropped_candidates': output_dir / 'pool_dropped_candidates.jsonl', 'report': output_dir / 'pool_corpus_report.json', 'manifest': output_dir / 'build_manifest.json'}
        artifact_records = {name: _ep_artifact_record(path, jsonl=name in {'vision_corpus', 'text_corpus', 'dropped_candidates'}) for (name, path) in staged.items()}
        for (name, record) in artifact_records.items():
            record['path'] = str(final_paths[name])
        report['artifacts'] = artifact_records
        _ep_atomic_write_json(staging_dir / 'pool_corpus_report.json', report)
        manifest_artifacts = dict(artifact_records)
        report_artifact = _ep_artifact_record(staging_dir / 'pool_corpus_report.json')
        report_artifact['path'] = str(final_paths['report'])
        manifest_artifacts['report'] = report_artifact
        manifest_inputs = {'evqa_train_csv': _rc_input_record(Path(sources.evqa_train.path)), 'evqa_val_csv': _rc_input_record(Path(sources.evqa_val.path)), 'pool_logical_train': _rc_input_record(input_paths['primary']), 'pool_logical_reserve': _rc_input_record(input_paths['reserve']), 'pool_exclusion_manifest': _rc_input_record(input_paths['exclusion']), 'pool_report': _rc_input_record(input_paths['pool_report']), 'pool_manifest': _rc_input_record(input_paths['pool_manifest']), 'catalog': _rc_input_record(input_paths['catalog']), 'wikipedia_kb': _rc_input_record(sources.wikipedia_kb.path)}
        manifest_inputs.update(archive_input_records)
        manifest = {'schema_version': _rc_SCHEMA_VERSION, 'stage': 'pool_corpus', 'created_at': _ep_utc_now(), 'seed': seed, 'environment': _rc_conda_environment_record(), 'sources_config_path': str(sources.config_path), 'sources_config_sha256': _fp_sha256_file(sources.config_path), 'inputs': manifest_inputs, 'pool_selection_fallback': pool_fallback_summary, 'heldout': {'manifest_path': str(input_paths['exclusion']), 'image_keys_sha256': _fp_stable_digest(sorted(excluded_keys)), 'normalized_paths_sha256': _fp_stable_digest(sorted(excluded_paths))}, 'final_sample_ids_sha256': _fp_stable_digest([row['sample_id'] for row in rl_rows]), 'query_pixels_sha256': query_pixel_digest['sha256'], 'candidate_pixels_sha256': candidate_pixel_digest['sha256'], 'pixel_fingerprints': {'query': query_pixel_digest, 'candidate': candidate_pixel_digest, 'combined_sha256': _fp_stable_digest({'query': query_pixel_digest, 'candidate': candidate_pixel_digest})}, 'artifacts': manifest_artifacts, 'report': report}
        _ep_atomic_write_json(staging_dir / 'build_manifest.json', manifest)
        staged['report'] = staging_dir / 'pool_corpus_report.json'
        staged['manifest'] = staging_dir / 'build_manifest.json'
        _ep_publish_artifact_set(staged, final_paths)
        (output_dir / 'pool_corpus_preflight_report.json').unlink(missing_ok=True)
        return manifest
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

def _rc_build_pool_corpus(sources: Any) -> dict[str, Any]:
    """Build the compact CPU-only RL dataset and retrieval corpora.

    The required pool artifacts are read from ``sources.output_dir``.  Final
    artifacts are staged and published as one rollback-protected generation.
    On failure, existing final artifacts remain untouched and a preflight
    diagnostic is written.
    """
    output_dir = Path(sources.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _rc_build_pool_corpus_impl(sources)
    except Exception as exc:
        _ep_atomic_write_json(output_dir / 'pool_corpus_preflight_report.json', {'schema_version': _rc_SCHEMA_VERSION, 'stage': 'pool_corpus', 'status': 'failed', 'created_at': _ep_utc_now(), 'error_type': type(exc).__name__, 'error': str(exc), 'partial_parquet_written': False, 'expected_outputs': [str(output_dir / 'train.parquet'), str(output_dir / 'vision_corpus.jsonl'), str(output_dir / 'text_corpus.jsonl'), str(output_dir / 'pool_corpus_report.json'), str(output_dir / 'build_manifest.json')]})
        raise
_FIXED_RECIPE = {
    "primary_counts": {"automatic": 10_000, "2_hop": 1_000},
    "image_count_weights": {
        "1": 0.30,
        "2": 0.25,
        "3": 0.25,
        "4": 0.10,
        "5": 0.10,
    },
    "supercategories": ["Plants", "Insects", "Birds"],
    "reserve_fraction": 0.25,
    "min_positive_candidates": 8,
    "corpus_images_per_category": 32,
    "image_selection": "first_k_of_five_in_csv_order",
}
_PUBLIC_FILENAMES = (
    "train.parquet",
    "vision_corpus.jsonl",
    "text_corpus.jsonl",
    "build_report.json",
    "build_manifest.json",
)
_CACHE_MARKER = "standalone_build_manifest.json"
_CATALOG_CACHE_MARKER = "catalog_stage_manifest.json"
_POOL_CACHE_MARKER = "pool_stage_manifest.json"
_CORPUS_CACHE_MARKER = "corpus_stage_manifest.json"
_FORCE_CACHE_FILES = (
    "catalog.sqlite",
    "catalog_manifest.json",
    "catalog_preflight_report.json",
    "pool_logical_train.jsonl",
    "pool_logical_reserve.jsonl",
    "pool_exclusion_manifest.json",
    "pool_report.json",
    "pool_manifest.json",
    "pool_preflight_report.json",
    "train.parquet",
    "vision_corpus.jsonl",
    "text_corpus.jsonl",
    "pool_dropped_candidates.jsonl",
    "pool_corpus_report.json",
    "pool_corpus_preflight_report.json",
    "build_manifest.json",
    _CATALOG_CACHE_MARKER,
    _POOL_CACHE_MARKER,
    _CORPUS_CACHE_MARKER,
    _CACHE_MARKER,
)

_CATALOG_STAGE_FILES = (
    "catalog.sqlite",
    "catalog_manifest.json",
)
_POOL_STAGE_FILES = (
    "pool_logical_train.jsonl",
    "pool_logical_reserve.jsonl",
    "pool_exclusion_manifest.json",
    "pool_report.json",
    "pool_manifest.json",
)
_CORPUS_STAGE_FILES = (
    "train.parquet",
    "vision_corpus.jsonl",
    "text_corpus.jsonl",
    "pool_dropped_candidates.jsonl",
    "pool_corpus_report.json",
    "build_manifest.json",
)


def _load_fixed_sources(config_path: Path) -> tuple[Any, dict[str, Any]]:
    config_path = config_path.resolve(strict=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object.")
    if "rl_pool" in raw:
        raise ValueError(
            "rl_pool is fixed and must not appear in the shared config."
        )
    output_value = raw.get("output_dir")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ValueError("output_dir must be a non-empty absolute path.")
    output_dir = Path(
        os.path.expandvars(os.path.expanduser(output_value.strip()))
    )
    if not output_dir.is_absolute():
        raise ValueError("output_dir must be absolute.")
    resolved_output = output_dir.resolve(strict=False)
    if resolved_output == PROJECT_ROOT or resolved_output.is_relative_to(
        PROJECT_ROOT
    ):
        raise ValueError(
            "output_dir must be outside the DualSearch repository checkout."
        )
    raw_seed = raw.get("seed", 42)
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ValueError("seed must be an integer.")
    sources = _ep_load_sources(config_path)
    for split, source in (
        ("train", sources.inat_train),
        ("val", sources.inat_val),
    ):
        if source.archive is None:
            raise ValueError(
                f"inaturalist.{split}.archive is required for the fixed "
                "11K build; automatic downloads are disabled."
            )
    return (
        replace(sources, output_dir=resolved_output),
        raw,
    )


def _input_fingerprint(sources: Any) -> tuple[str, dict[str, Any]]:
    inputs = _ep_consumed_input_fingerprints(sources)
    inputs["inaturalist_train_image_root"] = {
        "path": _fp_normalize_local_path(sources.inat_train.image_root),
    }
    inputs["inaturalist_val_image_root"] = {
        "path": _fp_normalize_local_path(sources.inat_val.image_root),
    }
    payload = {
        "schema_version": 2,
        "builder": "dual_search_fixed_11k_rl",
        "seed": int(sources.seed),
        "recipe": _FIXED_RECIPE,
        "inputs": inputs,
    }
    return _fp_stable_digest(payload), inputs


def _artifact_matches(path: Path, record: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and bool(record.get("sha256"))
        and _fp_sha256_file(path) == str(record["sha256"])
    )


def _artifact_sha_table(
    directory: Path,
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Required cached artifact is missing: {path}")
        table[name] = {
            "path": str(path),
            "sha256": _fp_sha256_file(path),
        }
    return table


def _stage_cache_reusable(
    cache_dir: Path,
    marker_name: str,
    stage_fingerprint: str,
    artifact_names: Sequence[str],
    *,
    validate_pixels: bool = False,
) -> bool:
    marker_path = cache_dir / marker_name
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("stage_fingerprint") != stage_fingerprint:
            return False
        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return False
        if not all(
            isinstance(artifacts.get(name), Mapping)
            and _artifact_matches(cache_dir / name, artifacts[name])
            for name in artifact_names
        ):
            return False
        if validate_pixels:
            manifest = json.loads(
                (cache_dir / "build_manifest.json").read_text(encoding="utf-8")
            )
            if not _pixel_fingerprints_match(cache_dir, manifest):
                return False
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return False
    return True


def _write_stage_cache_marker(
    cache_dir: Path,
    marker_name: str,
    stage: str,
    stage_fingerprint: str,
    artifact_names: Sequence[str],
) -> None:
    _ep_atomic_write_json(
        cache_dir / marker_name,
        {
            "schema_version": 2,
            "stage": stage,
            "status": "complete",
            "created_at": _ep_utc_now(),
            "stage_fingerprint": stage_fingerprint,
            "artifacts": _artifact_sha_table(cache_dir, artifact_names),
        },
    )


def _plain_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_nested(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _plain_nested(to_list())
    return value


def _current_pixel_fingerprints(directory: Path) -> dict[str, Any]:
    import pandas as pd

    train_path = directory / "train.parquet"
    vision_path = directory / "vision_corpus.jsonl"
    query_records: list[tuple[str, str]] = []
    frame = pd.read_parquet(train_path, columns=["query_images"])
    for raw_images in frame["query_images"].tolist():
        images = _plain_nested(raw_images)
        if not isinstance(images, list):
            raise ValueError("train.parquet query_images must be a list.")
        for image in images:
            if not isinstance(image, Mapping):
                raise ValueError("train.parquet contains an invalid query image.")
            query_records.append(
                (
                    _ep_clean_text(image.get("image_key")),
                    _ep_clean_text(image.get("image")),
                )
            )
    candidate_records = [
        (
            _ep_clean_text(row.get("image_key") or row.get("id")),
            _ep_clean_text(row.get("image")),
        )
        for row in _fp_iter_jsonl(vision_path)
    ]
    return {
        "query": _rc_pixel_digest(query_records),
        "candidate": _rc_pixel_digest(candidate_records),
    }


def _pixel_fingerprints_match(
    directory: Path,
    manifest: Mapping[str, Any],
) -> bool:
    expected = manifest.get("pixel_fingerprints")
    if not isinstance(expected, Mapping):
        return False
    current = _current_pixel_fingerprints(directory)
    for name in ("query", "candidate"):
        expected_record = expected.get(name)
        current_record = current[name]
        if not isinstance(expected_record, Mapping):
            return False
        if (
            str(expected_record.get("sha256")) != str(current_record["sha256"])
            or int(expected_record.get("count", -1))
            != int(current_record["count"])
        ):
            return False
    return True


def _can_reuse(
    output_dir: Path,
    cache_dir: Path,
    build_fingerprint: str,
) -> bool:
    cache_path = cache_dir / _CACHE_MARKER
    manifest_path = output_dir / "build_manifest.json"
    if not cache_path.is_file() or not manifest_path.is_file():
        return False
    try:
        cache_manifest = json.loads(cache_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        cache_manifest.get("build_fingerprint") != build_fingerprint
        or manifest.get("build_fingerprint") != build_fingerprint
        or cache_manifest.get("public_manifest_sha256")
        != _fp_sha256_file(manifest_path)
    ):
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return False
    if not all(
        isinstance(artifacts.get(name), Mapping)
        and _artifact_matches(output_dir / name, artifacts[name])
        for name in _PUBLIC_FILENAMES[:-1]
    ):
        return False
    heldout = manifest.get("heldout")
    if not isinstance(heldout, Mapping):
        return False
    embedded_keys = heldout.get("image_keys")
    embedded_paths = heldout.get("normalized_paths")
    if (
        not isinstance(embedded_keys, list)
        or not embedded_keys
        or not isinstance(embedded_paths, list)
        or not embedded_paths
        or _fp_stable_digest(sorted(embedded_keys))
        != str(heldout.get("image_keys_sha256") or "")
        or _fp_stable_digest(sorted(embedded_paths))
        != str(heldout.get("normalized_paths_sha256") or "")
    ):
        return False
    try:
        return _pixel_fingerprints_match(output_dir, manifest)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False


def _clear_known_cache_outputs(cache_dir: Path) -> None:
    for name in _FORCE_CACHE_FILES:
        path = cache_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _publish_public_artifacts(
    sources: Any,
    cache_dir: Path,
    build_fingerprint: str,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(sources.output_dir)
    required_names = (
        "train.parquet",
        "vision_corpus.jsonl",
        "text_corpus.jsonl",
        "pool_report.json",
        "pool_corpus_report.json",
        "build_manifest.json",
    )
    for name in required_names:
        path = cache_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Build stage did not produce required artifact: {path}"
            )

    pool_report = json.loads(
        (cache_dir / "pool_report.json").read_text(encoding="utf-8")
    )
    corpus_report = json.loads(
        (cache_dir / "pool_corpus_report.json").read_text(encoding="utf-8")
    )
    upstream_manifest = cache_dir / "build_manifest.json"
    upstream_manifest_value = json.loads(
        upstream_manifest.read_text(encoding="utf-8")
    )
    if not isinstance(upstream_manifest_value, Mapping):
        raise ValueError("Cached corpus build manifest must be a JSON object.")
    heldout = upstream_manifest_value.get("heldout")
    pixel_fingerprints = upstream_manifest_value.get("pixel_fingerprints")
    if not isinstance(heldout, Mapping):
        raise ValueError("Cached corpus build manifest has no heldout record.")
    if not isinstance(pixel_fingerprints, Mapping):
        raise ValueError(
            "Cached corpus build manifest has no pixel fingerprints."
        )
    exclusion_value = json.loads(
        (cache_dir / "pool_exclusion_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(exclusion_value, Mapping):
        raise ValueError("Cached heldout manifest must be a JSON object.")
    heldout_keys = exclusion_value.get("image_keys")
    heldout_paths = exclusion_value.get("normalized_paths")
    if not isinstance(heldout_keys, list) or not isinstance(
        heldout_paths, list
    ):
        raise ValueError(
            "Cached heldout manifest has no image key/path lists."
        )
    image_keys_sha256 = _fp_stable_digest(sorted(heldout_keys))
    normalized_paths_sha256 = _fp_stable_digest(sorted(heldout_paths))
    if image_keys_sha256 != str(heldout.get("image_keys_sha256") or ""):
        raise ValueError("Cached heldout image-key fingerprint is corrupt.")
    if normalized_paths_sha256 != str(
        heldout.get("normalized_paths_sha256") or ""
    ):
        raise ValueError("Cached heldout path fingerprint is corrupt.")
    public_heldout = {
        "image_keys": list(heldout_keys),
        "normalized_paths": list(heldout_paths),
        "image_keys_sha256": image_keys_sha256,
        "normalized_paths_sha256": normalized_paths_sha256,
    }
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".fixed-11k-publish-", dir=output_dir)
    )
    try:
        staged = {
            name: staging_dir / name for name in _PUBLIC_FILENAMES[:3]
        }
        for name, staged_path in staged.items():
            shutil.copyfile(cache_dir / name, staged_path)

        report = {
            "schema_version": 2,
            "stage": "fixed_11k_rl",
            "status": "complete",
            "created_at": _ep_utc_now(),
            "build_fingerprint": build_fingerprint,
            "recipe": _FIXED_RECIPE,
            "pool": pool_report,
            "corpus": corpus_report,
        }
        report_path = staging_dir / "build_report.json"
        _ep_atomic_write_json(report_path, report)
        staged["build_report.json"] = report_path

        final_paths = {
            name: output_dir / name for name in _PUBLIC_FILENAMES
        }
        artifact_records = {}
        for name, staged_path in staged.items():
            record = _ep_artifact_record(
                staged_path,
                jsonl=name.endswith(".jsonl"),
            )
            record["path"] = str(final_paths[name])
            artifact_records[name] = record
        manifest = {
            "schema_version": 2,
            "stage": "fixed_11k_rl",
            "status": "complete",
            "created_at": _ep_utc_now(),
            "build_fingerprint": build_fingerprint,
            "seed": int(sources.seed),
            "recipe": _FIXED_RECIPE,
            "inputs": dict(inputs),
            "heldout": public_heldout,
            "pixel_fingerprints": dict(pixel_fingerprints),
            "cache": {
                "path": str(cache_dir),
                "upstream_manifest_sha256": _fp_sha256_file(
                    upstream_manifest
                ),
            },
            "artifacts": artifact_records,
        }
        manifest_path = staging_dir / "build_manifest.json"
        _ep_atomic_write_json(manifest_path, manifest)
        staged["build_manifest.json"] = manifest_path
        _ep_publish_artifact_set(staged, final_paths)

        _ep_atomic_write_json(
            cache_dir / _CACHE_MARKER,
            {
                "schema_version": 2,
                "stage": "fixed_11k_rl_cache",
                "status": "complete",
                "created_at": _ep_utc_now(),
                "build_fingerprint": build_fingerprint,
                "public_manifest_sha256": _fp_sha256_file(
                    final_paths["build_manifest.json"]
                ),
            },
        )
        return report
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def build(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    sources, _ = _load_fixed_sources(config_path)
    output_dir = Path(sources.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".build_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    build_fingerprint, inputs = _input_fingerprint(sources)
    if not force and _can_reuse(
        output_dir, cache_dir, build_fingerprint
    ):
        return json.loads(
            (output_dir / "build_report.json").read_text(encoding="utf-8")
        )
    if force:
        _clear_known_cache_outputs(cache_dir)

    cache_sources = replace(sources, output_dir=cache_dir)

    catalog_fingerprint = _fp_stable_digest(
        {
            "schema_version": 2,
            "stage": "catalog",
            "inputs": {
                key: inputs[key]
                for key in (
                    "inaturalist_train_metadata",
                    "inaturalist_val_metadata",
                    "inaturalist_train_image_root",
                    "inaturalist_val_image_root",
                )
            },
        }
    )
    if not _stage_cache_reusable(
        cache_dir,
        _CATALOG_CACHE_MARKER,
        catalog_fingerprint,
        _CATALOG_STAGE_FILES,
    ):
        _ep_build_catalog(cache_sources)
        _write_stage_cache_marker(
            cache_dir,
            _CATALOG_CACHE_MARKER,
            "catalog",
            catalog_fingerprint,
            _CATALOG_STAGE_FILES,
        )

    catalog_artifacts = _artifact_sha_table(
        cache_dir,
        _CATALOG_STAGE_FILES,
    )
    pool_fingerprint = _fp_stable_digest(
        {
            "schema_version": 2,
            "stage": "pool",
            "seed": int(sources.seed),
            "recipe": _FIXED_RECIPE,
            "inputs": {
                key: inputs[key]
                for key in (
                    "evqa_train_csv",
                    "evqa_val_csv",
                    "wikipedia_kb",
                )
            },
            "catalog_artifacts": catalog_artifacts,
        }
    )
    if not _stage_cache_reusable(
        cache_dir,
        _POOL_CACHE_MARKER,
        pool_fingerprint,
        _POOL_STAGE_FILES,
    ):
        _rp_build_pool(cache_sources)
        _write_stage_cache_marker(
            cache_dir,
            _POOL_CACHE_MARKER,
            "pool",
            pool_fingerprint,
            _POOL_STAGE_FILES,
        )

    pool_artifacts = _artifact_sha_table(cache_dir, _POOL_STAGE_FILES)
    corpus_fingerprint = _fp_stable_digest(
        {
            "schema_version": 2,
            "stage": "corpus",
            "seed": int(sources.seed),
            "recipe": _FIXED_RECIPE,
            "inputs": {
                key: inputs[key]
                for key in (
                    "inaturalist_train_archive",
                    "inaturalist_val_archive",
                    "inaturalist_train_image_root",
                    "inaturalist_val_image_root",
                )
            },
            "pool_artifacts": pool_artifacts,
        }
    )
    if not _stage_cache_reusable(
        cache_dir,
        _CORPUS_CACHE_MARKER,
        corpus_fingerprint,
        _CORPUS_STAGE_FILES,
        validate_pixels=True,
    ):
        _rc_build_pool_corpus(cache_sources)
        _write_stage_cache_marker(
            cache_dir,
            _CORPUS_CACHE_MARKER,
            "corpus",
            corpus_fingerprint,
            _CORPUS_STAGE_FILES,
        )
    return _publish_public_artifacts(
        sources,
        cache_dir,
        build_fingerprint,
        inputs,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed 11K CPU-only DualSearch RL dataset and "
            "retrieval corpora from explicitly configured local files."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Shared local JSON config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore a matching cache manifest and rebuild.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build(args.config, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
