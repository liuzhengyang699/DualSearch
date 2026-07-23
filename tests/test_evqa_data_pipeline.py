import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from dual_search.data.evqa_pipeline import (
    _publish_artifact_set,
    _sample_id,
    build_catalog,
    build_corpus,
    build_split,
    load_sources,
    parse_answers,
    run_stage,
)
from dual_search.data.fingerprints import (
    assert_corpus_fingerprint,
    corpus_fingerprint,
    load_and_validate_index_meta,
    sha256_file,
    stable_digest,
    validate_embedding_sidecar,
)


CSV_FIELDS = [
    "wikipedia_title",
    "wikipedia_url",
    "question_original",
    "question",
    "question_type",
    "answer",
    "multi_answer",
    "evidence",
    "evidence_section_id",
    "evidence_section_title",
    "dataset_name",
    "dataset_category_id",
    "dataset_image_ids",
]


def image_bytes(color):
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(stream, format="PNG")
    return stream.getvalue()


def write_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evqa_row(**overrides):
    row = {
        "wikipedia_title": "Species A",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Species_A",
        "question_original": "Original question?",
        "question": "What is this species known for?",
        "question_type": "single_hop",
        "answer": "Answer A",
        "multi_answer": "",
        "evidence": "Evidence A",
        "evidence_section_id": "0",
        "evidence_section_title": "Intro",
        "dataset_name": "inaturalist",
        "dataset_category_id": "100",
        "dataset_image_ids": "1",
    }
    row.update(overrides)
    return row


class SyntheticEVQAPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train_root = self.root / "images" / "train"
        self.val_root = self.root / "images" / "val"
        self.train_metadata = self.root / "inat_train.json"
        self.val_metadata = self.root / "inat_val.json"
        categories = [
            {"id": 100, "name": "Species alpha", "common_name": "Alpha"},
            {"id": 200, "name": "Species beta", "common_name": "Beta"},
        ]
        # annotations intentionally precede images to exercise the streaming catalog.
        self.train_metadata.write_text(
            json.dumps(
                {
                    "annotations": [
                        {"image_id": 1, "category_id": 100},
                        {"image_id": 2, "category_id": 100},
                        {"image_id": 3, "category_id": 100},
                        {"image_id": 4, "category_id": 200},
                    ],
                    "images": [
                        {"id": 1, "file_name": "train/1.png"},
                        {"id": 2, "file_name": "train/2.png"},
                        {"id": 3, "file_name": "train/3.png"},
                        {"id": 4, "file_name": "train/4.png"},
                    ],
                    "categories": categories,
                }
            ),
            encoding="utf-8",
        )
        self.val_metadata.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 10, "file_name": "val/10.png"},
                        {"id": 11, "file_name": "val/11.png"},
                        {"id": 12, "file_name": "val/12.png"},
                    ],
                    "categories": categories,
                    "annotations": [
                        {"image_id": 10, "category_id": 100},
                        {"image_id": 11, "category_id": 100},
                        {"image_id": 12, "category_id": 100},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.train_archive = self.root / "train.tar.gz"
        self.val_archive = self.root / "val.tar.gz"
        write_archive(
            self.train_archive,
            {
                "train/1.png": image_bytes("red"),
                "train/2.png": image_bytes("green"),
                "train/3.png": image_bytes("blue"),
                "train/4.png": image_bytes("yellow"),
            },
        )
        write_archive(
            self.val_archive,
            {
                "val/10.png": image_bytes("purple"),
                "val/11.png": image_bytes("orange"),
                "val/12.png": b"not-an-image",
            },
        )
        self.train_csv = self.root / "train.csv"
        self.val_csv = self.root / "val.csv"
        write_csv(
            self.train_csv,
            [
                evqa_row(dataset_image_ids="1|2"),
                evqa_row(
                    wikipedia_title="Missing species",
                    wikipedia_url="https://en.wikipedia.org/wiki/Missing_species",
                    question="What is beta?",
                    answer="Beta answer",
                    dataset_category_id="200",
                    dataset_image_ids="4",
                ),
                evqa_row(
                    dataset_name="gldv2",
                    dataset_category_id="landmark-1",
                    dataset_image_ids="unused",
                ),
            ],
        )
        write_csv(
            self.val_csv,
            [
                evqa_row(
                    wikipedia_title="Species A|Second page",
                    wikipedia_url=(
                        "https://en.wikipedia.org/wiki/Species_A|"
                        "https://en.wikipedia.org/wiki/Second_page"
                    ),
                    question="What links these facts?",
                    question_type="2_hop",
                    answer="Linked answer",
                    evidence_section_id="0|0",
                    dataset_image_ids="10",
                )
            ],
        )
        self.kb = self.root / "kb.json"
        self.kb.write_text(
            json.dumps(
                {
                    "https://en.wikipedia.org/wiki/Species_A": {
                        "title": "Species A",
                        "section_titles": ["Intro"],
                        "section_texts": ["Species A evidence."],
                    },
                    "https://en.wikipedia.org/wiki/Second_page": {
                        "title": "Second page",
                        "section_titles": ["Intro"],
                        "section_texts": ["Second-hop evidence."],
                    },
                    "https://en.wikipedia.org/wiki/Unused": {
                        "title": "Unused",
                        "section_titles": ["Intro"],
                        "section_texts": ["Must not enter the corpus."],
                    },
                }
            ),
            encoding="utf-8",
        )
        self.output_dir = self.root / "output"
        self.config_path = self.root / "sources.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "output_dir": str(self.output_dir),
                    "seed": 42,
                    "evqa": {
                        "train_csv": str(self.train_csv),
                        "val_csv": str(self.val_csv),
                    },
                    "inaturalist": {
                        "train": {
                            "metadata": str(self.train_metadata),
                            "archive": str(self.train_archive),
                            "image_root": str(self.train_root),
                        },
                        "val": {
                            "metadata": str(self.val_metadata),
                            "archive": str(self.val_archive),
                            "image_root": str(self.val_root),
                        },
                    },
                    "wikipedia": {"kb": str(self.kb)},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_full_local_pipeline_is_leak_free_and_keeps_unresolvable_rows(self):
        import pandas as pd

        sources = load_sources(self.config_path)
        catalog_report = build_catalog(sources)
        self.assertEqual(catalog_report["images"], 7)

        split_report = build_split(sources)
        self.assertEqual(split_report["schema_version"], 2)
        self.assertEqual(split_report["train_samples"], 2)
        self.assertEqual(split_report["test_samples"], 1)
        self.assertEqual(split_report["gldv2_rows_skipped"], 1)
        self.assertEqual(
            split_report["distribution"]["train"],
            {
                "rl_questions": 2,
                "query_images": 3,
                "single_image_questions": 1,
                "multi_image_questions": 1,
                "image_count_histogram": {"1": 1, "2": 1},
                "by_question_type": {
                    "single_hop": {"query_images": 3, "questions": 2}
                },
                "by_category": {
                    "inaturalist:100": {"query_images": 2, "questions": 1},
                    "inaturalist:200": {"query_images": 1, "questions": 1},
                },
            },
        )
        logical_train = [
            json.loads(line)
            for line in (self.output_dir / "logical_train.jsonl").read_text().splitlines()
        ]
        multi_logical = next(
            row for row in logical_train if row["category_key"] == "inaturalist:100"
        )
        self.assertEqual(multi_logical["schema_version"], 2)
        self.assertEqual(multi_logical["image_count"], 2)
        self.assertEqual(multi_logical["image_keys"], ["inaturalist:1", "inaturalist:2"])
        self.assertEqual(
            [
                (image["image_index"], image["dataset_image_id"])
                for image in multi_logical["query_images"]
            ],
            [(1, "1"), (2, "2")],
        )
        for obsolete in (
            "dataset_image_id",
            "all_image_ids",
            "image_key",
            "image",
            "source_file_name",
            "source_split",
        ):
            self.assertNotIn(obsolete, multi_logical)

        manifest = build_corpus(sources)
        from dual_search.data.sft_builder import _extract_heldout_image_keys

        self.assertEqual(
            _extract_heldout_image_keys(manifest),
            {"inaturalist:1", "inaturalist:2", "inaturalist:4", "inaturalist:10"},
        )
        vision_rows = [
            json.loads(line)
            for line in (self.output_dir / "vision_corpus.jsonl").read_text().splitlines()
        ]
        self.assertEqual({row["image_key"] for row in vision_rows}, {"inaturalist:3", "inaturalist:11"})
        heldout = json.loads((self.output_dir / "heldout_manifest.json").read_text())
        self.assertTrue(set(heldout["image_keys"]).isdisjoint({row["image_key"] for row in vision_rows}))

        train = pd.read_parquet(self.output_dir / "train.parquet")
        test = pd.read_parquet(self.output_dir / "test.parquet")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 1)
        self.assertEqual(sorted(train["positive_candidate_count"].tolist()), [0, 2])
        self.assertEqual(train["retrieval_resolvable"].tolist().count(False), 1)
        self.assertTrue(bool(test.iloc[0]["retrieval_resolvable"]))
        multi_rl = train.loc[train["category_key"] == "inaturalist:100"].iloc[0]
        self.assertEqual(int(multi_rl["schema_version"]), 2)
        self.assertEqual(multi_rl["dataset_category_id"], "100")
        self.assertEqual(int(multi_rl["image_count"]), 2)
        self.assertEqual(list(multi_rl["image_keys"]), ["inaturalist:1", "inaturalist:2"])
        self.assertEqual(
            [image["image_index"] for image in multi_rl["query_images"]],
            [1, 2],
        )
        self.assertEqual(len(multi_rl["images"]), 2)
        prompt = multi_rl["prompt"][0]["content"]
        self.assertEqual(prompt.count("<image>"), 2)
        self.assertLess(prompt.index("Image 1:\n<image>"), prompt.index("Image 2:\n<image>"))
        self.assertIn("Images are numbered from 1", prompt)
        self.assertEqual(
            manifest["report"]["distribution"]["overall"]["rl_questions"], 3
        )
        self.assertEqual(
            manifest["report"]["distribution"]["overall"]["query_images"], 4
        )
        self.assertEqual(
            manifest["report"]["distribution"]["overall"]["retrieval_resolvable"], 2
        )
        self.assertEqual(
            manifest["report"]["image_materialization"]["query_images"], 4
        )
        self.assertEqual(
            manifest["report"]["image_materialization"]["dropped_candidates"][0]["image_key"],
            "inaturalist:12",
        )
        expected_inputs = {
            "evqa_train_csv": self.train_csv,
            "evqa_val_csv": self.val_csv,
            "inaturalist_train_metadata": self.train_metadata,
            "inaturalist_val_metadata": self.val_metadata,
            "inaturalist_train_archive": self.train_archive,
            "inaturalist_val_archive": self.val_archive,
            "wikipedia_kb": self.kb,
        }
        for key, path in expected_inputs.items():
            self.assertEqual(manifest["inputs"][key]["path"], str(path))
            self.assertEqual(manifest["inputs"][key]["sha256"], sha256_file(path))

        text = (self.output_dir / "text_corpus.jsonl").read_text(encoding="utf-8")
        self.assertIn("Second-hop evidence", text)
        self.assertNotIn("Must not enter the corpus", text)

    def test_query_in_wrong_official_split_fails_before_split_outputs(self):
        write_csv(self.val_csv, [evqa_row(dataset_image_ids="1")])
        sources = load_sources(self.config_path)
        build_catalog(sources)
        with self.assertRaises(RuntimeError):
            build_split(sources)
        self.assertFalse((self.output_dir / "logical_train.jsonl").exists())
        self.assertFalse((self.output_dir / "logical_test.jsonl").exists())
        preflight = json.loads((self.output_dir / "split_preflight_report.json").read_text())
        self.assertEqual(preflight["missing_queries"][0]["reason"], "found_only_in_wrong_split")

    def test_duplicate_query_ids_are_deduplicated_without_reordering(self):
        write_csv(self.train_csv, [evqa_row(dataset_image_ids="2|1|2")])
        sources = load_sources(self.config_path)
        build_catalog(sources)
        build_split(sources)

        logical = json.loads(
            (self.output_dir / "logical_train.jsonl").read_text(encoding="utf-8").strip()
        )
        self.assertEqual(logical["image_keys"], ["inaturalist:2", "inaturalist:1"])
        self.assertEqual(
            [image["image_index"] for image in logical["query_images"]],
            [1, 2],
        )

    def test_query_image_order_changes_parent_sample_id(self):
        write_csv(self.train_csv, [evqa_row(dataset_image_ids="1|2")])
        sources = load_sources(self.config_path)
        build_catalog(sources)
        build_split(sources)
        first = json.loads(
            (self.output_dir / "logical_train.jsonl").read_text(encoding="utf-8").strip()
        )

        write_csv(self.train_csv, [evqa_row(dataset_image_ids="2|1")])
        build_split(sources)
        second = json.loads(
            (self.output_dir / "logical_train.jsonl").read_text(encoding="utf-8").strip()
        )

        self.assertNotEqual(first["sample_id"], second["sample_id"])
        self.assertEqual(second["image_keys"], ["inaturalist:2", "inaturalist:1"])

    def test_cross_category_query_images_fail_split_atomically(self):
        write_csv(
            self.train_csv,
            [evqa_row(dataset_category_id="", dataset_image_ids="1|4")],
        )
        sources = load_sources(self.config_path)
        build_catalog(sources)

        with self.assertRaises(RuntimeError):
            build_split(sources)

        self.assertFalse((self.output_dir / "logical_train.jsonl").exists())
        self.assertFalse((self.output_dir / "logical_test.jsonl").exists())
        preflight = json.loads(
            (self.output_dir / "split_preflight_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            preflight["category_mismatches"][0]["reason"],
            "inconsistent_query_categories",
        )

    def test_corpus_rejects_v1_logical_rows(self):
        sources = load_sources(self.config_path)
        build_catalog(sources)
        build_split(sources)
        logical_path = self.output_dir / "logical_train.jsonl"
        rows = [json.loads(line) for line in logical_path.read_text().splitlines()]
        rows[0]["schema_version"] = 1
        logical_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "requires schema_version=2"):
            build_corpus(sources)

        self.assertFalse((self.output_dir / "train.parquet").exists())
        self.assertFalse((self.output_dir / "test.parquet").exists())
        preflight = json.loads(
            (self.output_dir / "corpus_preflight_report.json").read_text(encoding="utf-8")
        )
        self.assertIn("Rerun catalog, split, corpus, and sft", preflight["error"])

    def test_corpus_rejects_logical_heldout_mismatch_after_valid_fingerprint(self):
        sources = load_sources(self.config_path)
        build_catalog(sources)
        build_split(sources)
        logical_path = self.output_dir / "logical_train.jsonl"
        rows = [json.loads(line) for line in logical_path.read_text().splitlines()]
        multi = next(row for row in rows if row["image_count"] == 2)
        single = next(row for row in rows if row["image_count"] == 1)
        replacement = dict(single["query_images"][0])
        replacement["image_index"] = 1
        multi["query_images"][0] = replacement
        multi["image_keys"][0] = replacement["image_key"]
        multi["sample_id"] = _sample_id(
            multi,
            multi["official_split"],
            multi["source_row_index"],
            multi["image_keys"],
        )
        logical_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        split_report_path = self.output_dir / "split_report.json"
        split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
        refreshed = corpus_fingerprint(logical_path)
        refreshed["path"] = str(logical_path)
        split_report["artifacts"]["logical_train"] = refreshed
        split_report_path.write_text(
            json.dumps(split_report),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Heldout manifest does not exactly match"):
            build_corpus(sources)

        self.assertFalse((self.output_dir / "train.parquet").exists())
        self.assertFalse((self.output_dir / "test.parquet").exists())

    def test_artifact_set_publish_restores_previous_generation_on_failure(self):
        staged = {
            "first": self.root / "staged-first.txt",
            "commit": self.root / "staged-commit.txt",
        }
        final = {
            "first": self.root / "first.txt",
            "commit": self.root / "commit.txt",
        }
        for path, content in (
            (staged["first"], "new-first"),
            (staged["commit"], "new-commit"),
            (final["first"], "old-first"),
            (final["commit"], "old-commit"),
        ):
            path.write_text(content, encoding="utf-8")

        real_replace = __import__("os").replace
        published = 0

        def fail_second_publish(source, destination):
            nonlocal published
            if Path(source) in set(staged.values()):
                published += 1
                if published == 2:
                    raise OSError("simulated publish failure")
            return real_replace(source, destination)

        with patch(
            "dual_search.data.evqa_pipeline.os.replace",
            side_effect=fail_second_publish,
        ), self.assertRaisesRegex(OSError, "simulated publish failure"):
            _publish_artifact_set(staged, final)

        self.assertEqual(final["first"].read_text(encoding="utf-8"), "old-first")
        self.assertEqual(final["commit"].read_text(encoding="utf-8"), "old-commit")
        self.assertFalse(list(self.root.glob(".*.backup")))

    def test_corpus_reordering_invalidates_fingerprint(self):
        corpus = self.root / "corpus.jsonl"
        corpus.write_text(
            '{"id":"a","contents":"A"}\n{"id":"b","contents":"B"}\n',
            encoding="utf-8",
        )
        fingerprint = corpus_fingerprint(corpus, id_keys=("id",))
        corpus.write_text(
            '{"id":"b","contents":"B"}\n{"id":"a","contents":"A"}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            assert_corpus_fingerprint(fingerprint, corpus, id_keys=("id",))

    def test_index_fingerprint_rejects_a_different_encoder_configuration(self):
        corpus = self.root / "fingerprint_corpus.jsonl"
        corpus.write_text('{"id":"a","contents":"A"}\n', encoding="utf-8")
        encoder_config = {
            "encoder": "BGEM3DenseEncoder",
            "model_reference": "/models/bge-m3",
            "normalize_embeddings": True,
            "max_length": 8192,
            "use_fp16": True,
            "input_mode": "text",
        }
        meta = self.root / "text_index_meta.json"
        meta.write_text(
            json.dumps(
                {
                    "index_kind": "text",
                    "corpus_fingerprint": corpus_fingerprint(corpus, id_keys=("id",)),
                    "encoder_config": encoder_config,
                    "encoder_config_sha256": stable_digest(encoder_config),
                }
            ),
            encoding="utf-8",
        )

        load_and_validate_index_meta(
            meta,
            corpus,
            expected_kind="text",
            id_keys=("id",),
            expected_encoder_config=encoder_config,
        )
        mismatched = {**encoder_config, "model_reference": "/models/other-bge"}
        with self.assertRaisesRegex(ValueError, "encoder configuration mismatch"):
            load_and_validate_index_meta(
                meta,
                corpus,
                expected_kind="text",
                id_keys=("id",),
                expected_encoder_config=mismatched,
            )

    def test_precomputed_embedding_sidecar_binds_encoder_and_corpus(self):
        corpus = self.root / "embedding_corpus.jsonl"
        corpus.write_text('{"id":"a","contents":"A"}\n', encoding="utf-8")
        embedding = self.root / "embedding.memmap"
        embedding.write_bytes(b"\x00" * 8)
        encoder_config = {"encoder": "fixture", "model_reference": "model-a"}
        sidecar = self.root / "embedding.memmap.meta.json"
        sidecar.write_text(
            json.dumps(
                {
                    "row_count": 1,
                    "embedding_dim": 2,
                    "corpus_fingerprint": corpus_fingerprint(corpus, id_keys=("id",)),
                    "encoder_config": encoder_config,
                    "encoder_config_sha256": stable_digest(encoder_config),
                }
            ),
            encoding="utf-8",
        )

        validate_embedding_sidecar(
            embedding,
            sidecar,
            corpus,
            expected_rows=1,
            expected_dim=2,
            id_keys=("id",),
            expected_encoder_config=encoder_config,
        )
        with self.assertRaisesRegex(ValueError, "encoder configuration mismatch"):
            validate_embedding_sidecar(
                embedding,
                sidecar,
                corpus,
                expected_rows=1,
                expected_dim=2,
                id_keys=("id",),
                expected_encoder_config={"encoder": "fixture", "model_reference": "model-b"},
            )

    def test_answer_variants_preserve_multi_answer_components(self):
        self.assertEqual(
            parse_answers({"answer": "red&&blue|scarlet&&navy"}),
            ["red && blue", "scarlet && navy"],
        )

    def test_all_stage_uses_the_declared_dependency_order(self):
        order = []

        def stage(name):
            return lambda sources: order.append(name) or {"stage": name}

        with (
            patch("dual_search.data.evqa_pipeline.build_catalog", side_effect=stage("catalog")),
            patch("dual_search.data.evqa_pipeline.build_split", side_effect=stage("split")),
            patch("dual_search.data.evqa_pipeline.build_corpus", side_effect=stage("corpus")),
            patch("dual_search.data.evqa_pipeline.build_sft", side_effect=stage("sft")),
            patch("dual_search.data.evqa_pipeline.build_indexes", side_effect=stage("index")),
        ):
            report = run_stage("all", object())

        self.assertEqual(order, ["catalog", "split", "corpus", "sft", "index"])
        self.assertEqual(list(report["stages"]), order)

    def test_metadata_path_traversal_fails_catalog_atomically(self):
        metadata = json.loads(self.train_metadata.read_text(encoding="utf-8"))
        metadata["images"][0]["file_name"] = "../escape.png"
        self.train_metadata.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            build_catalog(load_sources(self.config_path))

        self.assertFalse((self.output_dir / "catalog.sqlite").exists())
        preflight = json.loads(
            (self.output_dir / "catalog_preflight_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(preflight["status"], "failed")
        self.assertFalse(preflight["partial_catalog_written"])
        self.assertFalse((self.root / "escape.png").exists())

    def test_unsafe_archive_member_fails_corpus_without_parquet(self):
        sources = load_sources(self.config_path)
        build_catalog(sources)
        build_split(sources)
        write_archive(
            self.train_archive,
            {
                "../escape.png": image_bytes("black"),
                "train/1.png": image_bytes("red"),
                "train/2.png": image_bytes("green"),
                "train/3.png": image_bytes("blue"),
                "train/4.png": image_bytes("yellow"),
            },
        )

        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            build_corpus(sources)

        self.assertFalse((self.output_dir / "train.parquet").exists())
        self.assertFalse((self.output_dir / "test.parquet").exists())
        preflight = json.loads(
            (self.output_dir / "corpus_preflight_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(preflight["status"], "failed")
        self.assertFalse(preflight["partial_parquet_written"])
        self.assertFalse((self.root / "escape.png").exists())

    def test_corrupt_query_pixel_fails_corpus_without_parquet(self):
        sources = load_sources(self.config_path)
        build_catalog(sources)
        build_split(sources)
        write_archive(
            self.train_archive,
            {
                "train/1.png": b"corrupt-query-pixel",
                "train/2.png": image_bytes("green"),
                "train/3.png": image_bytes("blue"),
                "train/4.png": image_bytes("yellow"),
            },
        )

        with self.assertRaisesRegex(RuntimeError, "corrupt_query_pixel"):
            build_corpus(sources)

        self.assertFalse((self.output_dir / "train.parquet").exists())
        self.assertFalse((self.output_dir / "test.parquet").exists())
        preflight = json.loads(
            (self.output_dir / "corpus_preflight_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(preflight["error_type"], "RuntimeError")
        self.assertIn("corrupt_query_pixel", preflight["error"])
        failure = json.loads(preflight["error"])["missing_query_images"][0]
        self.assertTrue(failure["sample_id"].startswith("evqa:"))
        self.assertEqual(failure["image_index"], 1)
        self.assertEqual(failure["image_key"], "inaturalist:1")
        self.assertEqual(Path(failure["path"]).name, "1.png")

    def test_missing_query_archive_member_fails_corpus_without_parquet(self):
        sources = load_sources(self.config_path)
        build_catalog(sources)
        build_split(sources)
        write_archive(
            self.train_archive,
            {
                "train/2.png": image_bytes("green"),
                "train/3.png": image_bytes("blue"),
                "train/4.png": image_bytes("yellow"),
            },
        )

        with self.assertRaisesRegex(RuntimeError, "missing_query_archive_member_or_file"):
            build_corpus(sources)

        self.assertFalse((self.output_dir / "train.parquet").exists())
        self.assertFalse((self.output_dir / "test.parquet").exists())
        preflight = json.loads(
            (self.output_dir / "corpus_preflight_report.json").read_text(encoding="utf-8")
        )
        self.assertIn("missing_query_archive_member_or_file", preflight["error"])
        failure = json.loads(preflight["error"])["missing_query_images"][0]
        self.assertTrue(failure["sample_id"].startswith("evqa:"))
        self.assertEqual(failure["image_index"], 1)
        self.assertEqual(failure["image_key"], "inaturalist:1")

    def test_corrupt_second_query_image_reports_parent_and_source_index(self):
        sources = load_sources(self.config_path)
        build_catalog(sources)
        build_split(sources)
        write_archive(
            self.train_archive,
            {
                "train/1.png": image_bytes("red"),
                "train/2.png": b"corrupt-second-query-pixel",
                "train/3.png": image_bytes("blue"),
                "train/4.png": image_bytes("yellow"),
            },
        )

        with self.assertRaisesRegex(RuntimeError, "corrupt_query_pixel"):
            build_corpus(sources)

        preflight = json.loads(
            (self.output_dir / "corpus_preflight_report.json").read_text(encoding="utf-8")
        )
        failure = json.loads(preflight["error"])["missing_query_images"][0]
        self.assertTrue(failure["sample_id"].startswith("evqa:"))
        self.assertEqual(failure["image_index"], 2)
        self.assertEqual(failure["image_key"], "inaturalist:2")
        self.assertFalse((self.output_dir / "train.parquet").exists())
        self.assertFalse((self.output_dir / "test.parquet").exists())


if __name__ == "__main__":
    unittest.main()
