import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from dual_search.search.vision_retrieval import make_image_embedding_input
from dual_search.data.fingerprints import corpus_fingerprint, stable_digest
from dual_search.search.vision_retrieval_server import Config, VisionDenseRetriever, VisionSearchQuery


class _CapturingEncoder:
    def __init__(self):
        self.inputs = []

    def encode(self, inputs):
        self.inputs.append(inputs)
        return np.ones((len(inputs), 2), dtype=np.float32)


class _FakeIndex:
    def search(self, embeddings, k):
        batch_size = embeddings.shape[0]
        return (
            np.ones((batch_size, k), dtype=np.float32),
            np.zeros((batch_size, k), dtype=np.int64),
        )


class VisionRetrievalQueryTest(unittest.TestCase):
    def test_mixed_embedding_input_preserves_image_and_text(self):
        item = make_image_embedding_input({"image": "/tmp/query.jpg", "query": "wing pattern"})
        self.assertEqual(item, {"image": "/tmp/query.jpg", "text": "wing pattern"})

    def test_bare_corpus_image_has_no_text(self):
        item = make_image_embedding_input("/tmp/corpus.jpg")
        self.assertEqual(item, {"image": "/tmp/corpus.jpg"})

    def test_server_schema_requires_strict_positive_index_and_nonempty_query(self):
        valid = VisionSearchQuery(image="/tmp/query.jpg", image_index=1, query="hint")
        self.assertEqual(valid.image_index, 1)
        for kwargs in (
            {"image": "/tmp/query.jpg", "image_index": True, "query": "hint"},
            {"image": "/tmp/query.jpg", "image_index": 0, "query": "hint"},
            {"image": "/tmp/query.jpg", "image_index": 1, "query": "   "},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                VisionSearchQuery(**kwargs)

    def test_single_search_uses_joint_image_text_embedding(self):
        retriever = object.__new__(VisionDenseRetriever)
        retriever.topk = 1
        retriever.encoder = _CapturingEncoder()
        retriever.index = _FakeIndex()
        retriever.corpus = [{"contents": '"candidate"\ncaption'}]

        query = VisionSearchQuery(image="/tmp/query.jpg", image_index=1, query="wing pattern")
        docs = retriever.search(query, num=1)

        self.assertEqual(
            retriever.encoder.inputs,
            [[{"image": "/tmp/query.jpg", "text": "wing pattern"}]],
        )
        self.assertEqual(docs[0]["contents"], '"candidate"\ncaption')

    def test_batch_search_uses_each_query_text(self):
        retriever = object.__new__(VisionDenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = _CapturingEncoder()
        retriever.index = _FakeIndex()
        retriever.corpus = [{"contents": '"candidate"\ncaption'}]
        queries = [
            VisionSearchQuery(image="/tmp/one.jpg", image_index=1, query="first hint"),
            VisionSearchQuery(image="/tmp/two.jpg", image_index=1, query="second hint"),
        ]

        retriever.batch_search(queries, num=1)

        self.assertEqual(
            retriever.encoder.inputs,
            [
                [
                    {"image": "/tmp/one.jpg", "text": "first hint"},
                    {"image": "/tmp/two.jpg", "text": "second hint"},
                ]
            ],
        )

    def test_service_rejects_encoder_mismatch_before_loading_faiss_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "vision.jsonl"
            corpus.write_text(
                '{"id":"inaturalist:1","image_key":"inaturalist:1","image":"one.jpg"}\n',
                encoding="utf-8",
            )
            encoder_config = {
                "encoder": "Qwen3VLImageEncoder",
                "model_reference": "/models/qwen-one",
                "normalize_embeddings": True,
                "truncate_dim": None,
                "corpus_input_mode": "image_only",
                "query_input_mode": "image_text_joint",
            }
            meta = root / "vision_index_meta.json"
            meta.write_text(
                json.dumps(
                    {
                        "index_kind": "vision",
                        "corpus_fingerprint": corpus_fingerprint(
                            corpus, id_keys=("id", "image_key")
                        ),
                        "encoder_config": encoder_config,
                        "encoder_config_sha256": stable_digest(encoder_config),
                    }
                ),
                encoding="utf-8",
            )
            config = Config(
                index_path=str(root / "does-not-need-to-exist.index"),
                corpus_path=str(corpus),
                retrieval_model_path="/models/qwen-two",
                faiss_gpu=False,
                meta_path=str(meta),
            )

            with self.assertRaisesRegex(ValueError, "encoder configuration mismatch"):
                VisionDenseRetriever(config)


if __name__ == "__main__":
    unittest.main()
