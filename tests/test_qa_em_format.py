import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "verl/utils/reward_score/qa_em_format.py"
SPEC = importlib.util.spec_from_file_location("dual_search_qa_em_format_test_module", MODULE_PATH)
qa_em_format = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qa_em_format)


class QAEMRetrievalPenaltyTest(unittest.TestCase):
    def test_ppo_uses_the_same_first_two_free_retrieval_penalty(self):
        solution = "<think>done</think><answer>Bronze Copper</answer>"
        ground_truth = {"target": ["Bronze Copper"]}

        first_two = qa_em_format.compute_score(
            solution,
            ground_truth,
            extra_info={"valid_search_stats": 1, "valid_vision_search_stats": 1},
        )
        third = qa_em_format.compute_score(
            solution,
            ground_truth,
            extra_info={"valid_search_stats": 2, "valid_vision_search_stats": 1},
        )
        eighth = qa_em_format.compute_score(
            solution,
            ground_truth,
            extra_info={"valid_search_stats": 4, "valid_vision_search_stats": 4},
        )

        self.assertEqual(first_two, 1.0)
        self.assertAlmostEqual(third, 0.98)
        self.assertAlmostEqual(eighth, 0.28)


if __name__ == "__main__":
    unittest.main()
