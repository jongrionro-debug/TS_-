"""Synthetic checks for issue 19 (condition C); no real data, no test split."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TRAINING = HAS_NUMPY and all(importlib.util.find_spec(n) is not None for n in ("lightgbm", "pyarrow"))
ROOT = Path(__file__).resolve().parents[1]


class NoTestAccessTests(unittest.TestCase):
    def test_sources_never_reference_test_split(self):
        for name in ("classify.py", "category.py"):
            self.assertNotIn("test.jsonl", (ROOT / name).read_text())


@unittest.skipUnless(HAS_NUMPY, "requires NumPy")
class ClassifyTests(unittest.TestCase):
    def test_nearest_prototype_by_max_cosine(self):
        import numpy as np
        from wildchat import classify as cl
        k = len(cl.CATEGORIES)
        proto = np.eye(k)
        prompts = np.array([[0.1] * k, [0.0] * k])
        prompts[0, 2] = 5
        prompts[1, 0] = 1
        labels, _ = cl.assign(prompts, proto, list(cl.CATEGORIES))
        self.assertEqual(labels, [cl.CATEGORIES[2], cl.CATEGORIES[0]])

    def test_split_files_exclude_test(self):
        from wildchat import classify as cl
        self.assertEqual(set(cl.SPLIT_FILES), {"train", "validation"})


@unittest.skipUnless(HAS_TRAINING, "requires the WildChat experiment environment")
class CategoryTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from wildchat import category as ct
        from wildchat import revalidate as rv
        self.np, self.ct = np, ct
        n = 6
        base = rv.Rows(np.zeros((n, 31)), np.array([10., 10, 10, 10, 20, 20]), np.arange(n), np.arange(n),
                       np.array([0, 1, 0, 1, 0, 0]), np.arange(n), np.array(["u1", "u1", "u1", "u1", "u2", "u3"], dtype=object),
                       np.full(n, np.nan), np.full(n, np.nan))
        self.rows = ct.with_categories(base, np.array(list("aabbab"), dtype=object))

    def test_take_keeps_categories(self):
        sub = self.rows.take(self.rows.categories == "b")
        self.assertEqual(sub.y.tolist(), [10, 10, 20])

    def test_row_and_user_weighting(self):
        np = self.np
        pred = {m: np.tile([0., 0, 0], (6, 1)) for m in self.ct.MODELS}
        # loss for y with all-zero predictions = y*(0.1+0.5+0.9)/3 = 0.5*y
        t = self.ct.score_table(self.rows, pred, self.ct.MODELS)
        self.assertAlmostEqual(t["all"]["row"]["A"]["mean_pinball_loss"], 0.5 * 80 / 6)
        self.assertAlmostEqual(t["all"]["user"]["A"]["mean_pinball_loss"], 0.5 * (10 + 20 + 20) / 3)
        self.assertEqual(t["followup_turn"]["row"]["A"]["n_rows"], 2)

    def test_bootstrap_identical_models_zero(self):
        np = self.np
        pred = {m: np.tile([1., 5, 9], (6, 1)) for m in self.ct.MODELS}
        b = self.ct.bootstrap_diffs(self.rows, pred, 0, 20)
        self.assertEqual(b["all"]["row"]["C-B"]["ci95"], [0.0, 0.0])

    def test_categories_align_with_prepared_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for split, ids in (("train", [(1, 0), (1, 2)]), ("validation", [(5, 0)])):
                (d / f"{split}.jsonl").write_text("".join(json.dumps({"metadata": {
                    "source_row_id": s, "user_message_index": m}}) + "\n" for s, m in ids))
            (d / "labels.jsonl").write_text("".join(json.dumps({"source_row_id": s, "user_message_index": m,
                "category": c}) + "\n" for s, m, c in ((5, 0, "z"), (1, 2, "y"), (1, 0, "x"))))
            cats = self.ct.read_categories(d, d / "labels.jsonl")
        self.assertEqual(cats["train"].tolist(), ["x", "y"])
        self.assertEqual(cats["validation"].tolist(), ["z"])


if __name__ == "__main__":
    unittest.main()
