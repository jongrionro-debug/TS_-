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
        for name in ("classify.py", "category.py", "classify_sims.py", "category_features.py", "embed_save.py", "category_pca.py"):
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


@unittest.skipUnless(HAS_NUMPY, "requires NumPy")
class SimilarityTests(unittest.TestCase):
    def test_per_category_argmax_matches_assign(self):
        import numpy as np
        from wildchat import classify as cl
        from wildchat import classify_sims as cs
        rng = np.random.default_rng(0)
        proto_labels = [c for c in cl.CATEGORIES for _ in cl.TAXONOMY[c]]
        proto, prompts = rng.normal(size=(len(proto_labels), 8)), rng.normal(size=(50, 8))
        labels, best = cl.assign(prompts, proto, proto_labels)
        sims = cs.per_category(prompts, proto, proto_labels)
        self.assertEqual(sims.shape, (50, len(cl.CATEGORIES)))
        self.assertEqual([cl.CATEGORIES[i] for i in sims.argmax(axis=1)], labels)
        np.testing.assert_allclose(sims.max(axis=1), best)


@unittest.skipUnless(HAS_TRAINING, "requires training dependencies")
class FeatureVariantTests(unittest.TestCase):
    def test_label_codes_follow_taxonomy_order(self):
        from wildchat import category_features as cf
        from wildchat.classify import CATEGORIES
        self.assertEqual(cf.label_codes([CATEGORIES[3], CATEGORIES[0]]).tolist(), [3.0, 0.0])
        with self.assertRaises(KeyError):
            cf.label_codes(["unknown"])

    def test_fixed_fit_uses_categorical_signal(self):
        import numpy as np
        from argparse import Namespace
        from wildchat import category_features as cf
        rng = np.random.default_rng(1)
        n = 3000
        cat = rng.integers(0, 7, n).astype(float)
        x = np.column_stack([rng.normal(size=n), cat])
        y = np.where(cat % 2 == 0, 10.0, 100.0) + rng.normal(size=n)
        sel = {"parameters": {"num_leaves": 7, "min_data_in_leaf": 20}, "iterations": [100, 100, 100]}
        pred = cf.fit_fixed(x, y, x, ["noise", "category"], sel, Namespace(seed=1, threads=1), categorical=[1])
        self.assertEqual(pred.shape, (n, 3))
        self.assertTrue((pred >= 0).all())
        self.assertLess(np.abs(pred[:, 1] - y).mean(), 5)


@unittest.skipUnless(HAS_TRAINING, "requires training dependencies")
class PCATests(unittest.TestCase):
    def test_pca_uses_only_fit_rows_and_orders_components(self):
        import numpy as np
        from wildchat.category_pca import PCA
        rng = np.random.default_rng(2)
        x = rng.normal(size=(500, 6)) * np.array([10, 5, 1, 1, 1, 1])
        p = PCA().fit(x, 3)
        self.assertEqual(p.components.shape, (3, 6))
        self.assertTrue(p.explained[0] >= p.explained[1] >= p.explained[2])
        np.testing.assert_allclose(p.transform(x).mean(axis=0), 0, atol=1e-9)
        shifted = PCA().fit(x + 100, 3)
        np.testing.assert_allclose(shifted.mean, p.mean + 100)
        other = np.full((2, 6), 1e6)
        before = p.mean.copy(); p.transform(other)
        np.testing.assert_array_equal(p.mean, before)

    def test_design_appends_first_k_components(self):
        import numpy as np
        from types import SimpleNamespace
        from wildchat.category_pca import PCA, design
        rng = np.random.default_rng(3)
        emb = rng.normal(size=(40, 10))
        rows = SimpleNamespace(x=rng.normal(size=(40, 5)))
        p = PCA().fit(emb, 8)
        d = design(rows, emb, [0, 2], p, 4)
        self.assertEqual(d.shape, (40, 6))
        np.testing.assert_allclose(d[:, 2:], p.transform(emb)[:, :4])
