"""Issue 23: synthetic causal features, alignment, train-only fit and paired CI."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from wildchat.temporal_features import H1_NAMES, h1_features

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TRAINING = HAS_NUMPY and all(importlib.util.find_spec(n) is not None for n in ("lightgbm", "pyarrow"))


class CausalityTests(unittest.TestCase):
    def same(self, a, b):
        self.assertEqual(len(a), len(b))
        for av, bv in zip(a, b):
            self.assertTrue(math.isnan(av) and math.isnan(bv) or av == bv)

    def test_known_values_ewma_std_difference_slope_ratios(self):
        actual = h1_features([1] * 4, [0, 1, 3, 4], [10, 20, 10, 40], [20, 60, 40, 100])
        self.assertTrue(all(math.isnan(v) for v in actual[0][:-1]))
        self.assertEqual(actual[0][-1], 0)
        self.same(actual[1], [20, 20, 0, math.nan, math.nan, 2, 2, 2, 1])
        self.same(actual[2], [32, 40, 20, 40, 40, 3, 2.3, .5, 2])
        self.assertAlmostEqual(actual[3][0], 34.4)
        self.assertAlmostEqual(actual[3][1], 40)
        self.assertAlmostEqual(actual[3][2], math.sqrt(800 / 3))
        self.same(actual[3][3:], [-20, -10, 4, .3 * 4 + .7 * 2.3, 4, 3])

    def test_current_and_all_future_responses_cannot_change_prefix(self):
        sources, turns, inputs, outputs = [1] * 5, list(range(5)), [2, 3, 5, 7, 11], [10, 20, 30, 40, 50]
        original = h1_features(sources, turns, inputs, outputs)
        for current in range(5):
            changed_y = outputs[:current] + [9999] * (5 - current)
            changed_x = inputs[:current + 1] + [8888] * (4 - current)
            changed = h1_features(sources, turns, changed_x, changed_y)
            for i in range(current + 1):
                self.same(original[i], changed[i])
            truncated = h1_features(sources[:current + 1], turns[:current + 1],
                                    inputs[:current + 1], outputs[:current + 1])
            self.same(original[current], truncated[-1])
        self.assertNotEqual(original[1][0], h1_features(sources, turns, inputs, [9999] + outputs[1:])[1][0])

    def test_interleaved_conversations_do_not_share_history(self):
        sources, turns, inputs, outputs = [1, 2, 1, 2, 1], [0, 0, 2, 1, 3], [10] * 5, [10, 9999, 30, 8888, 50]
        together = h1_features(sources, turns, inputs, outputs)
        alone = h1_features([1] * 3, [0, 2, 3], [10] * 3, [10, 30, 50])
        for a, b in zip([together[i] for i in (0, 2, 4)], alone):
            self.same(a, b)

    def test_current_prompt_is_available_only_input_multiplier_changes(self):
        first = h1_features([1, 1], [0, 1], [10, 20], [20, 50])[1]
        changed = h1_features([1, 1], [0, 1], [10, 100], [20, 50])[1]
        self.same(first[:7] + first[8:], changed[:7] + changed[8:])
        self.assertEqual((first[7], changed[7]), (2, 10))

    def test_zero_denominators_and_zero_output_are_well_defined(self):
        actual = h1_features([1] * 4, range(4), [10, 0, 20, 10], [0, 60, 40, 50])
        self.assertEqual(actual[1][5:8], [0, 0, 0])
        self.assertTrue(math.isnan(actual[2][5]))
        self.assertTrue(math.isnan(actual[2][7]))
        self.assertEqual(actual[2][6], 0)
        self.assertAlmostEqual(actual[3][6], .6)
        self.assertEqual(actual[3][-1], 3)

    def test_invalid_prior_pairs_are_skipped_with_original_turn_gap(self):
        from wildchat import prepare
        from wildchat.tests.test_prepare import CharacterEncoding, message
        source = {"conversation": [message("user", "aa"), message("assistant", "bbbb"),
                    message("user", "unanswered"), message("user", "cc"), message("assistant", "dddddddd"),
                    message("user", "ee"), message("assistant", "ff")]}
        samples = list(prepare.iter_samples(source, CharacterEncoding()))
        turns = [s["history_features"]["previous_user_turns"] for s in samples]
        self.assertEqual(turns, [0, 2, 3])
        actual = h1_features([1] * 3, turns,
                             [s["prompt_features"]["input_tokens"] for s in samples],
                             [s["target"]["output_tokens"] for s in samples])
        self.assertEqual(actual[1][-1], 1)
        self.assertEqual(actual[2][3:5], [4, 2])

    def test_out_of_order_or_invalid_numeric_inputs_fail(self):
        for turns in ([1, 1], [1, 0], [-1, 0]):
            with self.assertRaises(ValueError):
                h1_features([1, 1], turns, [10, 10], [20, 20])
        with self.assertRaises(ValueError):
            h1_features([1], [0], [10], [-1])
        with self.assertRaises(ValueError):
            h1_features([1], [0], [10], [math.inf])
        with self.assertRaises(ValueError):
            h1_features([1, 1], [0, 1], [10], [20, 20])


@unittest.skipUnless(HAS_TRAINING, "requires the existing WildChat numerical environment")
class ExperimentTests(unittest.TestCase):
    def test_cache_alignment_checksum_format_duplicates_and_extra_keys(self):
        import numpy as np
        from wildchat import temporal as tm
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, key in (("train", (1, 0)), ("validation", (2, 0))):
                (root / f"{split}.jsonl").write_text(json.dumps({"metadata": dict(zip(
                    ("source_row_id", "user_message_index"), key))}) + "\n")
            cache = root / "embeddings.npz"
            np.savez(cache, keys=np.array([[2, 0], [1, 0]]), vectors=np.array([[2.] * 384, [1.] * 384]))
            values, audit = tm.load_embeddings(root, cache, tm.rv.fingerprint(cache))
            self.assertEqual(values["train"][0, 0], 1)
            self.assertEqual(values["validation"][0, 0], 2)
            self.assertTrue(audit["exact_train_validation_coverage"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                tm.load_embeddings(root, cache, "incorrect")
            for keys, vectors in (([[1, 0], [1, 0]], np.ones((2, 384))),
                                  ([[1, 0]], np.ones((1, 384))),
                                  ([[1, 0], [2, 0], [3, 0]], np.ones((3, 384))),
                                  ([[1, 0], [2, 0]], np.ones((2, 383))),
                                  ([[1, 0], [2, 0]], np.full((2, 384), np.nan))):
                np.savez(cache, keys=np.array(keys), vectors=vectors)
                with self.assertRaises(ValueError):
                    tm.load_embeddings(root, cache, tm.rv.fingerprint(cache))

    def test_jsonl_guard_rejects_any_other_path_before_open(self):
        from wildchat import temporal as tm
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text("")
            with tm.allowed_jsonl_reads(root) as opened:
                (root / "train.jsonl").read_text()
                with self.assertRaises(PermissionError):
                    (root / "unapproved.jsonl").read_text()
            self.assertEqual(opened, {"train.jsonl"})

    def test_paired_bootstrap_matches_explicit_draws_for_both_differences(self):
        import numpy as np
        from wildchat import temporal as tm
        from wildchat.tests.test_revalidate import rows
        data = rows([10, 20, 40, 60, 90], ["a", "a", "b", "c", "c"], turns=[0, 1, 0, 1, 2])
        base = np.tile([1., 25, 80], (5, 1))
        predictions = {m: base + np.arange(5)[:, None] * (i + 1) for i, m in enumerate(tm.MODELS)}
        actual = tm.cg.bootstrap_diffs(data, predictions, 42, 100, tm.MODELS, tm.COMPARISONS)
        rng = np.random.Generator(np.random.PCG64(42))
        expected = {(c, w, a, b): [] for c in tm.rv.cohorts(data) for w in ("row", "user")
                    for a, b in tm.COMPARISONS}
        for _ in range(100):
            clusters = [np.flatnonzero(data.users == ["a", "b", "c"][i]) for i in rng.integers(0, 3, 3)]
            for cohort, mask in tm.rv.cohorts(data).items():
                selected = [ix[mask[ix]] for ix in clusters if mask[ix].any()]
                if not selected:
                    continue
                for a, b in tm.COMPARISONS:
                    diffs = [tm.rv.contributions(data.y[ix], predictions[a][ix])[:, 0] -
                             tm.rv.contributions(data.y[ix], predictions[b][ix])[:, 0] for ix in selected]
                    expected[(cohort, "row", a, b)].append(np.concatenate(diffs).mean())
                    expected[(cohort, "user", a, b)].append(np.mean([d.mean() for d in diffs]))
        for (cohort, weight, a, b), values in expected.items():
            entry = actual[cohort][weight][f"{a}-{b}"]
            self.assertEqual(entry["effective_replicates"], len(values))
            np.testing.assert_allclose(entry["ci95"], np.quantile(values, [.025, .975]), atol=1e-12)

    def test_full_run_never_fits_pca_or_models_to_validation(self):
        import lightgbm as lgb
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from wildchat import temporal as tm
        from wildchat.tests.test_experiment import BASE, sample
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts, all_samples = {}, []
            for split, groups in (("train", range(80)), ("validation", range(100, 120))):
                samples = [sample(group, turn, split) for group in groups for turn in range(5)]
                if split == "validation":
                    for s in samples:
                        s["target"]["output_tokens"] += 10000
                (root / f"{split}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
                parts[split] = {"samples": len(samples), "groups": len(groups), "conversations": len(groups),
                                "recorded_at_min": samples[0]["metadata"]["recorded_at"],
                                "recorded_at_max": samples[-1]["metadata"]["recorded_at"]}
                all_samples += samples
            source = root / "synthetic.parquet"
            pq.write_table(pa.table({"hashed_ip": [f"user-{i}" for i in range(120)],
                           "timestamp": [BASE + timedelta(days=i) for i in range(120)]}), source)
            (root / "manifest.json").write_text(json.dumps({"lags": 3, "partitions": parts,
                "input_files": [{"name": source.name, "rows": 120}],
                "boundaries": {"train_validation_cutoff": (BASE + timedelta(days=100)).isoformat()}}))
            embeddings = np.random.default_rng(42).normal(size=(500, 384)).astype(np.float16)
            embeddings[400:] += 1000
            cache = root / "embeddings.npz"
            keys = np.array([[s["metadata"]["source_row_id"], s["metadata"]["user_message_index"]] for s in all_samples])
            np.savez(cache, keys=keys, vectors=embeddings)
            reference = root / "reference.json"
            reference.write_text(json.dumps({"embeddings_sha256": tm.rv.fingerprint(cache)}))
            args = argparse.Namespace(prepared=root, source=source, embeddings=cache,
                embedding_reference=reference, output=root / "results", seed=42, threads=1,
                inner_fraction=.8, max_rounds=4, early_stopping_rounds=2, bootstrap_replicates=20)
            calls, pca_rows = [], []
            train_impl, pca_impl = lgb.train, tm.PCA.fit

            def checked_train(params, dataset, *pos, **kw):
                self.assertLess(float(np.max(dataset.get_label())), 1000)
                for valid in kw.get("valid_sets", []):
                    self.assertLess(float(np.max(valid.get_label())), 1000)
                calls.append("valid_sets" in kw)
                return train_impl(params, dataset, *pos, **kw)

            def checked_pca(pca, x, k):
                self.assertLess(float(np.max(x)), 100)
                pca_rows.append(len(x))
                return pca_impl(pca, x, k)

            with mock.patch.object(lgb, "train", checked_train), mock.patch.object(tm.PCA, "fit", checked_pca), redirect_stdout(io.StringIO()):
                result = tm.run(args)
            self.assertEqual(calls, [True] * 36 + [False] * 12)
            self.assertEqual(pca_rows, [320, 400])
            self.assertEqual(result["data_access"]["opened_jsonl_files"], ["train.jsonl", "validation.jsonl"])
            self.assertEqual([len(result["features"][m]) for m in tm.MODELS], [31, 40, 63, 72])
            self.assertEqual(result["splits"]["outer"]["checks"]["user_intersection"], 0)
            self.assertEqual(set(result["validation_metrics"]), {"all", "first_turn", "followup_turn"})
            self.assertEqual(len(result["bootstrap_ci95"]["all"]["row"]), 2)
            self.assertEqual(json.loads((args.output / "temporal_h1.json").read_text()), result)
            report = (args.output / "temporal_h1.md").read_text()
            self.assertIn("| outer/later | 100 | 20 | 20 |\n| inner/earlier | 320 |", report)
            changed = deepcopy(all_samples[400:])
            for s in changed:
                s["target"]["output_tokens"] *= 3
            (root / "validation.jsonl").write_text("".join(json.dumps(s) + "\n" for s in changed))
            embeddings[400:] *= 2
            np.savez(cache, keys=keys, vectors=embeddings)
            reference.write_text(json.dumps({"embeddings_sha256": tm.rv.fingerprint(cache)}))
            with redirect_stdout(io.StringIO()):
                second = tm.run(args)
            self.assertEqual(result["training"], second["training"])
            self.assertEqual(result["pca"], second["pca"])


if __name__ == "__main__":
    unittest.main()
