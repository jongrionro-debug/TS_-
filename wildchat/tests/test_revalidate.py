"""Synthetic checks for issue 13; only train/validation fixtures are used."""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TRAINING = HAS_NUMPY and all(importlib.util.find_spec(n) is not None for n in ("lightgbm", "pyarrow"))
if HAS_NUMPY:
    import numpy as np
    from wildchat import revalidate as rv


def rows(y, users, groups=None, turns=None, stamps=None, previous=None, cumulative=None):
    n = len(y)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    return rv.Rows(np.zeros((n, 31)), np.asarray(y, dtype=float), groups, groups.copy(),
                   np.zeros(n, dtype=int) if turns is None else np.asarray(turns),
                   np.arange(n) if stamps is None else np.asarray(stamps), np.asarray(users, dtype=object),
                   np.full(n, np.nan) if previous is None else np.asarray(previous, dtype=float),
                   np.full(n, np.nan) if cumulative is None else np.asarray(cumulative, dtype=float))


@unittest.skipUnless(HAS_NUMPY, "requires the existing NumPy experiment environment")
class SplitAndMetricTests(unittest.TestCase):
    def test_purge_removes_both_sides_missing_ids_and_whole_groups(self):
        earlier = rows([1] * 5, ["shared", "peer", "", "safe", "safe"],
                       [1, 1, 2, 3, 3], stamps=[1, 1, 2, 3, 3])
        later = rows([2] * 4, ["shared", "new", "peer2", "new2"],
                     [4, 5, 5, 6], stamps=[10, 11, 11, 12])
        left, right, audit = rv.purge_and_check(earlier, later, 10)
        self.assertEqual(left.users.tolist(), ["safe", "safe"])
        self.assertEqual(right.users.tolist(), ["new", "peer2", "new2"])
        self.assertEqual(audit["checks"]["user_intersection"], 0)
        self.assertEqual(audit["removed_rows"], {"earlier": 3, "later": 1})

    def test_inner_boundary_ties_and_spanning_groups_are_purged(self):
        data = rows([1] * 7, list("aabccde"), [0, 0, 1, 2, 2, 3, 4],
                    stamps=[1, 9, 2, 3, 3, 4, 5])
        left, right, audit = rv.inner_split(data, .6)
        self.assertEqual(audit["cutoff"], rv.iso_time(4))
        self.assertEqual(left.groups.tolist(), [1, 2, 2])
        self.assertEqual(right.groups.tolist(), [3, 4])
        self.assertLess(left.stamps.max(), right.stamps.min())

    def test_crossings_are_scored_with_original_quantile_identity(self):
        pred = np.array([[9., 3., 1.]])
        result = dict(zip(rv.METRICS, rv.contributions(np.array([4.]), pred)[0]))
        self.assertAlmostEqual(result["pinball_p10"], 4.5)
        self.assertAlmostEqual(result["pinball_p50"], .5)
        self.assertAlmostEqual(result["pinball_p90"], 2.7)
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["mean_interval_width"], -8)
        self.assertEqual(result["crossing_rate"], 1)
        np.testing.assert_array_equal(pred, [[9., 3., 1.]])

    def test_ratio_is_pinball_optimal_weighted_quantile_and_fallback(self):
        data = rows([10., 9., 20.], ["a", "b", "c"], previous=[1, 9, np.nan],
                    cumulative=[5, 9, np.nan])
        fitted = rv.fit_baselines(data)
        self.assertEqual(fitted["previous_ratio"], [1., 1., 1.])
        predicted = rv.predict_baselines(data, fitted)
        np.testing.assert_array_equal(predicted["previous_ratio"][2], fitted["fixed"])
        np.testing.assert_array_equal(predicted["cumulative_mean"][:2, 1], [5, 9])
        np.testing.assert_array_equal(predicted["cumulative_mean"][2], fitted["fixed"])

    def test_bootstrap_matches_explicit_user_resampling_for_all_metrics(self):
        data = rows([0, 5, 9, 10, 30], ["a", "a", "b", "c", "c"], turns=[0, 2, 0, 0, 1])
        a = np.array([[1, 2, 4], [3, 6, 8], [4, 8, 12], [5, 9, 13], [10, 20, 35]], dtype=float)
        predictions = {name: a + index for index, name in enumerate(rv.MODELS)}
        # Include a raw crossing in the reference, too.
        predictions["B"][0] = [12, 8, 3]
        best = {c: {w: "previous_ratio" for w in ("row", "user")} for c in rv.cohorts(data)}
        seed, repeats = 17, 64
        info, actual = rv.bootstrap(data, predictions, best, seed, repeats)
        self.assertEqual(info["groups_per_draw"], 3)
        self.assertFalse(info["refit_or_reselect"])
        rng = np.random.Generator(np.random.PCG64(seed))
        expected = {c: {w: [] for w in ("row", "user")} for c in best}
        for _ in range(repeats):
            chosen = rng.integers(0, 3, size=3)
            indices = np.concatenate([np.flatnonzero(data.users == ["a", "b", "c"][i]) for i in chosen])
            # Duplicated users are distinct sampled clusters with equal multiplicity.
            draws = [np.flatnonzero(data.users == ["a", "b", "c"][i]) for i in chosen]
            for cohort, mask in rv.cohorts(data).items():
                selected_rows = indices[mask[indices]]
                if not len(selected_rows):
                    continue
                for weight in ("row", "user"):
                    values = {}
                    for name, pred in predictions.items():
                        if weight == "row":
                            values[name] = rv.contributions(data.y[selected_rows], pred[selected_rows]).mean(axis=0)
                        else:
                            user_means = [rv.contributions(data.y[ix[mask[ix]]], pred[ix[mask[ix]]]).mean(axis=0)
                                          for ix in draws if mask[ix].any()]
                            values[name] = np.mean(user_means, axis=0)
                    expected[cohort][weight].append(values)
        for cohort, weighted in actual.items():
            for weight, entry in weighted.items():
                self.assertEqual(entry["effective_replicates"], len(expected[cohort][weight]))
                for label, comparator in (("B-A", "A"), ("B-best_baseline", "previous_ratio")):
                    deltas = [v["B"] - v[comparator] for v in expected[cohort][weight]]
                    bounds = np.quantile(deltas, [.025, .975], axis=0, method="linear")
                    for index, metric in enumerate(rv.METRICS):
                        np.testing.assert_allclose(entry["comparisons"][label]["ci95"][metric], bounds[:, index], atol=1e-12)
        metrics = rv.evaluate_all(data, predictions)
        self.assertNotEqual(metrics["all"]["row"]["B"]["mean_pinball_loss"],
                            metrics["all"]["user"]["B"]["mean_pinball_loss"])
        self.assertEqual((info, actual), rv.bootstrap(data, predictions, best, seed, repeats))

    def test_empty_cohort_reports_null_and_counts(self):
        data = rows([1, 2], ["a", "b"])
        predictions = {name: np.tile([0., 1., 3.], (2, 1)) for name in rv.MODELS}
        best = {c: {w: "fixed" for w in ("row", "user")} for c in rv.cohorts(data)}
        _, result = rv.bootstrap(data, predictions, best, 42, 10)
        self.assertIsNone(result["followup_turn"]["row"]["comparisons"]["B-A"]["ci95"])
        self.assertEqual(result["followup_turn"]["row"]["empty_replicates"], 10)


@unittest.skipUnless(HAS_TRAINING, "requires the existing LightGBM/PyArrow environment")
class IntegrationTests(unittest.TestCase):
    def test_only_allowed_files_causal_means_and_train_only_fit(self):
        import lightgbm as lgb
        import pyarrow as pa
        import pyarrow.parquet as pq
        base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix=".issue13-check-", dir=root) as directory:
            prepared = Path(directory)
            source = prepared / "synthetic.parquet"
            partitions, source_users, source_stamps = {}, [], []
            for split, ids in (("train", range(40)), ("validation", range(40, 50))):
                samples = []
                for group in ids:
                    stamp = base + timedelta(days=group)
                    source_users.append(f"user-{group}")
                    source_stamps.append(stamp)
                    for turn in range(6):
                        history = {"previous_user_turns": turn}
                        for lag in range(1, 4):
                            for side in ("input", "output"):
                                for unit in ("tokens", "chars", "utf8_bytes"):
                                    history[f"lag_{lag}_{side}_{unit}"] = None if turn < lag else 10 + turn - lag
                                history[f"lag_{lag}_{side}_missing"] = turn < lag
                            history[f"lag_{lag}_turn_missing"] = turn < lag
                        samples.append({"metadata": {"source_row_id": group, "group_id": group, "split": split,
                                        "turn_index": turn, "user_message_index": turn * 2, "recorded_at": stamp.isoformat()},
                                        "prompt_features": dict(zip(rv.feature_names(3)["A"], [10, 30, 40])),
                                        "history_features": history, "target": {"output_tokens": 10 + turn}})
                (prepared / f"{split}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
                partitions[split] = {"samples": len(samples), "groups": len(ids), "conversations": len(ids),
                                     "recorded_at_min": samples[0]["metadata"]["recorded_at"],
                                     "recorded_at_max": samples[-1]["metadata"]["recorded_at"]}
            manifest = {"partitions": partitions, "lags": 3,
                        "input_files": [{"name": source.name, "rows": 50, "sha256": "synthetic"}],
                        "boundaries": {"train_validation_cutoff": (base + timedelta(days=40)).isoformat()}}
            (prepared / "manifest.json").write_text(json.dumps(manifest))
            pq.write_table(pa.table({"hashed_ip": source_users, "timestamp": source_stamps}), source)
            original_open = Path.open
            opened = []
            def guarded_open(path, *args, **kwargs):
                if path.suffix == ".jsonl":
                    self.assertIn(path.name, ("train.jsonl", "validation.jsonl"))
                    opened.append(path.name)
                return original_open(path, *args, **kwargs)
            args = argparse.Namespace(prepared=prepared, source=source, output=prepared / "results",
                                      seed=42, inner_fraction=.8, max_rounds=4,
                                      early_stopping_rounds=2, threads=1, bootstrap_replicates=16)
            fit_calls = []
            real_train = lgb.train
            def checked_train(params, dataset, *pos, **kw):
                self.assertLess(np.max(dataset.get_label()), 100)
                fit_calls.append("valid_sets" in kw)
                if "valid_sets" in kw:
                    self.assertEqual(kw["valid_names"], ["inner_holdout"])
                    for valid in kw["valid_sets"]:
                        self.assertLess(np.max(valid.get_label()), 100)
                return real_train(params, dataset, *pos, **kw)
            # Sentinel val labels cannot appear in any fitting/early-stopping dataset.
            path = prepared / "validation.jsonl"
            content = [json.loads(line) for line in path.read_text().splitlines()]
            for sample in content:
                sample["target"]["output_tokens"] += 1000
            path.write_text("".join(json.dumps(s) + "\n" for s in content))
            with mock.patch.object(Path, "open", guarded_open), mock.patch.object(lgb, "train", checked_train):
                result = rv.run(args)
            self.assertEqual(opened, ["train.jsonl", "validation.jsonl"])
            self.assertEqual(fit_calls, [True] * 18 + [False] * 6)
            self.assertEqual(result["splits"]["outer"]["checks"]["user_intersection"], 0)
            self.assertEqual(len(result["validation_metrics"]), 3)
            loaded, _ = rv.load_allowed(prepared, manifest, rv.feature_names(3)["B"])
            np.testing.assert_allclose(loaded["train"].cumulative[:6], [np.nan, 10, 10.5, 11, 11.5, 12], equal_nan=True)
            # Reports are valid JSON even with tiny/empty cohorts.
            json.loads((args.output / "revalidation.json").read_text())


if __name__ == "__main__":
    unittest.main()
