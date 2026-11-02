"""Synthetic-only experiment checks; optional numerical integration in the venv."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from wildchat import train
from wildchat.metrics import baseline, evaluate

BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_LIGHTGBM = HAS_NUMPY and importlib.util.find_spec("lightgbm") is not None


def sample(group, turn=0, split="train"):
    def lengths(position):
        tokens = 3 + (group * 11 + position * 23) % 400
        output = 5 + (group * 7 + position * 13) % 300
        return tokens, output
    tokens, output = lengths(turn)
    history = {"previous_user_turns": turn}
    for lag in range(1, 4):
        missing = turn < lag
        previous_input, previous_output = lengths(turn - lag)
        history[f"lag_{lag}_turn_missing"] = missing
        for side, value in (("input", previous_input), ("output", previous_output)):
            history[f"lag_{lag}_{side}_missing"] = missing
            for unit, factor in (("tokens", 1), ("chars", 3), ("utf8_bytes", 4)):
                history[f"lag_{lag}_{side}_{unit}"] = None if missing else value * factor
        history[f"lag_{lag}_turn_interval_seconds"] = None
        history[f"lag_{lag}_turn_interval_missing"] = True
    return {"prompt_features": dict(zip(train.PROMPT_FEATURES, (tokens, tokens * 3, tokens * 4))),
            "history_features": history, "target": {"output_tokens": output},
            "metadata": {"group_id": group, "source_row_id": group, "split": split,
                         "recorded_at": (BASE + timedelta(days=group)).isoformat(),
                         "turn_index": turn, "user_message_index": turn * 2},
            "diagnostics": {"interval_exclusion_reasons": {str(i): "unverified" for i in range(1, 4)}}}


def prepared_fixture(path):
    path.mkdir()
    partitions = {}
    for split, groups in (("train", range(80)), ("validation", range(100, 120))):
        rows = [sample(group, turn, split) for group in groups for turn in range(5)]
        (path / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        partitions[split] = {"groups": len(groups), "conversations": len(groups), "samples": len(rows),
                             "recorded_at_min": rows[0]["metadata"]["recorded_at"],
                             "recorded_at_max": rows[-1]["metadata"]["recorded_at"]}
    partitions["test"] = {"groups": 10, "conversations": 10, "samples": 50,
                          "recorded_at_min": (BASE + timedelta(days=200)).isoformat(),
                          "recorded_at_max": (BASE + timedelta(days=209)).isoformat()}
    # A poisoned file plus an open() guard makes accidental test access fail.
    (path / "test.jsonl").write_text("FORBIDDEN: not JSON\n")
    manifest = {"lags": 3, "verified_user_timestamps": False, "partitions": partitions,
                "boundaries": {"train_validation_cutoff": (BASE + timedelta(days=100)).isoformat(),
                               "validation_test_cutoff": (BASE + timedelta(days=200)).isoformat()},
                "revision": "synthetic", "first_shard_only": True,
                "input_files": [{"name": "synthetic.parquet", "rows": 110, "sha256": "synthetic"}],
                "versions": {"python": sys.version.split()[0]}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class FeatureAndContractTests(unittest.TestCase):
    def test_allowlist_difference_and_first_turn_missing_not_dropped(self):
        names = train.feature_names(3)
        self.assertEqual(names["A"], ["input_tokens", "input_chars", "input_utf8_bytes"])
        self.assertEqual(len(names["B"]), 31)
        self.assertEqual(names["B"][:3], names["A"])
        for lag in range(1, 4):
            self.assertIn(f"lag_{lag}_input_tokens", names["B"])
            self.assertIn(f"lag_{lag}_output_chars", names["B"])
            self.assertNotIn(f"lag_{lag}_turn_interval_seconds", names["B"])
        first = train.feature_vector(sample(0), names["B"])
        self.assertEqual(first[:3], train.feature_vector(sample(0), names["A"]))
        self.assertTrue(math.isnan(first[names["B"].index("lag_1_output_tokens")]))
        self.assertEqual(first[names["B"].index("lag_1_turn_missing")], 1)
        self.assertEqual(first[names["B"].index("previous_user_turns")], 0)

    def test_target_metadata_and_extra_fields_cannot_enter_features(self):
        original = sample(2, 3)
        changed = copy.deepcopy(original)
        changed["target"]["output_tokens"] = 999999
        changed["metadata"] = {"future": "must not be a feature"}
        changed["prompt_features"]["future_output"] = 999999
        changed["prompt_features"]["language"] = "ko"
        changed["prompt_features"]["prompt_type"] = "semantic feature must be ignored"
        changed["history_features"]["whole_conversation_total"] = 999999
        for names in train.feature_names(3).values():
            self.assertEqual(train.feature_vector(original, names), train.feature_vector(changed, names))
        changed = copy.deepcopy(original)
        changed["history_features"]["lag_1_output_tokens"] += 1
        names = train.feature_names(3)
        self.assertEqual(train.feature_vector(original, names["A"]), train.feature_vector(changed, names["A"]))
        self.assertNotEqual(train.feature_vector(original, names["B"]), train.feature_vector(changed, names["B"]))
        for bad in ("raw text", -1, math.inf, True):
            changed = sample(2)
            changed["prompt_features"]["input_tokens"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                train.feature_vector(changed, names["A"])

    def test_group_source_row_and_chronological_intersections_rejected(self):
        a = train.Partition(None, None, {"recorded_at_max": BASE.isoformat()}, {1}, {10})
        b = train.Partition(None, None, {"recorded_at_min": (BASE + timedelta(days=1)).isoformat()}, {2}, {20})
        train.check_disjoint(a, b)
        for groups, rows, stamp in (({1}, {20}, b.summary["recorded_at_min"]),
                                    ({2}, {10}, b.summary["recorded_at_min"]),
                                    ({2}, {20}, BASE.isoformat())):
            invalid = train.Partition(None, None, {"recorded_at_min": stamp}, groups, rows)
            with self.assertRaises(ValueError):
                train.check_disjoint(a, invalid)

    def test_metric_formulas_and_signed_differences(self):
        result = evaluate([0, 4], [[1, 2, 3], [1, 2, 3]])
        self.assertEqual(result["mae_p50"], 2)
        self.assertAlmostEqual(result["mean_absolute_log1p_error_p50"], (math.log(3) + math.log(5 / 3)) / 2)
        for key, expected in (("p10", .6), ("p50", 1), ("p90", .6)):
            self.assertAlmostEqual(result["pinball_loss"][key], expected)
        self.assertAlmostEqual(result["mean_pinball_loss"], 2.2 / 3)
        self.assertEqual(result["p10_p90_coverage"], 0)
        self.assertEqual(result["mean_interval_width"], 2)
        other = evaluate([1, 3], [[1, 2, 3]] * 2)
        self.assertEqual(other["p10_p90_coverage"], 1)
        delta = train.metric_difference(result, other)
        self.assertEqual(delta["mae_p50"], -1)
        self.assertEqual(delta["p10_p90_coverage"], 1)
        self.assertAlmostEqual(delta["pinball_loss"]["p50"], -.5)

    def test_cli_imports_without_inspect_shadowing(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, str(root / "wildchat/train.py"), "--help"],
                                capture_output=True, text=True, cwd=root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prepared", result.stdout)


@unittest.skipUnless(HAS_NUMPY, "numerical checks require the issue-authorized NumPy environment")
class NumericalTests(unittest.TestCase):
    def test_paired_group_bootstrap_matches_explicit_group_row_resampling(self):
        import numpy as np
        targets = np.array([0, 5, 9, 10, 30])
        groups = np.array([10, 10, 20, 30, 30])  # deliberately unequal cluster sizes
        turns = np.array([0, 2, 0, 0, 1])
        a = np.array([[1, 2, 4], [3, 6, 8], [4, 8, 12], [5, 9, 13], [10, 20, 35]])
        predictions = {"baseline": np.tile([1, 7, 20], (5, 1)), "A": a, "B": a + 1}
        seed, repeats = 17, 128
        metadata, result = train.cluster_bootstrap(targets, predictions, groups, turns, seed, repeats)
        self.assertEqual(metadata["groups_per_draw"], 3)
        self.assertEqual(metadata["replicates"], repeats)
        self.assertFalse(metadata["refit_or_reselect"])
        rng = np.random.Generator(np.random.PCG64(seed))
        explicit = {name: [] for name in ("all", "first_turn", "followup_turn")}
        for _ in range(repeats):
            # Independent reference: materialize every selected group's full rows,
            # including repeated groups; call #6 evaluate instead of aggregated sums.
            draw = rng.integers(0, 3, size=3)
            indices = np.concatenate([np.flatnonzero(groups == [10, 20, 30][g]) for g in draw])
            subsets = {"all": indices, "first_turn": indices[turns[indices] == 0],
                       "followup_turn": indices[turns[indices] > 0]}
            for cohort, rows in subsets.items():
                if not len(rows):
                    continue
                metrics = {name: evaluate(targets[rows].tolist(), pred[rows].tolist())
                           for name, pred in predictions.items()}
                metrics["B-A"] = train.metric_difference(metrics["A"], metrics["B"])
                explicit[cohort].append(metrics)
        for cohort, mask in train.cohort_masks(turns).items():
            actual = result[cohort]
            self.assertEqual(actual["n_samples"], int(mask.sum()))
            self.assertEqual(actual["n_groups"], len(set(groups[mask])))
            self.assertEqual(actual["bootstrap_effective_replicates"], len(explicit[cohort]))
            self.assertEqual(actual["bootstrap_empty_replicates"], repeats - len(explicit[cohort]))
            for name, pred in predictions.items():
                self.assertEqual(actual["models"][name]["metrics"], evaluate(targets[mask].tolist(), pred[mask].tolist()))
            for name in (*train.COMPARISONS, "B-A"):
                entry = actual["difference_b_minus_a"] if name == "B-A" else actual["models"][name]
                for metric in train.BOOTSTRAP_METRICS:
                    expected = np.quantile([m[name][metric] for m in explicit[cohort]], [.025, .975], method="linear")
                    np.testing.assert_allclose(entry["ci95"][metric], expected, atol=1e-12)
        self.assertGreater(result["followup_turn"]["bootstrap_empty_replicates"], 0)
        self.assertEqual((metadata, result), train.cluster_bootstrap(targets, predictions, groups, turns, seed, repeats))

    def test_bootstrap_degenerate_paired_difference_and_empty_cohort(self):
        import numpy as np
        y = np.array([0, 1, 2])
        a = np.tile([3, 4, 5], (3, 1))
        predictions = {"baseline": a, "A": a, "B": a + 1}
        _, result = train.cluster_bootstrap(y, predictions, [7, 7, 7], [0, 1, 2], 42, 25)
        for cohort in result.values():
            np.testing.assert_allclose(cohort["difference_b_minus_a"]["ci95"]["mae_p50"], [1, 1])
            np.testing.assert_allclose(cohort["difference_b_minus_a"]["ci95"]["mean_pinball_loss"], [.5, .5])
        _, no_later = train.cluster_bootstrap(y, predictions, [7, 8, 9], [0, 0, 0], 42, 25)
        empty = no_later["followup_turn"]
        self.assertEqual(empty["n_samples"], 0)
        self.assertEqual(empty["bootstrap_effective_replicates"], 0)
        self.assertEqual(empty["bootstrap_empty_replicates"], 25)
        self.assertIsNone(empty["difference_b_minus_a"]["metrics"])
        for entry in empty["models"].values():
            self.assertIsNone(entry["metrics"])
            self.assertTrue(all(value is None for value in entry["ci95"].values()))
        for seed, repeats in ((42, 0), (-1, 10)):
            with self.assertRaises(ValueError):
                train.cluster_bootstrap(y, predictions, [7, 7, 7], [0, 1, 2], seed, repeats)
        with self.assertRaises(ValueError):
            train.cluster_bootstrap(y, {"A": a, "B": a}, [7, 7, 7], [0, 1, 2], 42, 10)

    def test_crossing_repair_clipping_and_metrics(self):
        import numpy as np
        raw = [[5, 1, 3], [-2, 0, 4], [1, 1, 2]]
        ordered, diagnostics = train.order_quantiles(raw)
        np.testing.assert_equal(ordered, [[1, 3, 5], [0, 0, 4], [1, 1, 2]])
        self.assertEqual(diagnostics["raw_crossing_rows"], 1)
        self.assertAlmostEqual(diagnostics["raw_crossing_rate"], 1 / 3)
        self.assertEqual(diagnostics["negative_prediction_rows"], 1)
        self.assertEqual(evaluate([3, 0, 1], ordered.tolist())["p10_p90_coverage"], 1)
        with self.assertRaises(ValueError):
            evaluate([3], [raw[0]])
        for bad in ([], [[1, 2]], [[1, math.nan, 3]]):
            with self.assertRaises(ValueError):
                train.order_quantiles(bad)

    def test_distributions_correlations_and_average_ties(self):
        import numpy as np
        np.testing.assert_equal(train.average_ranks([20, 10, 20, 40]), [2.5, 1, 2.5, 4])
        inputs = np.array([0, 1, 3, 3], dtype=float)
        outputs = (inputs + 1) ** 2 - 1
        stats = train.descriptive_stats(inputs, outputs)
        self.assertEqual(stats["input_tokens"]["p50"], 1)
        self.assertEqual(stats["input_tokens"]["p90"], 3)
        self.assertAlmostEqual(stats["correlations"]["spearman_average_ties"], 1)
        self.assertAlmostEqual(stats["correlations"]["pearson_log1p"], 1)
        self.assertLess(stats["correlations"]["pearson_raw"], 1)
        self.assertIsNone(train.pearson([1, 1], [2, 3]))

    def test_interval_columns_depend_only_on_verified_training_observations(self):
        import numpy as np
        names = train.feature_names(3, (1, 2, 3))["B"]
        x = np.full((2, len(names)), math.nan)
        x[0, names.index("lag_1_turn_interval_seconds")] = 10
        part = train.Partition(x, None, {}, set(), set())
        self.assertEqual(train.select_interval_lags(part, names, 3, True), [1])
        self.assertEqual(train.select_interval_lags(part, names, 3, False), [])
        x[:] = math.nan
        self.assertEqual(train.select_interval_lags(part, names, 3, True), [])

    def test_loader_counts_boundaries_target_and_test_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prepared"
            manifest = prepared_fixture(path)
            names = train.feature_names(3)["B"]
            a = train.load_partition(path, "train", manifest, names)
            b = train.load_partition(path, "validation", manifest, names)
            train.check_disjoint(a, b)
            self.assertEqual(a.summary["first_turn_samples"], 80)
            self.assertEqual(b.summary["first_turn_samples"], 20)
            self.assertEqual(len(b.y), 100)
            self.assertEqual(list(b.row_groups[:5]), [100] * 5)
            self.assertEqual(list(b.turn_indices[:5]), [0, 1, 2, 3, 4])
            with self.assertRaisesRegex(ValueError, "test is forbidden"):
                train.load_partition(path, "test", manifest, names)
            invalid_manifest = copy.deepcopy(manifest)
            invalid_manifest["partitions"]["train"]["samples"] -= 1
            with self.assertRaisesRegex(ValueError, "sample count"):
                train.load_partition(path, "train", invalid_manifest, names)
            rows = [json.loads(line) for line in (path / "validation.jsonl").read_text().splitlines()]
            original = copy.deepcopy(rows)
            for mutate in (lambda row: row["metadata"].update(split="test"),
                           lambda row: row["metadata"].update(recorded_at=BASE.isoformat()),
                           lambda row: row["target"].update(output_tokens=-1)):
                rows = copy.deepcopy(original)
                mutate(rows[0])
                (path / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    train.load_partition(path, "validation", manifest, names)

    def test_interval_diagnostics_reject_unverified_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prepared"
            manifest = prepared_fixture(path)
            manifest["verified_user_timestamps"] = True
            rows = [json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()]
            rows[2]["history_features"]["lag_1_turn_interval_seconds"] = 10
            rows[2]["history_features"]["lag_1_turn_interval_missing"] = False
            names = train.feature_names(3, (1,))["B"]
            def write_rows():
                (path / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_rows()
            with self.assertRaisesRegex(ValueError, "invalid turn interval"):
                train.load_partition(path, "train", manifest, names)
            del rows[2]["diagnostics"]["interval_exclusion_reasons"]["1"]
            write_rows()
            part = train.load_partition(path, "train", manifest, names)
            self.assertEqual(train.select_interval_lags(part, names, 1, True), [1])


@unittest.skipUnless(HAS_LIGHTGBM, "LightGBM integration requires the issue-authorized venv")
class LightGBMIntegrationTests(unittest.TestCase):
    def test_full_run_fair_budget_train_only_fitting_and_no_test_access(self):
        import lightgbm as lgb
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prepared = tmp / "prepared"
            prepared_fixture(prepared)
            args = argparse.Namespace(prepared=prepared, output=tmp / "results", seed=42,
                                      max_rounds=6, early_stopping_rounds=2, threads=1, bootstrap_replicates=64)
            original_open = Path.open
            opened = []
            def guarded_open(path, *args, **kwargs):
                self.assertNotEqual(path.name, "test.jsonl", "test must never be opened")
                opened.append(path.name)
                return original_open(path, *args, **kwargs)
            with mock.patch.object(Path, "open", guarded_open), mock.patch.object(lgb, "train", wraps=lgb.train) as fit:
                with mock.patch("sys.stdout", new_callable=io.StringIO):
                    result = train.run(args)
            self.assertFalse(result["test"]["read"])
            self.assertFalse(result["test"]["evaluated"])
            self.assertNotIn("test.jsonl", opened)
            self.assertEqual(fit.call_count, 18)
            calls_a, calls_b = fit.call_args_list[:9], fit.call_args_list[9:]
            for ca, cb in zip(calls_a, calls_b):
                self.assertEqual(ca.args[0], cb.args[0])  # same alpha, seeds and search settings
                self.assertEqual(ca.kwargs["num_boost_round"], cb.kwargs["num_boost_round"])
                self.assertEqual(ca.args[1].num_data(), 400)
                self.assertEqual(ca.args[1].num_feature(), 3)
                self.assertEqual(cb.args[1].num_feature(), 31)
                np.testing.assert_equal(ca.args[1].get_label(), cb.args[1].get_label())
                valid_a, valid_b = ca.kwargs["valid_sets"][0], cb.kwargs["valid_sets"][0]
                self.assertIs(valid_a.reference, ca.args[1])
                self.assertIs(valid_b.reference, cb.args[1])
                self.assertEqual(valid_a.num_data(), 100)
                np.testing.assert_equal(valid_a.get_label(), valid_b.get_label())
            self.assertEqual(result["partitions"]["validation"]["first_turn_samples"], 20)
            self.assertEqual(result["comparison_models"], ["baseline", "A", "B"])
            self.assertEqual(result["bootstrap"]["replicates"], 64)
            self.assertEqual(result["bootstrap"]["seed"], 42)
            self.assertFalse(result["semantic_features"])
            for cohort, n in (("all", 100), ("first_turn", 20), ("followup_turn", 80)):
                grouped = result["validation_by_cohort"][cohort]
                self.assertEqual(grouped["n_samples"], n)
                self.assertEqual(grouped["n_groups"], 20)
                self.assertEqual(grouped["bootstrap_effective_replicates"], 64)
                self.assertEqual(set(grouped["models"]), {"baseline", "A", "B"})
                for entry in grouped["models"].values():
                    self.assertEqual(entry["metrics"]["n"], n)
                    self.assertEqual(set(entry["ci95"]), set(train.BOOTSTRAP_METRICS))
                    for bounds in entry["ci95"].values():
                        self.assertLessEqual(bounds[0], bounds[1])
            self.assertEqual(result["baseline"]["quantiles"], dict(zip(("p10", "p50", "p90"), baseline([
                sample(g, t)["target"]["output_tokens"] for g in range(80) for t in range(5)]))))
            for name, model in result["models"].items():
                self.assertEqual(model["validation_metrics"]["n"], 100)
                scores = [trial["validation_metrics"]["mean_pinball_loss"] for trial in model["trials"]]
                self.assertEqual(model["selected_trial"], scores.index(min(scores)))
            saved = json.loads((args.output / "experiment.json").read_text())
            self.assertEqual(saved, result)
            self.assertIn("test 미평가", (args.output / "report.md").read_text())
            self.assertIn("절대 수치는 낙관적일 수 있음(A·B 동일 조건)", (args.output / "report.md").read_text())
            self.assertEqual({p.name for p in args.output.iterdir()}, {"experiment.json", "report.md"})
            # Same seed and inputs reproduce every model/trial metric and selection.
            args.output = tmp / "repeat"
            with mock.patch.object(Path, "open", guarded_open), mock.patch("sys.stdout", new_callable=io.StringIO):
                repeated = train.run(args)
            self.assertEqual(result["models"], repeated["models"])
            self.assertEqual(result["validation_by_cohort"], repeated["validation_by_cohort"])


if __name__ == "__main__":
    unittest.main()
