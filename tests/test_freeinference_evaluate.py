"""Synthetic split, baseline leakage, metric arithmetic and paired-bootstrap tests."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "pandas", "pyarrow"))
if HAS_DEPS:
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from freeinference import evaluate, loader, split
    from freeinference.baselines import (MIN_GROUP_ROWS, Baselines, build_history,
                                        PREDICTION_COLUMNS)


@unittest.skipUnless(HAS_DEPS, "numpy, pandas and pyarrow are required")
class SplitTests(unittest.TestCase):
    def test_first_start_assigns_entire_public_session_and_masks_only_labels(self):
        v, h = split.VALIDATION_START_MS, split.HOLDOUT_START_MS
        rows = pd.DataFrame({"week": [10, 10, 10, 9, 9, 9, 11, 11],
                             "session_id": ["s"] * 8,
                             "start_ms": [v - 3, v + 1, v - 2, h - 3, h + 1, h - 2, v, v + 2],
                             "end_ms": [v - 1, v + 3, None, h - 1, h, None, v + 1, v + 3]})
        masks = split.split_masks(rows)
        self.assertEqual(masks.session_week.tolist(), [10, 10, 10, 9, 9, 9, 11, 11])
        self.assertEqual(masks.train.tolist(), [True] * 6 + [False] * 2)
        self.assertEqual(masks.train_target.tolist(), [True, False, False, True, True, False, False, False])
        self.assertEqual(masks.fit_target.tolist(), [False] * 3 + [True, False, False, False, False])
        self.assertEqual(masks.holdout.tolist(), [True] * 3 + [False] * 5)
        self.assertEqual(masks.validation.tolist(), [False] * 6 + [True] * 2)
        self.assertEqual(len(rows), len(masks))
        # Exact train boundary is excluded too; release week is identity, not assignment.
        rows.loc[0, "end_ms"] = v
        self.assertFalse(split.split_masks(rows).train_target.iloc[0])
        rows.loc[0:2, "start_ms"] = v
        self.assertTrue(split.split_masks(rows).validation.iloc[:3].all())

    def test_test_times_and_release_weeks_fail_closed(self):
        rows = pd.DataFrame({"week": [13], "session_id": ["s"],
                             "start_ms": [loader.TEST_START_MS], "end_ms": [loader.TEST_START_MS + 1]})
        with self.assertRaisesRegex(ValueError, "timestamp"):
            split.split_masks(rows)
        rows.loc[0, "start_ms"] -= 1
        rows.loc[0, "week"] = 14
        with self.assertRaisesRegex(ValueError, "1–13"):
            split.split_masks(rows)

    def test_cli_refuses_test_folders_and_timestamps_without_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "REVISION").write_text(loader.REVISION)
            (root / "release").mkdir()
            with self.assertRaisesRegex(ValueError, "1–13"):
                evaluate.run(root, root / "out")
            paths = []
            from datetime import timedelta
            for i in range(13):
                start = loader.FIRST_WEEK + timedelta(weeks=i)
                path = root / "release" / f"{start}_{start + timedelta(days=6)}"
                (path / "processed").mkdir(parents=True)
                paths.append(path)
            # Loader rejects the timestamp even before projected columns are read.
            pd.DataFrame({"start_ms": [loader.TEST_START_MS]}).to_parquet(
                paths[0] / "processed" / "requests.parquet")
            with self.assertRaisesRegex(ValueError, "timestamp"):
                evaluate.run(root, root / "out")
            (root / "release" / "2026-08-16_2026-08-22").mkdir()
            with self.assertRaisesRegex(ValueError, "Held-out"):
                evaluate.run(root, root / "out")
            self.assertFalse((root / "out").exists())

    def test_cli_generates_aggregate_json_and_markdown_from_all_13_fixture_weeks(self):
        from datetime import timedelta

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "REVISION").write_text(loader.REVISION)
            for i in range(13):
                start = loader.FIRST_WEEK + timedelta(weeks=i)
                path = root / "release" / f"{start}_{start + timedelta(days=6)}" / "processed"
                path.mkdir(parents=True)
                ms = split.FIRST_MS + i * split.WEEK_MS
                pd.DataFrame({"session_id": ["private-session"] * 2, "step": [0, 1],
                              "start_ms": [ms, ms + 100], "end_ms": [ms + 50, ms + 150],
                              "link_reason": [None, "tool_call_id"],
                              "completion_tokens": [10 + i, 20 + i], "model": ["m", "m"]
                              }).to_parquet(path / "requests.parquet")
                pd.DataFrame({"session_id": ["private-session"], "harness": ["h"],
                              "user_id": ["private-account"]}).to_parquet(path / "sessions.parquet")
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate.run(root, root / "output")
            output = root / "output" / "baselines.json"
            recorded = json.loads(output.read_text())
            self.assertEqual(recorded["metrics"], result["metrics"])
            self.assertEqual(recorded["bootstrap"], result["bootstrap"])
            self.assertEqual(recorded["sets"]["train_target"], {"rows": 20, "sessions": 10})
            self.assertEqual(recorded["sets"]["validation"], {"rows": 6, "sessions": 3})
            self.assertEqual(set(recorded["metrics"]["overall"]["metrics"]),
                             {"fixed", "model", "previous_completion", "session_mean"})
            self.assertEqual(set(recorded["source_sha256"]),
                             {"loader.py", "features.py", "split.py", "baselines.py", "evaluate.py"})
            for file in (output, root / "output" / "baselines.md"):
                text = file.read_text()
                self.assertNotIn("private-session", text)
                self.assertNotIn("private-account", text)


@unittest.skipUnless(HAS_DEPS, "numpy, pandas and pyarrow are required")
class BaselineTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame({"week": [1] * 5, "session_id": ["s"] * 5,
                             "step": [0, 1, 2, 3, 4], "start_ms": [0, 100, 200, 300, 400],
                             "end_ms": [50, 150, 250, 350, 450],
                             "link_reason": [None] + ["tool_call_id"] * 4,
                             "completion_tokens": [10, 20, 30, 40, 50], "model": ["m"] * 5})

    def test_history_matches_features_gate_lag_and_full_cumulative_mean(self):
        from freeinference.features import build_features, REQUEST_COLUMNS
        rows = self.rows()
        history = build_history(rows)
        self.assertTrue(pd.isna(history.iloc[0, 1]))
        self.assertEqual(history.session_completion_mean.iloc[4], 25)  # all four, not last three
        inputs = rows.reindex(columns=[*REQUEST_COLUMNS, "week"])
        sessions = pd.DataFrame({"week": [1], "session_id": ["s"], "harness": ["h"]})
        calls = pd.DataFrame(columns=["week", "session_id", "step", "category", "outcome", "result_tokens"])
        features = build_features(inputs, sessions, calls)["L"]
        np.testing.assert_allclose(history.completion_lag_1, features.completion_lag_1, equal_nan=True)
        shuffled = rows.iloc[[3, 0, 4, 1, 2]].copy()
        shuffled.index = [7, 7, 1, 2, 3]
        actual = build_history(shuffled)
        expected = history.iloc[[3, 0, 4, 1, 2]].copy()
        expected.index = shuffled.index
        assert_frame_equal(actual, expected)

    def test_unfinished_current_and_future_completion_cannot_leak(self):
        rows = self.rows()
        rows.loc[0, "end_ms"] = 350  # older unfinished request; predecessor step 1 is complete
        original = build_history(rows)
        self.assertEqual(original.session_completion_mean.iloc[2], 20)
        rows.loc[[0, 2, 3, 4], "completion_tokens"] = 999999
        changed = build_history(rows)
        assert_frame_equal(original.iloc[[2]], changed.iloc[[2]])
        rows.loc[0, "end_ms"] = 200  # equality must still be excluded
        self.assertEqual(build_history(rows).session_completion_mean.iloc[2], 20)
        rows.loc[1, "end_ms"] = 200  # equality of direct predecessor disables all history
        self.assertTrue(build_history(rows).iloc[2].isna().all())
        rows = self.rows()
        rows.loc[4, ["start_ms", "end_ms", "completion_tokens"]] = [75, 90, 999999]
        self.assertEqual(build_history(rows).session_completion_mean.iloc[2], 15)
        for reason in (None, pd.NA, "unknown"):
            rows.loc[2, "link_reason"] = reason
            self.assertTrue(build_history(rows).iloc[2].isna().all())

    def test_public_week_isolates_history(self):
        first, second = self.rows(), self.rows()
        second.week = 2
        second.completion_tokens *= 10
        result = build_history(pd.concat([first, second], ignore_index=True))
        self.assertEqual(result.session_completion_mean.iloc[2], 15)
        self.assertEqual(result.session_completion_mean.iloc[7], 150)

    def test_train_only_statistics_masked_boundaries_and_validation_invariance(self):
        rows = self.rows()
        v = split.VALIDATION_START_MS
        rows.week = [10, 10, 10, 11, 11]
        rows.session_id = ["train"] * 3 + ["val"] * 2
        rows.start_ms = [v - 100, v - 80, v - 60, v + 1, v + 30]
        rows.end_ms = [v - 90, v - 70, v, v + 20, v + 40]
        rows.step = [0, 1, 2, 0, 1]
        history = build_history(rows)
        mask = split.split_masks(rows).train_target
        first = Baselines().fit(rows, history, mask)
        np.testing.assert_allclose(first.fixed, [11, 15, 19])
        expected = first.predict(rows, history)[0]
        rows.loc[2:, "completion_tokens"] = [999999, 888888, 777777]
        # Cached prediction-time history is unchanged by modifying the target at predict.
        actual = Baselines().fit(rows, history, mask).predict(rows, history)[0]
        for name in expected:
            assert_frame_equal(expected[name], actual[name])
        # Rebuilding also leaves the first validation prediction invariant.
        rebuilt = Baselines().fit(rows, build_history(rows), mask).predict(rows, build_history(rows))[0]
        for name in expected:
            assert_frame_equal(expected[name].iloc[[3]], rebuilt[name].iloc[[3]])

    def test_ratios_fallback_min_group_threshold_zero_lag_and_crossing(self):
        rows = pd.DataFrame({"model": ["large"] * MIN_GROUP_ROWS + ["small", "val"],
                             "completion_tokens": [20] * MIN_GROUP_ROWS + [100, 999999]})
        history = pd.DataFrame({"completion_lag_1": [10] * len(rows),
                                "session_completion_mean": [5] * len(rows)})
        train = np.arange(len(rows)) < len(rows) - 1
        model = Baselines().fit(rows, history, train)
        np.testing.assert_allclose(model.ratios["completion_lag_1"], [2, 2, 2])
        np.testing.assert_allclose(model.ratios["session_completion_mean"], [4, 4, 4])
        new = pd.DataFrame({"model": ["large", "small", "never", "large", "large"]})
        h = pd.DataFrame({"completion_lag_1": [np.nan, np.nan, np.nan, 3, 0],
                          "session_completion_mean": [np.nan, np.nan, np.nan, 3, 0]})
        predictions, fallback = model.predict(new, h)
        self.assertEqual(fallback.model.tolist(), [False, True, True, False, False])
        self.assertEqual(fallback.previous_completion.tolist(), [True, True, True, False, False])
        np.testing.assert_allclose(predictions["model"].iloc[0], [20] * 3)
        np.testing.assert_allclose(predictions["previous_completion"].iloc[1], model.fixed)
        np.testing.assert_allclose(predictions["previous_completion"].iloc[2], model.fixed)
        np.testing.assert_allclose(predictions["previous_completion"].iloc[3], [6] * 3)
        np.testing.assert_allclose(predictions["session_mean"].iloc[3], [12] * 3)
        np.testing.assert_allclose(predictions["previous_completion"].iloc[4], [0] * 3)
        for prediction in predictions.values():
            self.assertTrue((np.diff(prediction.to_numpy(), axis=1) >= 0).all())
        history = history.astype(float)
        history[:] = np.nan
        model = Baselines().fit(rows, history, train)
        self.assertTrue(model.predict(new, h)[1].session_mean.all())
        # The max(lag,1) denominator, including zero history, is part of training.
        history[:] = 0
        model = Baselines().fit(rows, history, train)
        np.testing.assert_allclose(model.ratios["completion_lag_1"], [20] * 3)


@unittest.skipUnless(HAS_DEPS, "numpy, pandas and pyarrow are required")
class EvaluationTests(unittest.TestCase):
    def fixture(self):
        rows = pd.DataFrame({"week": [11] * 3, "session_id": ["a", "a", "b"],
                             "step": [0, 1, 0], "harness": ["h1", "h1", "h2"],
                             "model": ["m1", "m1", "m2"], "completion_tokens": [0, 2, 4]})
        prediction = pd.DataFrame([[0, 1, 2]] * 3, columns=PREDICTION_COLUMNS)
        return rows, {"fixed": prediction, "same": prediction.copy(),
                      "other": prediction + 1}

    def test_hand_computed_metrics_both_weights_and_all_groups(self):
        rows, predictions = self.fixture()
        result = evaluate.evaluate_predictions(rows, predictions)
        # Row losses: (.0,.5,.2), (.2,.5,.0), (.4,1.5,1.8).
        expected_request = [.2, 5 / 6, 2 / 3, 17 / 30, 5 / 3, 2 / 3, 2]
        # Session a averages first two rows; session b contributes one unit.
        expected_session = [.25, 1, .95, 11 / 15, 2, .5, 2]
        for weighting, expected in (("request", expected_request), ("session", expected_session)):
            np.testing.assert_allclose(list(result["overall"]["metrics"]["fixed"][weighting].values()), expected)
        self.assertEqual((result["overall"]["rows"], result["overall"]["sessions"]), (3, 2))
        np.testing.assert_allclose(list(result["step/0"]["metrics"]["fixed"]["request"].values()),
                                   [.2, 1, 1, 11 / 15, 2, .5, 2])
        np.testing.assert_allclose(list(result["step/followup"]["metrics"]["fixed"]["session"].values()),
                                   [.2, .5, 0, 7 / 30, 1, 1, 2])
        for group in ("harness/value=h1", "model/value=m1"):
            self.assertEqual((result[group]["rows"], result[group]["sessions"]), (2, 1))
            np.testing.assert_allclose(list(result[group]["metrics"]["fixed"]["session"].values()),
                                       [.1, .5, .1, 7 / 30, 1, 1, 2])
        self.assertEqual(result["harness/value=h2"]["rows"], 1)
        self.assertEqual(result["model/value=m2"]["sessions"], 1)

    def test_bootstrap_reproducible_shared_samples_and_cluster_rows_move_together(self):
        rows, predictions = self.fixture()
        first = evaluate.paired_bootstrap(rows, predictions, B=80, seed=12)
        self.assertEqual(first, evaluate.paired_bootstrap(rows, predictions, B=80, seed=12))
        draws = np.concatenate(list(evaluate.cluster_draws(2, B=80, seed=12)))
        np.testing.assert_equal(draws.sum(axis=1), 2)
        # Independent brute-force row replication uses the SAME draws for every predictor.
        codes = np.array([0, 0, 1])
        row_multiplicities = draws[:, codes]
        np.testing.assert_equal(row_multiplicities[:, 0], row_multiplicities[:, 1])
        self.assertTrue((row_multiplicities[:, 0] == 0).any())
        expected = {name: {w: [] for w in evaluate.WEIGHTINGS} for name in predictions}
        for counts in draws:
            indexes = []
            for cluster, count in enumerate(counts):
                for _ in range(count):
                    indexes.extend(np.flatnonzero(codes == cluster))
            sampled = rows.iloc[indexes].reset_index(drop=True)
            # Each duplicate cluster copy needs a separate session identity.
            copy_ids = []
            for cluster, count in enumerate(counts):
                for repeat in range(count):
                    copy_ids.extend([f"{cluster}-{repeat}"] * (codes == cluster).sum())
            sampled.session_id = copy_ids
            sampled_predictions = {name: value.iloc[indexes].reset_index(drop=True)
                                   for name, value in predictions.items()}
            scores = evaluate.evaluate_predictions(sampled, sampled_predictions)["overall"]["metrics"]
            for name in predictions:
                for weighting in evaluate.WEIGHTINGS:
                    expected[name][weighting].append([scores[name][weighting][k] - scores["fixed"][weighting][k]
                                                       for k in evaluate.METRICS])
        for name in predictions:
            for weighting in evaluate.WEIGHTINGS:
                ci = first["groups"]["overall"]["differences"][name][weighting]
                manual = np.percentile(expected[name][weighting], [2.5, 97.5], axis=0)
                np.testing.assert_allclose(np.array(list(ci.values())).T, manual, atol=1e-14)
        for group in first["groups"].values():
            for name in ("fixed", "same"):
                for weighting in group["differences"][name].values():
                    self.assertTrue(all(value == [0., 0.] for value in weighting.values()))
        # A different clustering key supports future account/day blocks.
        rows["account"] = ["one"] * 3
        account = evaluate.paired_bootstrap(rows, predictions, cluster_columns=["account"], B=10)
        self.assertEqual(account["clusters"], 1)

    def test_invalid_metrics_and_bootstrap_inputs_are_rejected(self):
        rows, predictions = self.fixture()
        predictions["fixed"].iloc[0, 0] = 20
        with self.assertRaisesRegex(ValueError, "crossing"):
            evaluate.evaluate_predictions(rows, predictions)
        rows, predictions = self.fixture()
        predictions["fixed"].index = [3, 4, 5]
        with self.assertRaisesRegex(ValueError, "align"):
            evaluate.evaluate_predictions(rows, predictions)
        rows, predictions = self.fixture()
        with self.assertRaisesRegex(ValueError, "Reference"):
            evaluate.paired_bootstrap(rows, predictions, reference="absent", B=2)

    def test_bootstrap_all_predictors_and_groups_match_shared_row_resampling(self):
        sizes = np.array([1, 2, 3, 4, 5])
        codes = np.repeat(np.arange(5), sizes)
        steps = np.concatenate([np.arange(size) for size in sizes])
        rows = pd.DataFrame({"week": 11, "session_id": codes.astype(str), "step": steps,
                             "harness": np.where(codes < 2, "h1", "h2"),
                             "model": "m", "completion_tokens": np.arange(15) ** 2})
        fixed = pd.DataFrame(np.tile([0, 10, 100], (15, 1)), columns=PREDICTION_COLUMNS)
        predictions = {"fixed": fixed, "one": fixed + codes[:, None] * 3,
                      "two": fixed * 1.5 + steps[:, None]}
        actual = evaluate.paired_bootstrap(rows, predictions, B=31, seed=42)
        expected = {group: {name: {weight: [] for weight in evaluate.WEIGHTINGS}
                            for name in predictions} for group in actual["groups"]}
        for draw in np.concatenate(list(evaluate.cluster_draws(5, B=31, seed=42))):
            positions, session_copies = [], []
            for cluster, multiplicity in enumerate(draw):
                for copy in range(multiplicity):
                    selected = np.flatnonzero(codes == cluster)
                    positions.extend(selected)
                    session_copies.extend([f"{cluster}:{copy}"] * len(selected))
            sample = rows.iloc[positions].reset_index(drop=True)
            sample.session_id = session_copies
            prediction_sample = {name: value.iloc[positions].reset_index(drop=True)
                                 for name, value in predictions.items()}
            scores = evaluate.evaluate_predictions(sample, prediction_sample)
            for group, values in scores.items():
                for name in predictions:
                    for weight in evaluate.WEIGHTINGS:
                        expected[group][name][weight].append([
                            values["metrics"][name][weight][key] - values["metrics"]["fixed"][weight][key]
                            for key in evaluate.METRICS])
        for group, values in actual["groups"].items():
            for name in predictions:
                for weight in evaluate.WEIGHTINGS:
                    replicates = expected[group][name][weight]
                    self.assertEqual(values["valid_replicates"], len(replicates))
                    manual_ci = np.percentile(replicates, [2.5, 97.5], axis=0)
                    reported_ci = np.array(list(values["differences"][name][weight].values())).T
                    np.testing.assert_allclose(reported_ci, manual_ci, atol=1e-12)

    def test_missing_groups_stay_separate_and_empty_bootstrap_groups_are_explicit(self):
        rows, predictions = self.fixture()
        rows["harness"] = pd.Series([None, "<missing>", "other"], dtype="string")
        rows["model"] = pd.Series([None, "m", "m"], dtype="string")
        scores = evaluate.evaluate_predictions(rows, predictions)
        self.assertEqual(scores["harness/missing"]["rows"], 1)
        self.assertEqual(scores["harness/value=<missing>"]["rows"], 1)
        self.assertEqual(scores["model/missing"]["rows"], 1)
        result = evaluate.paired_bootstrap(rows, predictions, B=1, seed=1)
        for group in result["groups"].values():
            if group["valid_replicates"] == 0:
                self.assertEqual(group["differences"]["fixed"]["request"]["mean_pinball"], [None, None])


if __name__ == "__main__":
    unittest.main()
