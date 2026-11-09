"""Synthetic model contracts; no trace access or real-data evaluation."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "pandas", "pyarrow", "lightgbm"))
if HAS_DEPS:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from freeinference import loader, models, split
    from freeinference.features import FEATURE_SETS, REQUEST_COLUMNS, build_features


@unittest.skipUnless(HAS_DEPS, "numpy, pandas, pyarrow and LightGBM are required")
class ModelTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(models, "GRID", tuple(
            {"num_leaves": 5, "min_data_in_leaf": 2, "learning_rate": .1, "target_transform": t}
            for t in ("identity", "log1p"))))
        self.stack.enter_context(patch.object(models, "MAX_ROUNDS", 20))
        self.stack.enter_context(patch.object(models, "EARLY_STOPPING_ROUNDS", 4))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def fixture(self):
        records = []
        for week in range(1, 14):
            for session in range(5):
                for step in range(4):
                    ms = split.FIRST_MS + (week - 1) * split.WEEK_MS + session * 1000 + step * 100
                    records.append({"week": week, "session_id": f"private-session-{session}",
                                    "step": step, "start_ms": ms, "end_ms": ms + 50,
                                    "completion_tokens": (session + 1) * 12 + step * 8 + week,
                                    "prompt_tokens": (session + 1) * 20 + step * 4,
                                    "link_reason": "tool_call_id" if step else None,
                                    "model": "m" if session < 3 else "n", "provider": "p",
                                    "stream": True, "step_trigger": "user" if not step else "tool",
                                    "finish_reason": "stop", "role_tokens": {"user": 5},
                                    "tool_definition_names": []})
        rows = pd.DataFrame(records).reindex(columns=["week", *REQUEST_COLUMNS])
        sessions = rows[["week", "session_id"]].drop_duplicates().assign(harness="h")
        calls = pd.DataFrame(columns=["week", "session_id", "step", "category", "outcome", "result_tokens"])
        features = build_features(rows, sessions, calls)
        return rows, features

    def test_validation_targets_cannot_affect_selection_or_predictions_and_seed_is_fixed(self):
        rows, features = self.fixture()
        first = models.QuantileModels().fit(rows, features)
        expected, _ = first.predict(features)
        changed = rows.copy()
        changed.loc[split.split_masks(rows).validation, "completion_tokens"] = 999999
        # Prediction-time history is already observed: changing evaluation labels
        # must not retroactively change it (previous validation outcomes ARE valid history).
        second = models.QuantileModels().fit(changed, features)
        actual, crossing = second.predict(features)
        self.assertEqual(first.selection, second.selection)
        for condition in FEATURE_SETS:
            assert_frame_equal(expected[condition], actual[condition])
            self.assertTrue((np.diff(actual[condition].to_numpy(), axis=1) >= 0).all())
            self.assertTrue(0 <= crossing[condition] <= 1)
        self.assertEqual(models.PARAMETERS["seed"], models.SEED)
        self.assertTrue(models.PARAMETERS["deterministic"])
        self.assertEqual(models.PARAMETERS["num_threads"], models.NUM_THREADS)

    def test_training_and_selection_receive_exact_target_masks(self):
        rows, features = self.fixture()
        # A fit-boundary label can enter final train, but never selection.
        rows.loc[0, "end_ms"] = split.HOLDOUT_START_MS
        # Train boundary and missing end cannot enter either stage.
        rows.loc[1, "end_ms"] = split.VALIDATION_START_MS
        rows.loc[2, "end_ms"] = np.nan
        rows.loc[180, "end_ms"] = split.VALIDATION_START_MS
        masks = split.split_masks(rows)
        for frame in features.values():
            frame.loc[0, "model"] = "fit-boundary-only"
            frame.loc[1, "model"] = "excluded-only"
            frame.loc[181, "model"] = "holdout-only"
            frame.loc[masks.validation, "model"] = "validation-only"
        original_train = lgb.train
        calls = []

        def train(params, dataset, **kwargs):
            condition = tuple(FEATURE_SETS)[len(calls) // (3 * (len(models.GRID) + 1))]
            self.assertEqual(tuple(dataset.data.columns), FEATURE_SETS[condition])
            vocabulary = set(dataset.data.model.cat.categories)
            self.assertNotIn("validation-only", vocabulary)
            self.assertNotIn("excluded-only", vocabulary)
            if kwargs.get("valid_sets"):
                self.assertNotIn("holdout-only", vocabulary)
                self.assertNotIn("fit-boundary-only", vocabulary)
                self.assertTrue(pd.isna(kwargs["valid_sets"][0].data.loc[181, "model"]))
            else:
                self.assertIn("holdout-only", vocabulary)
                self.assertIn("fit-boundary-only", vocabulary)
            calls.append((params.copy(), dataset.get_label().copy(),
                          [valid.get_label().copy() for valid in kwargs.get("valid_sets", [])]))
            return original_train(params, dataset, **kwargs)

        with patch.object(lgb, "train", side_effect=train):
            first = models.QuantileModels().fit(rows, features)
        self.assertEqual(len(calls), 3 * 3 * (len(models.GRID) + 1))
        for pos, (_, target, holdouts) in enumerate(calls):
            candidate = pos % (len(models.GRID) + 1)
            if candidate < len(models.GRID):
                transform = models.GRID[candidate]["target_transform"]
                expected = rows.loc[masks.fit_target, "completion_tokens"].to_numpy(float)
                expected_holdout = rows.loc[masks.holdout_target, "completion_tokens"].to_numpy(float)
                if transform == "log1p":
                    expected, expected_holdout = np.log1p(expected), np.log1p(expected_holdout)
                np.testing.assert_array_equal(target, expected)
                np.testing.assert_array_equal(holdouts[0], expected_holdout)
            else:
                condition = tuple(FEATURE_SETS)[pos // (3 * (len(models.GRID) + 1))]
                label = models.PREDICTION_COLUMNS[(pos // (len(models.GRID) + 1)) % 3]
                selected = first.selection[condition][label]
                expected = rows.loc[masks.train_target, "completion_tokens"].to_numpy(float)
                if selected["settings"]["target_transform"] == "log1p":
                    expected = np.log1p(expected)
                np.testing.assert_array_equal(target, expected)
                self.assertEqual(holdouts, [])
        changed = rows.copy()
        changed.loc[[1, 2, 180], "completion_tokens"] = 999999
        second = models.QuantileModels().fit(changed, features)
        self.assertEqual(first.selection, second.selection)
        for condition, prediction in first.predict(features)[0].items():
            assert_frame_equal(prediction, second.predict(features)[0][condition])
        changed.loc[0, "completion_tokens"] = 888888
        third = models.QuantileModels().fit(changed, features)
        self.assertEqual(first.selection, third.selection)

    def test_unseen_and_missing_categories_are_na_and_predict(self):
        rows, features = self.fixture()
        model = models.QuantileModels().fit(rows, features)
        new = {name: frame.iloc[:3].copy() for name, frame in features.items()}
        for condition, frame in new.items():
            for column in set(models.CATEGORICAL) & set(frame):
                frame[column] = pd.Series(["never-seen", None, "m"], index=frame.index)
            encoded = model.encoders[condition].transform(frame)
            for column in set(models.CATEGORICAL) & set(frame):
                self.assertTrue(encoded[column].iloc[:2].isna().all())
                self.assertEqual(encoded[column].cat.codes.iloc[:2].tolist(), [-1, -1])
            self.assertEqual(tuple(encoded.columns), FEATURE_SETS[condition])
            self.assertNotIn("never-seen", model.encoders[condition].categories["model"])
        predictions, _ = model.predict(new)
        self.assertTrue(all(np.isfinite(frame.to_numpy()).all() for frame in predictions.values()))

    def test_input_columns_and_order_must_exactly_match_feature_sets(self):
        rows, features = self.fixture()
        self.assertEqual(tuple(features["L"].columns), FEATURE_SETS["L"])
        self.assertNotIn("step", features["L"])
        features["L"] = features["B"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            models.QuantileModels().fit(rows, features)
        rows, features = self.fixture()
        features["A"] = features["A"].iloc[:, ::-1]
        with self.assertRaisesRegex(ValueError, "exactly"):
            models.QuantileModels().fit(rows, features)

    def test_forced_crossing_is_counted_before_row_sorting(self):
        rows, features = self.fixture()
        fitted = models.QuantileModels().fit(rows, features)
        for condition in FEATURE_SETS:
            for label, value in zip(models.PREDICTION_COLUMNS, (30., 20., 10.)):
                fitted.models[condition][label].predict = lambda frame, value=value, **kw: np.full(len(frame), value)
                fitted.selection[condition][label]["settings"]["target_transform"] = "identity"
        predictions, crossing = fitted.predict(features)
        self.assertEqual(crossing, dict.fromkeys(FEATURE_SETS, 1.))
        for prediction in predictions.values():
            np.testing.assert_array_equal(prediction.to_numpy(), np.tile([10., 20., 30.], (len(rows), 1)))
    def test_log_inverse_is_used_for_early_stopping_selection_and_predictions(self):
        rows, features = self.fixture()
        rows.completion_tokens = 6.
        masks = split.split_masks(rows)
        evaluations = []

        class Booster:
            best_iteration = 3

            def __init__(self, value, width):
                self.value, self.width = value, width

            def predict(self, frame, **kwargs):
                return np.full(len(frame), self.value)

            def feature_importance(self, **kwargs):
                return np.ones(self.width)

        def train(params, dataset, **kwargs):
            log = dataset.get_label().max() < 3
            value = 2. if log else 12.
            if "feval" in kwargs:
                valid = kwargs["valid_sets"][0]
                score = kwargs["feval"](np.full(len(valid.get_label()), value), valid)
                raw_y = rows.loc[masks.holdout_target, "completion_tokens"]
                expected = models.pinball(raw_y, np.expm1(value) if log else value, params["alpha"])
                self.assertEqual(score, ("original_pinball", expected, False))
                evaluations.append(score)
            else:
                self.assertEqual(kwargs["num_boost_round"], 3)
            return Booster(value, len(dataset.data.columns))

        with patch.object(lgb, "train", side_effect=train):
            first = models.QuantileModels().fit(rows, features)
            for selection in first.selection.values():
                for selected in selection.values():
                    self.assertEqual(selected["settings"]["target_transform"], "log1p")
            for prediction in first.predict(features)[0].values():
                np.testing.assert_allclose(prediction.to_numpy(), np.expm1(2.))
            rows.loc[masks.validation, "completion_tokens"] = 12.
            second = models.QuantileModels().fit(rows, features)
            self.assertEqual(first.selection, second.selection)
            rows.loc[masks.holdout_target, "completion_tokens"] = 12.
            third = models.QuantileModels().fit(rows, features)
            for selection in third.selection.values():
                self.assertTrue(all(s["settings"]["target_transform"] == "identity" for s in selection.values()))
        self.assertEqual(len(evaluations), 3 * 3 * len(models.GRID) * 3)

    def test_cli_generates_aggregate_artifacts_with_primary_ci_from_13_fixture_weeks(self):
        from datetime import timedelta
        rows, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "REVISION").write_text(loader.REVISION)
            for number in range(1, 14):
                start = loader.FIRST_WEEK + timedelta(weeks=number - 1)
                path = root / "release" / f"{start}_{start + timedelta(days=6)}" / "processed"
                path.mkdir(parents=True)
                weekly = rows.loc[rows.week.eq(number)].drop(columns="week")
                weekly.to_parquet(path / "requests.parquet")
                weekly[["session_id"]].drop_duplicates().assign(
                    harness="h", user_id="private-account").to_parquet(path / "sessions.parquet")
                pd.DataFrame({"session_id": pd.Series(dtype="str"), "step": pd.Series(dtype="int"),
                              "category": pd.Series(dtype="str"), "outcome": pd.Series(dtype="str"),
                              "result_tokens": pd.Series(dtype="float")}).to_parquet(path / "tool_calls.parquet")
            with patch.object(models, "BOOTSTRAP_B", 20):
                result = models.run(root, root / "out")
            recorded = json.loads((root / "out" / "models.json").read_text())
            self.assertEqual(recorded["primary"], result["primary"])
            self.assertEqual(recorded["environment"]["lightgbm"], lgb.__version__)
            self.assertEqual(recorded["sets"]["train_target"], {"rows": 200, "sessions": 50})
            self.assertEqual(recorded["sets"]["validation"], {"rows": 60, "sessions": 15})
            self.assertEqual(set(recorded["metrics"]["overall"]["metrics"]),
                             {"fixed", "model", "previous_completion", "session_mean", "A", "L", "B"})
            self.assertEqual(set(recorded["bootstrap"]), {"B_minus_L", "L_minus_A",
                             "conditions_minus_fixed", "conditions_minus_previous_completion"})
            self.assertEqual(recorded["bootstrap"]["B_minus_L"]["reference"], "L")
            self.assertEqual(recorded["bootstrap"]["B_minus_L"]["cluster_columns"], ["week", "session_id"])
            self.assertEqual(recorded["primary"]["ci95"]["request"],
                             recorded["bootstrap"]["B_minus_L"]["groups"]["overall"]["differences"]["B"]["request"]["mean_pinball"])
            for name in ("models.json", "models.md"):
                text = (root / "out" / name).read_text()
                self.assertNotIn("private-session", text)
                self.assertNotIn("private-account", text)
            # Refuse all test directories before writing anything.
            (root / "release" / "2026-08-16_2026-08-22").mkdir()
            with self.assertRaisesRegex(ValueError, "Held-out"):
                models.run(root, root / "forbidden")
            self.assertFalse((root / "forbidden").exists())

    def test_primary_interpretation_uses_request_mean_pinball_ci(self):
        metrics = {"overall": {"metrics": {key: {w: {"mean_pinball": score}
                    for w in ("request", "session")} for key, score in (("L", 2.), ("B", 1.))}}}
        for ci, expected in (([-2, -1], "validation에서 B가 L보다 개선"),
                             ([-1, 0], "차이 미확인"), ([-1, 1], "차이 미확인"),
                             ([.1, 1], "validation에서 B가 L보다 악화")):
            bootstrap = {"B_minus_L": {"groups": {"overall": {"differences": {"B": {
                "request": {"mean_pinball": ci}, "session": {"mean_pinball": [-1, 1]}}}}}}}
            self.assertEqual(models.primary_result(metrics, bootstrap)["interpretation"], expected)


if __name__ == "__main__":
    unittest.main()
