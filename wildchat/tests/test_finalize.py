"""Final evaluation safety checks: synthetic fixture.jsonl only, no test data."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_LIGHTGBM = HAS_NUMPY and importlib.util.find_spec("lightgbm") is not None
if HAS_NUMPY:
    import numpy as np
    from wildchat import finalize as final
    from wildchat.tests.test_experiment import sample
    from wildchat.tests.test_revalidate import rows


@unittest.skipUnless(HAS_NUMPY, "requires existing NumPy environment")
class FinalSafetyTests(unittest.TestCase):
    def test_guard_rejects_each_existing_artifact_and_failed_attempt(self):
        for name in ("test_final.log", "test_final.json", "test_final.md"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / name).touch()
                with self.assertRaises(FileExistsError):
                    final.reserve(output, "hash", 42)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with final.reserve(output, "frozen", 42) as handle:
                final.event(handle, "failed_no_retry", error="synthetic failure")
            log = [json.loads(line) for line in (output / "test_final.log").read_text().splitlines()]
            self.assertEqual(log[0]["config_sha256"], "frozen")
            self.assertEqual(log[0]["seed"], 42)
            self.assertEqual(log[1]["status"], "failed_no_retry")
            with self.assertRaises(FileExistsError):
                final.reserve(output, "frozen", 42)

    def test_synthetic_loader_streams_once_and_current_target_is_not_history(self):
        samples = [sample(10, turn, "test") for turn in range(3)]
        expected = {"samples": 3, "groups": 1, "conversations": 1,
                    "recorded_at_min": samples[0]["metadata"]["recorded_at"],
                    "recorded_at_max": samples[0]["metadata"]["recorded_at"]}
        cutoff = final.rv.timestamp_us(expected["recorded_at_min"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            path.write_text("".join(json.dumps(s) + "\n" for s in samples))
            real_open = Path.open
            opens = []
            def guarded_open(p, *args, **kw):
                self.assertNotEqual(p.name, "test.jsonl")
                opens.append(p)
                return real_open(p, *args, **kw)
            with mock.patch.object(Path, "open", guarded_open):
                data, digest = final.load_final(path, expected, final.rv.feature_names(3)["B"], cutoff)
            self.assertEqual(opens, [path])
            self.assertEqual(digest, final.rv.fingerprint(path))
            np.testing.assert_allclose(data.cumulative,
                [np.nan, samples[0]["target"]["output_tokens"],
                 np.mean([s["target"]["output_tokens"] for s in samples[:2]])], equal_nan=True)
            samples[1]["target"]["output_tokens"] += 9999
            path.write_text("".join(json.dumps(s) + "\n" for s in samples))
            changed, _ = final.load_final(path, expected, final.rv.feature_names(3)["B"], cutoff)
            np.testing.assert_array_equal(changed.x[:2], data.x[:2])
            np.testing.assert_allclose(changed.cumulative[:2], data.cumulative[:2], equal_nan=True)
            self.assertNotEqual(changed.cumulative[2], data.cumulative[2])
            with self.assertRaises(ValueError):
                final.load_final(path, expected | {"samples": 4}, final.rv.feature_names(3)["B"], cutoff)
            samples[1]["metadata"]["turn_index"] = 0
            samples[1]["history_features"]["previous_user_turns"] = 0
            samples[1]["metadata"]["user_message_index"] = 3
            path.write_text("".join(json.dumps(s) + "\n" for s in samples))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                final.load_final(path, expected, final.rv.feature_names(3)["B"], cutoff)

    def test_exclusions_remove_whole_groups_without_changing_known_data(self):
        known = {"train": rows([1], ["train-user"], [1], stamps=[1]),
                 "validation": rows([2], ["val-user"], [2], stamps=[5])}
        data = rows([3] * 8, ["train-user", "peer", "val-user", "", "fresh1", "fresh2", "fresh3", "fresh4"],
                    [10, 10, 11, 12, 13, 14, 15, 1], stamps=[10, 10, 10, 10, 10, 9, 10, 10])
        kept, audit = final.filter_final(data, known, 10)
        self.assertEqual(kept.users.tolist(), ["fresh1", "fresh3"])
        self.assertEqual(audit["removed_rows"], 6)
        self.assertFalse(audit["training_changed"])
        self.assertEqual(known["train"].users.tolist(), ["train-user"])
        self.assertEqual(known["validation"].users.tolist(), ["val-user"])
        self.assertTrue(audit["checks"]["strict_chronology"])

    def test_config_tamper_fails_before_reserving_or_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "frozen.json"
            config.write_text('{}\n')
            config.with_suffix('.sha256').write_text('wrong  frozen.json\n')
            with mock.patch.object(final, "CONFIG", config), mock.patch.object(final, "load_final") as load:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    final.evaluate()
                load.assert_not_called()

    def test_existing_log_stops_evaluate_before_any_data_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "test_final.log").touch()
            with mock.patch.object(final, "OUTPUT", output), \
                 mock.patch.object(final, "checked_config", return_value=({"seed": 42}, "hash")), \
                 mock.patch.object(final.rv, "load_allowed") as load, \
                 mock.patch.object(final, "load_final") as load_final:
                with self.assertRaises(FileExistsError):
                    final.evaluate()
                load.assert_not_called()
                load_final.assert_not_called()


@unittest.skipUnless(HAS_LIGHTGBM, "requires existing LightGBM environment")
class FrozenTrainingTests(unittest.TestCase):
    def test_identical_to_issue13_refit_with_frozen_iterations_and_no_callbacks(self):
        import lightgbm as lgb
        names = final.rv.feature_names(3)
        samples = [sample(g, t) for g in range(50) for t in range(3)]
        train = rows([s["target"]["output_tokens"] for s in samples], [f"user-{g}" for g in range(len(samples))])
        train.x = np.asarray([final.rv.feature_vector(s, names["B"]) for s in samples])
        train.previous[:] = 10
        train.cumulative[:] = 20
        config = {"features": names, "threads": 1, "common_parameters": final.rv.common_parameters(42, 1),
                  "baseline_fit": final.rv.fit_baselines(train),
                  "models": {n: {"selected_parameters": {"num_leaves": 7, "min_data_in_leaf": 5},
                            "selected_quantile_fits": [{"quantile": q, "best_iteration": i + 2}
                                                       for i, q in enumerate(final.rv.QUANTILES)]} for n in ("A", "B")}}
        real_train, calls = lgb.train, []
        def checked_train(params, dataset, **kwargs):
            self.assertNotIn("valid_sets", kwargs)
            self.assertNotIn("callbacks", kwargs)
            calls.append((params, kwargs["num_boost_round"]))
            return real_train(params, dataset, **kwargs)
        with mock.patch.object(lgb, "train", checked_train):
            models = final.fit_models(train, config)
        self.assertEqual([n for _, n in calls], [2, 3, 4, 2, 3, 4])
        predicted, diagnostics = final.predict(train, models, config)
        for name in ("A", "B"):
            columns = [names["B"].index(n) for n in names[name]]
            expected, diag = final.rv.refit_predict(train, train, columns, names[name], config["models"][name],
                                                  argparse.Namespace(seed=42, threads=1))
            np.testing.assert_array_equal(predicted[name], expected)
            self.assertEqual(diagnostics[name], diag)


if __name__ == "__main__":
    unittest.main()
