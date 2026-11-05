"""Issue 21: synthetic train/validation only; no final-evaluation module imports."""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TRAINING = HAS_NUMPY and all(importlib.util.find_spec(p) is not None for p in ("lightgbm", "pyarrow"))
if HAS_NUMPY:
    import numpy as np
    from wildchat import cross_history as ch
    from wildchat import revalidate as rv


def rows(targets, users, sources, completions, anchors, groups=None, turns=None):
    n = len(targets)
    return ch.HistoryRows(x=np.zeros((n, 31)), y=np.asarray(targets, dtype=float),
                          groups=np.asarray(sources if groups is None else groups),
                          sources=np.asarray(sources), turns=np.asarray([1] * n if turns is None else turns),
                          stamps=np.asarray(completions), users=np.asarray(users, dtype=object),
                          previous=np.full(n, np.nan), cumulative=np.full(n, np.nan),
                          indices=np.arange(n), anchors=np.asarray(anchors, dtype=np.int64),
                          completions=np.asarray(completions, dtype=np.int64), embeddings=np.zeros((n, 384)))


@unittest.skipUnless(HAS_NUMPY, "requires existing NumPy environment")
class CausalityTests(unittest.TestCase):
    def features(self, data, partition="train"):
        with redirect_stdout(io.StringIO()):
            return ch.cross_features(data, partition)

    def test_current_and_future_completion_quality_cannot_change_prefix_anchor(self):
        base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        messages = [{"role": "user", "timestamp": None},
                    {"role": "assistant", "timestamp": base},
                    {"role": "user", "timestamp": None},
                    {"role": "assistant", "timestamp": base + timedelta(seconds=5)},
                    {"role": "user", "timestamp": None},
                    {"role": "assistant", "timestamp": base + timedelta(seconds=10)}]
        anchors, complete = ch.temporal_metadata(messages, rv.timestamp_us(messages[-1]["timestamp"]))
        self.assertEqual(anchors[2], rv.timestamp_us(base))
        self.assertIsNotNone(complete)
        for bad in (None, base - timedelta(seconds=1), base + timedelta(days=100)):
            altered = [dict(m) for m in messages]
            altered[3]["timestamp"] = bad
            changed, _ = ch.temporal_metadata(altered, rv.timestamp_us(messages[-1]["timestamp"]))
            self.assertEqual(changed[2], anchors[2])
            self.assertIsNone(changed[0])
        changed, complete = ch.temporal_metadata(messages, None)
        self.assertEqual(changed, anchors)
        self.assertIsNone(complete)

    def test_strict_past_other_conversation_same_user_and_missing_first_anchor(self):
        data = rows([10, 30, 100, 900, 999], ["a", "a", "a", "b", "a"],
                    [0, 0, 1, 2, 3], [10, 10, 30, 5, 20],
                    [ch.MISSING_TIME, 3, 20, ch.MISSING_TIME, 10], turns=[0, 1, 1, 0, 1])
        x, available, audit = self.features(data)
        # Source 0 is fully complete at 10, source 3 ties anchor 20 and
        # cannot enter source 1's features. Source 2 is another user.
        np.testing.assert_allclose(x[2, :7], [20, 20, 20, np.log1p(20), 1, .00001, 0])
        self.assertEqual(available.tolist(), [False, False, True, False, False])
        self.assertTrue(np.isnan(x[0, :4]).all())
        self.assertTrue(np.isnan(x[0, 8:]).all())
        self.assertEqual(audit["checked_history_links"], 1)
        self.assertEqual(audit["violations"], 0)

    def test_future_current_duplicate_group_and_foreign_user_targets_cannot_change_features(self):
        data = rows([10, 800, 999, 55, 500], ["a", "a", "a", "b", "a"],
                    [0, 1, 2, 3, 4], [10, 80, 25, 5, 15], [0, 20, 10, 0, 0],
                    groups=[0, 1, 2, 3, 1])
        original, _, _ = self.features(data)
        changed = replace(data, y=np.asarray([10, 999999, 333333, 444444, 777777.]))
        altered, _, _ = self.features(changed)
        # Current source 1, future source 2, foreign user 3 and same duplicate
        # group source 4 are all excluded for the query in source 1.
        np.testing.assert_allclose(original[1], altered[1], equal_nan=True)
        self.assertEqual(original[1, 0], 10)
        self.assertEqual(original[1, 4], 1)

    def test_conversation_completion_required_not_just_an_earlier_turn(self):
        data = rows([1, 1000, 200], ["a"] * 3, [0, 0, 1], [100, 100, 90], [0, 10, 50])
        x, available, _ = self.features(data)
        self.assertFalse(available[2])
        self.assertTrue(np.isnan(x[2, 0]))

    def test_partition_rebuild_does_not_inherit_history_and_input_order_is_irrelevant(self):
        data = rows([10, 20, 30], ["a"] * 3, [0, 1, 2], [10, 20, 40], [0, 11, 30])
        full, _, _ = self.features(data)
        order = np.asarray([2, 0, 1])
        shuffled, _, _ = self.features(data.take(order))
        np.testing.assert_allclose(shuffled[np.argsort(order)], full, equal_nan=True)
        isolated, available, audit = self.features(data.take(np.asarray([2])), "inner_holdout")
        self.assertFalse(available.any())
        self.assertEqual(audit["checked_history_links"], 0)
        self.assertEqual(isolated[0, 4], 0)

    def test_ewma_median_counts_and_duplicate_group_deduplication(self):
        data = rows([10, 30, 100, 200, 500, 999], ["a"] * 6,
                    [0, 0, 1, 2, 3, 4], [10, 10, 20, 25, 30, 50], [0, 0, 0, 0, 0, 40],
                    groups=[0, 0, 1, 1, 3, 4])
        x, _, audit = self.features(data)
        self.assertEqual(x[-1, 0], 500)
        self.assertEqual(x[-1, 1], 500)
        self.assertAlmostEqual(x[-1, 2], .3 * 500 + .7 * (.3 * 100 + .7 * 20))
        self.assertEqual(x[-1, 4], 3)
        self.assertEqual(audit["eligible_completed_conversation_groups"], 4)


@unittest.skipUnless(HAS_NUMPY, "requires existing NumPy environment")
class EvaluationTests(unittest.TestCase):
    def test_all_metrics_and_bootstrap_match_explicit_user_draws(self):
        data = rows([0, 10, 2, 7, 20], ["a", "a", "b", "c", "c"],
                    [0, 0, 1, 2, 2], [10] * 5, [0] * 5, turns=[0, 1, 0, 0, 1])
        predictions = {m: np.asarray([[1, 2, 4], [3, 6, 8], [2, 4, 6], [4, 7, 9], [9, 12, 25.]], dtype=float) + i
                       for i, m in enumerate(ch.MODELS)}
        predictions[ch.MODELS[2]][0] = [12, 8, 3]  # Retain crossing and signed width.
        available = np.asarray([False, True, False, False, False])
        metrics, bootstrap = ch.evaluate(data, predictions, available, 17, 50)
        m = metrics["all_users"]["all"]["row"][ch.MODELS[2]]
        self.assertGreater(m["crossing_rate"], 0)
        self.assertEqual(m["n_users"], 3)
        self.assertEqual(metrics["users_with_history"]["first_turn"]["row"]["B"]["n_rows"], 1)
        self.assertEqual(metrics["users_without_history"]["all"]["row"]["B"]["n_rows"], 3)
        self.assertAlmostEqual(m["mean_pinball_loss"], np.mean([m[k] for k in ("pinball_p10", "pinball_p50", "pinball_p90")]))
        rng = np.random.Generator(np.random.PCG64(17))
        explicit = {w: [] for w in ("row", "user")}
        for _ in range(50):
            draw = rng.integers(0, 3, size=3)
            groups = [np.flatnonzero(data.users == ["a", "b", "c"][i]) for i in draw]
            ix = np.concatenate(groups)
            a, b = ch.COMPARISONS[0]
            loss = rv.contributions(data.y, predictions[a])[:, 0] - rv.contributions(data.y, predictions[b])[:, 0]
            explicit["row"].append(loss[ix].mean())
            explicit["user"].append(np.mean([loss[g].mean() for g in groups]))
        for w in explicit:
            actual = bootstrap["all_users"]["all"][w]["B_cross_hist-B"]
            np.testing.assert_allclose(actual["ci95"], np.quantile(explicit[w], [.025, .975], method="linear"))
        self.assertEqual((metrics, bootstrap), ch.evaluate(data, predictions, available, 17, 50))

    def test_empty_subgroups_are_null_not_zero_or_nan(self):
        data = rows([1, 2], ["a", "b"], [0, 1], [10, 20], [0, 0], turns=[0, 0])
        predictions = {m: np.tile([0., 1., 3.], (2, 1)) for m in ch.MODELS}
        metrics, boot = ch.evaluate(data, predictions, np.zeros(2, dtype=bool), 42, 8)
        self.assertIsNone(metrics["users_with_history"]["all"]["row"]["B"])
        self.assertIsNone(boot["all_users"]["followup_turn"]["row"]["B_cross_hist-B"]["ci95"])
        self.assertEqual(boot["users_with_history"]["all"]["user"]["B_cross_hist-B"]["empty_replicates"], 8)
        json.dumps([metrics, boot], allow_nan=False)


@unittest.skipUnless(HAS_TRAINING, "requires existing LightGBM/PyArrow environment")
class IntegrationTests(unittest.TestCase):
    def test_metadata_projection_train_only_selection_and_full_run(self):
        import lightgbm as lgb
        import pyarrow as pa
        import pyarrow.parquet as pq
        base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory(prefix=".issue21-", dir=Path(__file__).resolve().parents[2]) as directory:
            root = Path(directory)
            source = root / "synthetic.parquet"
            source_rows, partitions, keys = [], {}, []
            for split, ids in (("train", range(40)), ("validation", range(40, 50))):
                samples = []
                for group in ids:
                    completion = base + timedelta(days=group, seconds=10)
                    messages = []
                    for turn in range(3):
                        messages += [{"role": "user", "timestamp": None},
                                     {"role": "assistant", "timestamp": completion - timedelta(seconds=2 - turn)}]
                        history = {"previous_user_turns": turn}
                        for lag in range(1, 4):
                            for side in ("input", "output"):
                                for unit in ("tokens", "chars", "utf8_bytes"):
                                    history[f"lag_{lag}_{side}_{unit}"] = None if turn < lag else 10 + turn - lag
                                history[f"lag_{lag}_{side}_missing"] = turn < lag
                            history[f"lag_{lag}_turn_missing"] = turn < lag
                        target = 10 + turn + (1000 if split == "validation" else 0)
                        samples.append({"metadata": {"source_row_id": group, "group_id": group, "split": split,
                                                     "turn_index": turn, "user_message_index": turn * 2,
                                                     "recorded_at": completion.isoformat()},
                                        "prompt_features": dict(zip(rv.feature_names(3)["A"], [10, 30, 40])),
                                        "history_features": history, "target": {"output_tokens": target}})
                        keys.append([group, turn * 2])
                    source_rows.append({"hashed_ip": f"fixture-{group // 2}", "timestamp": completion,
                                        "conversation": messages, "header": {"user-agent": "synthetic", "accept-language": "en"},
                                        "country": "fixture", "state": None, "model": "fixture"})
                (root / (split + ".jsonl")).write_text("".join(json.dumps(s) + "\n" for s in samples))
                partitions[split] = {"samples": len(samples), "groups": len(ids), "conversations": len(ids),
                                     "recorded_at_min": samples[0]["metadata"]["recorded_at"],
                                     "recorded_at_max": samples[-1]["metadata"]["recorded_at"]}
            pq.write_table(pa.Table.from_pylist(source_rows), source)
            manifest = {"partitions": partitions, "lags": 3,
                        "input_files": [{"name": source.name, "rows": 50, "sha256": "synthetic"}],
                        "boundaries": {"train_validation_cutoff": (base + timedelta(days=40)).isoformat()}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            vectors = np.random.default_rng(42).normal(size=(len(keys), 384)).astype(np.float16)
            np.savez(root / "embeddings.npz", keys=np.asarray(keys), vectors=vectors)
            args = argparse.Namespace(prepared=root, source=source, embeddings=root / "embeddings.npz", output=root / "results",
                                      seed=42, inner_fraction=.8, max_rounds=2, early_stopping_rounds=1,
                                      threads=1, bootstrap_replicates=8)
            original_open, original_train, original_pca = Path.open, lgb.train, ch.PCA.fit
            original_batches = pq.ParquetFile.iter_batches
            selections, pca_fits, projections = [], [], []

            def guarded_open(path, *a, **kw):
                if path.suffix == ".jsonl":
                    self.assertIn(path.name, ("train.jsonl", "validation.jsonl"))
                return original_open(path, *a, **kw)

            def checked_train(params, dataset, *a, **kw):
                self.assertLess(np.max(dataset.get_label()), 100)
                for holdout in kw.get("valid_sets", []):
                    self.assertLess(np.max(holdout.get_label()), 100)
                selections.append("valid_sets" in kw)
                return original_train(params, dataset, *a, **kw)

            def checked_pca(pca, x, k):
                self.assertFalse(any(np.array_equal(row, v) for row in x for v in vectors[120:].astype(np.float32)))
                pca_fits.append(len(x))
                return original_pca(pca, x, k)

            def checked_batches(parquet, *a, **kw):
                projections.append(kw["columns"])
                self.assertFalse(any("content" in c or c == "conversation" for c in kw["columns"]))
                return original_batches(parquet, *a, **kw)

            with mock.patch.object(Path, "open", guarded_open), mock.patch.object(lgb, "train", checked_train), \
                    mock.patch.object(ch.PCA, "fit", checked_pca), mock.patch.object(pq.ParquetFile, "iter_batches", checked_batches), \
                    redirect_stdout(io.StringIO()):
                result = ch.run(args)
            self.assertEqual(selections, [True] * 72 + [False] * 12)
            self.assertEqual(pca_fits, [96, 120])
            self.assertEqual(len(projections), 1)
            self.assertEqual(result["split_audits"]["outer"]["checks"]["user_intersection"], 0)
            self.assertEqual(result["history_audits"]["validation"]["history_rows"], 10)
            self.assertEqual(result["metadata_audit"]["user_timestamps_present"], 0)
            self.assertEqual(set(result["selection"]), set(ch.MODELS))
            self.assertEqual(set(result["validation_metrics"]), {"all_users", "users_with_history", "users_without_history"})
            self.assertEqual(json.loads((args.output / "cross_history.json").read_text())["issue"], 21)
            report = (args.output / "cross_history.md").read_text()
            self.assertIn("P50 MAE", report)
            self.assertNotIn("fixture-", report)
            self.assertNotIn("fixture-", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
