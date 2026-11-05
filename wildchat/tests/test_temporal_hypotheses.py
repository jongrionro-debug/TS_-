"""Synthetic safety/metric checks only; no production data or model calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest import mock

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    from wildchat import temporal_causal as causal
    from wildchat import temporal_hypotheses as th
    from wildchat import revalidate as rv


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class TemporalSafetyTests(unittest.TestCase):
    def rows(self):
        n = 6
        return causal.HistoryRows(
            x=np.column_stack((np.arange(1, n+1), np.zeros((n, 30)))),
            y=np.array([10., 20., 30., 8., 16., 32.]),
            groups=np.array([0, 0, 0, 1, 1, 1]), sources=np.array([0, 0, 0, 1, 1, 1]),
            turns=np.array([0, 1, 2, 0, 1, 2]), stamps=np.ones(n, dtype=np.int64),
            users=np.array(["a", "a", "a", "b", "b", "b"]),
            previous=np.full(n, np.nan), cumulative=np.full(n, np.nan),
            indices=np.array([0, 2, 4, 0, 2, 4]),
            anchors=np.full(n, causal.MISSING_TIME, dtype=np.int64),
            completions=np.full(n, causal.MISSING_TIME, dtype=np.int64),
            embeddings=np.zeros((n, 384)))

    def test_current_future_labels_and_other_conversations_do_not_change_prefix(self):
        rows = self.rows()
        pcs = np.random.default_rng(42).normal(size=(6, 32))
        before = th.sequence_features(rows, pcs)
        rows.y[2:] = 900000
        pcs[3:] = 900000
        after = th.sequence_features(rows, pcs)
        np.testing.assert_allclose(before[0][:3], after[0][:3], equal_nan=True)
        np.testing.assert_allclose(before[1][:3], after[1][:3], equal_nan=True)
        self.assertEqual(before[0][2, 0], 13.)
        self.assertEqual(before[0][2, 3], 10.)
        self.assertEqual(before[0][3, -1], 0)
        self.assertTrue(np.isnan(before[1][3]).all())

    def test_prefix_time_checks_do_not_use_later_response_or_recorded_time(self):
        messages = [{"role": "user"}, {"role": "assistant", "timestamp": "2023-04-10T01:00:00Z"},
                    {"role": "user"}, {"role": "assistant", "timestamp": "2023-04-10T01:03:00Z"},
                    {"role": "user"}, {"role": "assistant", "timestamp": "2020-01-01T00:00:00Z"}]
        first, _ = causal.temporal_metadata(messages, 0)
        messages[-1]["timestamp"] = None
        changed, _ = causal.temporal_metadata(messages, None)
        self.assertEqual(first, changed)
        self.assertIsNone(first[0])
        self.assertEqual(first[4], rv.timestamp_us("2023-04-10T01:03:00Z"))
        messages[1]["timestamp"] = None
        bad, _ = causal.temporal_metadata(messages, None)
        self.assertIsNone(bad[4])

    def test_metadata_loader_uses_only_prefix_and_keeps_cadence_through_take(self):
        rows = self.rows().take(np.arange(6) < 3)
        messages = [{"role": "user"}, {"role": "assistant", "timestamp": "2023-04-10T01:00:00Z"},
                    {"role": "user"}, {"role": "assistant", "timestamp": "2023-04-10T01:03:00Z"},
                    {"role": "user"}, {"role": "assistant", "timestamp": None}]
        rows.stamps[:] = rv.timestamp_us("2023-04-10T02:00:00Z")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"train.jsonl").write_text("".join(json.dumps({"metadata": {"user_message_index": int(i)}})+"\n" for i in rows.indices))
            side = root/"train_validation_metadata.jsonl"
            side.write_text(json.dumps({"source_row_id": 0, "hashed_ip": "fixture", "recorded_at": "2023-04-10T02:00:00Z", "messages": messages})+"\n")
            actual = th.load_metadata({"train": rows}, root, side)["train"]
        self.assertTrue(np.isnan(actual.x[0, 31]))
        self.assertTrue(np.isnan(actual.x[1, 31]))
        self.assertEqual(actual.x[2, 31], 180.)
        self.assertEqual(actual.take(np.array([False, False, True])).x[0, 31], 180.)
        self.assertTrue((actual.completions == causal.MISSING_TIME).all())

    def test_calendar_uses_only_anchor_not_targets_or_final_times(self):
        rows = self.rows()
        rows.anchors[1] = rv.timestamp_us("2023-04-10T01:00:00Z")
        a = th.calendar_features(rows)
        rows.y[:] = 999999
        rows.stamps[:] = 999999
        np.testing.assert_allclose(a, th.calendar_features(rows), equal_nan=True)
        self.assertTrue(np.isnan(a[0, :-1]).all())
        self.assertEqual(a[0, -1], 1)
        self.assertEqual(a[1, -1], 0)

    def test_mixed_cache_is_rejected_before_vectors_are_materialized(self):
        rows = self.rows()
        class Cache:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __getitem__(self, key):
                if key != "keys": raise AssertionError("vectors must not be loaded")
                return np.array([[999, 999]])
        with mock.patch.object(np, "load", return_value=Cache()):
            with self.assertRaisesRegex(ValueError, "exactly train/validation keys"):
                th.attach_embeddings({"train": rows}, Path("unused.npz"))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "requires pyarrow")
    def test_source_projection_materializes_only_allowed_metadata(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = self.rows().take(np.arange(6) < 3)
        rows.stamps[:] = rv.timestamp_us("2023-04-10T02:00:00Z")
        messages = [{"role": "user", "timestamp": None, "content": "synthetic raw text"},
                    {"role": "assistant", "timestamp": "2023-04-10T01:00:00Z", "content": "ignored"},
                    {"role": "user", "timestamp": None, "content": "ignored"},
                    {"role": "assistant", "timestamp": "2023-04-10T01:03:00Z", "content": "ignored"},
                    {"role": "user", "timestamp": None, "content": "ignored"}]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"train.jsonl").write_text("".join(json.dumps({"metadata": {"user_message_index": int(i)}})+"\n" for i in rows.indices))
            source = root/"fixture.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"hashed_ip": "allowed", "timestamp": "2023-04-10T02:00:00Z", "conversation": messages},
                {"hashed_ip": "excluded", "timestamp": "invalid", "conversation": messages}]), source)
            actual, audit = th.attach_source_metadata({"train": rows}, root, source, {"name": source.name, "rows": 2})
        self.assertEqual(actual["train"].users.tolist(), ["allowed"]*3)
        self.assertEqual(audit["selected_sources"], 1)
        self.assertEqual(audit["python_metadata_sources_outside_allowlist"], 0)
        self.assertFalse(any("content" in col for col in audit["projected_columns"]))
        self.assertEqual(actual["train"].x[2, 31], 180.)

    def test_cli_guard_denies_heldout_filename_before_file_open(self):
        # Use only a nonexistent synthetic path, never the actual dataset.
        code = '''
import sys
from pathlib import Path
from wildchat import temporal_hypotheses as th
sys.argv = ["fixture", "--prepared", "/nonexistent_fixture", "--source", "/nonexistent_fixture.parquet", "--check"]
th.main()
try:
    Path("/nonexistent_fixture/test.jsonl").open()
except PermissionError:
    print("GUARD_DENIED")
else:
    raise AssertionError("heldout open not denied")
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GUARD_DENIED", result.stdout)

    def test_paired_bootstrap_matches_explicit_user_sampling(self):
        rows = self.rows()
        base = np.column_stack((rows.y*.5, rows.y*.9, rows.y*1.5))
        preds = {m: np.maximum(base+j, 0) for j, m in enumerate(causal.MODELS)}
        metrics, boot = causal.evaluate(rows, preds, np.ones(6, dtype=bool), 42, 80)
        rng = np.random.Generator(np.random.PCG64(42))
        explicit = {(c, w): [] for c in rv.cohorts(rows) for w in ("row", "user")}
        for _ in range(80):
            chosen = rng.integers(0, 2, size=2)
            clusters = [np.flatnonzero(rows.users == ["a", "b"][u]) for u in chosen]
            for cohort, mask in rv.cohorts(rows).items():
                clusters_c = [ix[mask[ix]] for ix in clusters if mask[ix].any()]
                ix = np.concatenate(clusters_c)
                for weight in ("row", "user"):
                    deltas = {}
                    for m in ("baseline", causal.MODELS[1]):
                        if weight == "row":
                            deltas[m] = rv.contributions(rows.y[ix], preds[m][ix])[:, 0].mean()
                        else:
                            deltas[m] = np.mean([rv.contributions(rows.y[k], preds[m][k])[:, 0].mean() for k in clusters_c])
                    explicit[cohort, weight].append(deltas[causal.MODELS[1]]-deltas["baseline"])
        for (cohort, weight), values in explicit.items():
            actual = boot["all_users"][cohort][weight][causal.MODELS[1]+"-baseline"]
            np.testing.assert_allclose(actual["ci95"], np.quantile(values, [.025, .975]), atol=1e-12)
            self.assertEqual(actual["effective_replicates"], 80)
        self.assertEqual(metrics["all_users"]["all"]["row"]["baseline"]["n_rows"], 6)


if __name__ == "__main__":
    unittest.main()
