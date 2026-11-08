"""Synthetic checks for EDA aggregation and time semantics."""
import importlib.util
import unittest

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("pandas", "pyarrow", "matplotlib"))
if HAS_DEPS:
    import pandas as pd
    from freeinference.eda import BIN_MS, active_sessions, adjacent_pairs, aggregate_10min


@unittest.skipUnless(HAS_DEPS, "pandas, pyarrow and matplotlib are required")
class EDATests(unittest.TestCase):
    def test_utc_bins_boundary_gaps_and_sums(self):
        requests = pd.DataFrame({"start_ms": [1, BIN_MS - 1, BIN_MS, 3 * BIN_MS],
                                 "prompt_tokens": [1, 2, 4, 8],
                                 "completion_tokens": [10, 20, 40, 80]})
        result = aggregate_10min(requests)
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertEqual(result.index.tolist(), pd.to_datetime([i * BIN_MS for i in range(4)], unit="ms", utc=True).tolist())
        self.assertEqual(result.prompt_tokens.tolist(), [3, 4, 0, 8])
        self.assertEqual(result.completion_tokens.tolist(), [30, 40, 0, 80])
        self.assertEqual(len(aggregate_10min(requests, 0, 5 * BIN_MS)), 5)
        with self.assertRaises(ValueError):
            aggregate_10min(requests, 0, 3 * BIN_MS)

    def test_strictly_completed_adjacent_steps_and_week_isolation(self):
        rows = pd.DataFrame({"week": [1, 1, 1, 1, 2], "session_id": ["s"] * 5,
                             "step": [0, 1, 2, 4, 1], "start_ms": [0, 10, 20, 40, 50],
                             "end_ms": [10, 19, 30, 45, 55], "completion_tokens": [1, 2, 3, 4, 5]})
        pairs = adjacent_pairs(rows)
        self.assertEqual(pairs.step.tolist(), [1])
        self.assertEqual(pairs.next_completion.tolist(), [3])

    def test_active_session_spans_inclusive_and_distinct_across_weeks(self):
        rows = pd.DataFrame({"week": [1, 1, 1, 2], "session_id": ["a", "a", "b", "a"],
                             "start_ms": [0, 10, 10, 20], "end_ms": [5, 20, 10, 25]})
        self.assertEqual(active_sessions(rows).tolist(), [1, 2, 2, 2])

    def test_eda_refuses_held_out_week(self):
        from pathlib import Path
        import tempfile
        from freeinference.eda import run
        from freeinference.loader import REVISION

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "REVISION").write_text(REVISION, encoding="utf-8")
            (root / "release" / "2026-08-16_2026-08-22").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Held-out"):
                run(root, root / "output")
            self.assertFalse((root / "output").exists())
