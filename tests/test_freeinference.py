"""Synthetic fixtures for held-out boundaries and aggregate audit behavior."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("pandas", "pyarrow"))
if HAS_DEPS:
    import pandas as pd
    from freeinference import loader
    from freeinference.audit import audit


@unittest.skipUnless(HAS_DEPS, "pandas and pyarrow are required")
class FreeInferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "REVISION").write_text(loader.REVISION, encoding="utf-8")
        self.week = self.make_week("2026-05-17_2026-05-23")

    def make_week(self, name):
        path = self.root / "release" / name
        (path / "processed").mkdir(parents=True)
        pd.DataFrame({"session_id": ["s"], "start_ms": [1], "end_ms": [2],
                      "step": [0], "link_reason": [None], "model": ["m"],
                      "provider": ["p"], "block_ids": [[123]]}).to_parquet(path / "processed" / "requests.parquet")
        pd.DataFrame({"session_id": ["s"], "user_id": ["u"], "harness": ["h"]}).to_parquet(path / "processed" / "sessions.parquet")
        (path / "manifest.json").write_text('{"excluded_sessions": {}}', encoding="utf-8")
        return path

    def test_environment_path_and_no_block_ids(self):
        with patch.dict(os.environ, {"FREEINFERENCE_DATA_DIR": str(self.root)}):
            self.assertEqual(loader.data_dir(), self.root)
            self.assertEqual(loader.weeks(), [self.week])
            self.assertNotIn("block_ids", loader.load_table(self.week, "requests").columns)
            for column in ("block_ids", "block_ids.element"):
                with self.subTest(column=column), self.assertRaises(ValueError):
                    loader.load_table(self.week, "requests", [column])

    def test_test_week_refused_unless_explicit(self):
        test_week = self.make_week("2026-08-16_2026-08-22")
        with self.assertRaisesRegex(ValueError, "Held-out"):
            loader.weeks(self.root)
        with self.assertRaisesRegex(ValueError, "Held-out"):
            loader.load_table(test_week, "requests")
        self.assertIn(test_week, loader.weeks(self.root, allow_test=True))
        self.assertEqual(len(loader.load_table(test_week, "requests", allow_test=True)), 1)

    def test_test_timestamp_refused_for_every_table_and_projection(self):
        path = self.week / "processed" / "requests.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "start_ms"] = loader.TEST_START_MS
        frame.to_parquet(path)
        for table, columns in (("requests", ["model"]), ("sessions", ["harness"])):
            with self.subTest(table=table), self.assertRaisesRegex(ValueError, "timestamp"):
                loader.load_table(self.week, table, columns)
        self.assertEqual(len(loader.load_table(self.week, "requests", ["model"], allow_test=True)), 1)

    def test_revision_mismatch_refused(self):
        (self.root / "REVISION").write_text("different revision", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "REVISION"):
            loader.weeks(self.root)

    def test_audit_requires_complete_train_validation(self):
        with self.assertRaisesRegex(ValueError, "13"):
            audit(self.root)

    def test_audit_aggregates_time_and_link_counts(self):
        import json
        from datetime import date, datetime, timedelta, timezone

        for i in range(13):
            start = date(2026, 5, 17) + timedelta(days=7 * i)
            name = f"{start}_{start + timedelta(days=6)}"
            week = self.week if i == 0 else self.make_week(name)
            ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
            pd.DataFrame({"session_id": ["s", "s"], "step": [0, 1],
                          "start_ms": [ms, ms + 1], "end_ms": [ms + 2, ms + 3],
                          "link_reason": [None, "tool_call_id"], "model": ["m", "m"],
                          "provider": ["p", "p"], "block_ids": [[5], [6]]}).to_parquet(week / "processed" / "requests.parquet")
            (week / "manifest.json").write_text(json.dumps({"excluded_sessions": {"reason": 1}}))
        result = audit(self.root)
        self.assertEqual(result["overall"]["requests"], 26)
        self.assertEqual(result["overall"]["adjacent_step_pairs_not_strictly_completed"], 13)
        self.assertEqual(result["overall"]["adjacent_step_end_order_inversions"], 0)
        self.assertEqual(result["overall"]["excluded_sessions"], {"reason": 13})
        self.assertEqual(result["overall"]["link_reason"]["tool_call_id"]["count"], 13)

    def test_link_pairs_inversions_and_week_boundary(self):
        from datetime import date, datetime, timedelta, timezone

        for i in range(1, 13):
            start = date(2026, 5, 17) + timedelta(days=7 * i)
            self.make_week(f"{start}_{start + timedelta(days=6)}")
        base = int(datetime(2026, 5, 17, tzinfo=timezone.utc).timestamp() * 1000)
        boundary = base + 7 * 24 * 60 * 60 * 1000
        pd.DataFrame({"session_id": ["s"] * 5, "step": [0, 1, 2, 3, 4],
                      "start_ms": [base, base + 200, base + 300, base + 500, base + 550],
                      "end_ms": [base + 100, base + 400, base + 350, base + 600, boundary],
                      "link_reason": [None, "tool_call_id", "tool_call_id", "assistant_message", "assistant_message"],
                      "model": ["m"] * 5, "provider": ["p"] * 5}).to_parquet(self.week / "processed" / "requests.parquet")
        overall = audit(self.root)["overall"]
        for reason in ("tool_call_id", "assistant_message"):
            self.assertEqual(overall["link_reason_adjacent_pairs"][reason],
                             {"later_step_requests": 2, "adjacent_pairs": 2,
                              "strictly_completed": 1, "strictly_completed_rate": 0.5,
                              "not_strictly_completed": 1, "unassessable_requests": 0})
        self.assertEqual(overall["adjacent_step_end_order_inversions"], 1)
        self.assertEqual(overall["sessions_crossing_week_boundary"], 1)
