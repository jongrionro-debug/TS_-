"""Synthetic tests: no production text, network, or data files are needed."""
from copy import deepcopy
from contextlib import closing
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from wildchat import prepare


class CharacterEncoding:
    """Predictable stand-in to test causal feature construction without packages."""
    def encode(self, text, disallowed_special=()):
        return list(text)


BASE = datetime(2023, 4, 1, tzinfo=timezone.utc)


def message(role, content, seconds=None):
    return {"role": role, "content": content,
            "timestamp": BASE + timedelta(seconds=seconds) if seconds is not None else None}


def conversation(identity, day, prompt=None):
    return {"conversation_hash": identity, "timestamp": BASE + timedelta(days=day),
            "conversation": [message("user", prompt or f"prompt {identity}"),
                             message("assistant", f"answer {identity}", day * 86400)]}


def indexed(rows):
    db = sqlite3.connect(":memory:")
    counts, excluded = prepare.index_rows(db, rows)
    boundaries = prepare.assign_splits(db)
    prepare.validate_partitions(db)
    return db, counts, excluded, boundaries


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.encoding = CharacterEncoding()
        self.row = {"conversation": [
            message("user", "a", 0), message("assistant", "bb", 3),
            message("user", "ccc", 10), message("assistant", "dddd", 13),
            message("user", "eeeee", 25), message("assistant", "ffffff", 30),
            message("user", "ggggggg", 50), message("assistant", "hhhhhhhh", 60),
        ]}

    def samples(self, row=None, **kwargs):
        return list(prepare.iter_samples(self.row if row is None else row, self.encoding, **kwargs))

    def test_separate_prompt_and_output_target(self):
        sample = self.samples()[0]
        self.assertEqual(sample["prompt_features"], {"input_tokens": 1, "input_chars": 1, "input_utf8_bytes": 1})
        self.assertEqual(sample["target"], {"output_tokens": 2})
        self.assertNotIn("output_tokens", sample["prompt_features"])
        self.assertEqual(self.samples()[1]["history_features"]["lag_1_output_tokens"], 2)

    def test_current_response_and_future_cannot_change_current_features(self):
        baseline = self.samples()[1]
        changed = deepcopy(self.row)
        changed["timestamp"] = BASE + timedelta(days=999)
        changed["turn"] = 999
        changed["model"] = "future-metadata"
        changed["conversation"][3] = message("assistant", "a much longer changed target", 99999)
        changed["conversation"][4:] = [message("user", "different future", 999999),
                                        message("assistant", "different answer", 9999999)]
        actual = self.samples(changed)[1]
        for field in ("prompt_features", "history_features"):
            self.assertEqual(actual[field], baseline[field])
        self.assertNotEqual(actual["target"], baseline["target"])
        truncated = deepcopy(self.row)
        truncated["conversation"] = truncated["conversation"][:4]
        self.assertEqual(self.samples(truncated)[1]["history_features"], baseline["history_features"])

    def test_lag_order_and_first_turn_missing(self):
        samples = self.samples()
        for lag in range(1, 4):
            first = samples[0]["history_features"]
            self.assertTrue(first[f"lag_{lag}_turn_missing"])
            self.assertTrue(first[f"lag_{lag}_input_missing"])
            self.assertTrue(first[f"lag_{lag}_output_missing"])
            self.assertIsNone(first[f"lag_{lag}_input_tokens"])
            self.assertIsNone(first[f"lag_{lag}_output_chars"])
        last = samples[3]["history_features"]
        self.assertEqual(last["previous_user_turns"], 3)
        for lag, expected in enumerate(((5, 6), (3, 4), (1, 2)), start=1):
            self.assertEqual((last[f"lag_{lag}_input_tokens"], last[f"lag_{lag}_output_tokens"]), expected)
            self.assertEqual(last[f"lag_{lag}_input_chars"], expected[0])
            self.assertFalse(last[f"lag_{lag}_turn_missing"])

    def test_invalid_prior_pair_keeps_its_position(self):
        row = {"conversation": [message("user", "old"), message("assistant", "observed"),
                                message("user", "unanswered"), message("user", "now"),
                                message("assistant", "target")]}
        sample = self.samples(row)[-1]
        history = sample["history_features"]
        self.assertEqual(history["previous_user_turns"], 2)
        self.assertEqual(history["lag_1_input_tokens"], 10)
        self.assertIsNone(history["lag_1_output_tokens"])
        self.assertTrue(history["lag_1_output_missing"])
        self.assertEqual(history["lag_2_output_tokens"], 8)

    def test_user_intervals_require_explicit_semantics_and_historical_times(self):
        default = self.samples()[-1]
        self.assertIsNone(default["history_features"]["lag_1_turn_interval_seconds"])
        self.assertEqual(default["diagnostics"]["interval_exclusion_reasons"]["1"],
                         "user_submission_timestamps_unverified")
        verified = self.samples(verified_user_timestamps=True)[-1]["history_features"]
        self.assertEqual(verified["lag_1_turn_interval_seconds"], 15)
        self.assertEqual(verified["lag_2_turn_interval_seconds"], 10)
        self.assertIsNone(verified["lag_3_turn_interval_seconds"])
        changed = deepcopy(self.row)
        changed["conversation"][6]["timestamp"] = BASE + timedelta(days=999)
        self.assertEqual(self.samples(changed, verified_user_timestamps=True)[-1]["history_features"], verified)

    def test_missing_or_reversed_user_time_never_falls_back_to_assistant(self):
        for replacement, reason in ((None, "user_submission_timestamp_missing_or_invalid"),
                                    (BASE - timedelta(seconds=1), "user_submission_timestamps_nonmonotonic")):
            with self.subTest(reason=reason):
                row = deepcopy(self.row)
                row["conversation"][2]["timestamp"] = replacement
                sample = self.samples(row, verified_user_timestamps=True)[2]
                self.assertIsNone(sample["history_features"]["lag_1_turn_interval_seconds"])
                self.assertTrue(sample["history_features"]["lag_1_turn_interval_missing"])
                self.assertEqual(sample["diagnostics"]["interval_exclusion_reasons"]["1"], reason)

    def test_pair_exclusions_are_exhaustive_and_counted_once(self):
        messages = [message("user", None), message("assistant", "x"),
                    message("user", " "), message("assistant", "x"),
                    message("user", "missing answer"), message("user", "null answer"),
                    message("assistant", None), message("user", "empty answer"),
                    message("assistant", " "), message("user", "ok"), message("assistant", "yes")]
        with closing(sqlite3.connect(":memory:")) as db:
            counts, excluded = prepare.index_rows(db, [{"conversation": messages}])
        self.assertEqual(counts["user_candidates"], 6)
        self.assertEqual(counts["structurally_valid_samples"], 1)
        self.assertEqual(dict(excluded), dict.fromkeys(prepare.PAIR_REASONS, 1))
        self.assertEqual(len(self.samples({"conversation": messages})), 1)


class GroupSplitTests(unittest.TestCase):
    def test_transitive_hash_visible_prefix_groups_have_no_intersection(self):
        rows = [conversation(f"id-{i}", i) for i in range(20)]
        # 0 --same declared hash--> 1 --same first user--> 2.
        rows[1]["conversation_hash"] = rows[0]["conversation_hash"]
        rows[2]["conversation"][0]["content"] = rows[1]["conversation"][0]["content"]
        # Exact visible duplicate with a different declared ID.
        rows[3]["conversation"] = deepcopy(rows[4]["conversation"])
        # Conservatively connect a first user even after a system message.
        rows[5]["conversation"].insert(0, message("system", "preamble"))
        rows[6]["conversation"][0]["content"] = rows[5]["conversation"][1]["content"]
        db, _, _, _ = indexed(rows)
        self.addCleanup(db.close)
        memberships = dict(db.execute("SELECT id, group_id FROM records"))
        self.assertEqual(memberships[0], memberships[1])
        self.assertEqual(memberships[1], memberships[2])
        self.assertEqual(memberships[3], memberships[4])
        self.assertEqual(memberships[5], memberships[6])
        group_sets = {split: {r[0] for r in db.execute("SELECT group_id FROM groups WHERE split=?", (split,))}
                      for split in prepare.SPLITS}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(group_sets[left] & group_sets[right])
        # Every repeated identity (also transitive paths) is confined to one split or purged.
        assignments = dict(db.execute("SELECT group_id, split FROM groups"))
        keys = {}
        for index, row in enumerate(rows):
            for key in prepare.identity_keys(row):
                keys.setdefault(key, set()).add(assignments[memberships[index]])
        self.assertTrue(all(len(splits) == 1 for splits in keys.values()))
        self.assertEqual(prepare.group_summary(db)["connected_components"]["groups"], 16)

    def test_70_15_15_uses_groups_not_samples(self):
        rows = [conversation(str(i), i) for i in range(20)]
        rows[0]["conversation"] += [message("user", "another"), message("assistant", "longer")]
        db, _, _, boundaries = indexed(rows)
        self.addCleanup(db.close)
        partitions, excluded = prepare.partition_summary(db)
        self.assertEqual([partitions[s]["groups"] for s in prepare.SPLITS], [14, 3, 3])
        self.assertEqual([partitions[s]["samples"] for s in prepare.SPLITS], [15, 3, 3])
        self.assertFalse(excluded)
        self.assertEqual(boundaries["train_validation_cutoff"], (BASE + timedelta(days=14)).isoformat())

    def test_spanning_group_purged_whole_and_boundary_ties_go_later(self):
        rows = [conversation(str(i), i) for i in range(20)]
        rows += [conversation("extended", 19, rows[0]["conversation"][0]["content"])]
        db, _, _, _ = indexed(rows)
        self.addCleanup(db.close)
        self.assertEqual(db.execute("SELECT split,exclusion,conversations,samples FROM groups WHERE group_id=0").fetchone(),
                         (None, "crosses_time_boundary", 2, 2))
        self.assertEqual(prepare.split_for_interval(10, 10, (10, 20)), "validation")
        self.assertEqual(prepare.split_for_interval(20, 20, (10, 20)), "test")
        self.assertIsNone(prepare.split_for_interval(9, 10, (10, 20)))
        self.assertIsNone(prepare.split_for_interval(10, 20, (10, 20)))
        tied = [conversation(str(i), 0 if i < 16 else 1) for i in range(20)]
        tie_db, _, _, _ = indexed(tied)
        self.addCleanup(tie_db.close)
        self.assertEqual(tie_db.execute("SELECT COUNT(DISTINCT split) FROM groups WHERE start_us=?",
                                       (prepare.timestamp_us(BASE),)).fetchone()[0], 1)

    def test_missing_record_time_quarantines_entire_connected_group(self):
        rows = [conversation(str(i), i) for i in range(20)]
        rows += [conversation("unknown", 3, rows[0]["conversation"][0]["content"])]
        rows[-1]["timestamp"] = None
        db, _, _, _ = indexed(rows)
        self.addCleanup(db.close)
        self.assertEqual(db.execute("SELECT exclusion,conversations FROM groups WHERE group_id=0").fetchone(),
                         ("missing_or_invalid_record_time", 2))
        self.assertIsNone(prepare.timestamp_us("not-a-time"))
        self.assertIsNone(prepare.timestamp_us(datetime(2023, 4, 1)))
        self.assertEqual(prepare.timestamp_us("2023-04-01T09:00:00+09:00"), prepare.timestamp_us(BASE))

    def test_empty_input_and_no_valid_groups_are_auditable(self):
        for rows in ([], [{"conversation": []}]):
            with self.subTest(rows=rows):
                db, _, _, boundaries = indexed(rows)
                partitions, _ = prepare.partition_summary(db)
                self.assertEqual(boundaries["eligible_groups_before_purge"], 0)
                self.assertEqual(sum(p["samples"] for p in partitions.values()), 0)
                db.close()


HAS_PACKAGES = all(importlib.util.find_spec(name) for name in ("pyarrow", "tiktoken", "datasets"))


@unittest.skipUnless(HAS_PACKAGES, "integration requires the issue-authorized WildChat environment")
class IntegrationTests(unittest.TestCase):
    def test_cli_real_tokenizer_reproducible_outputs_and_report(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import tiktoken
        rows = [conversation(str(i), i, f"질문 {i} 🙂 <|endoftext|>") for i in range(20)]
        rows[0]["conversation"] += [message("user", "다음 질문"), message("assistant", "이전 답변 뒤의 응답")]
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="wildchat-test-") as tmp:
            tmp = Path(tmp)
            source = tmp / "input.parquet"
            pq.write_table(pa.Table.from_pylist(rows), source)
            for name, batch in (("first", "1"), ("second", "7")):
                result = subprocess.run([sys.executable, str(root / "wildchat/prepare.py"),
                                         "--input", str(source), "--output", str(tmp / name),
                                         "--seed", "42", "--batch-size", batch],
                                        capture_output=True, text=True, cwd=root)
                self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "groups.jsonl"):
                self.assertEqual((tmp / "first" / name).read_bytes(), (tmp / "second" / name).read_bytes())
            manifest = json.loads((tmp / "first/manifest.json").read_text())
            self.assertEqual(manifest["written_samples"], 21)
            self.assertEqual(manifest["observed_interval_slots"], 0)
            sample = json.loads((tmp / "first/train.jsonl").read_text().splitlines()[0])
            enc = tiktoken.get_encoding("o200k_base")
            self.assertEqual(sample["prompt_features"]["input_tokens"],
                             len(enc.encode(rows[0]["conversation"][0]["content"], disallowed_special=())))
            self.assertEqual(sample["target"]["output_tokens"],
                             len(enc.encode(rows[0]["conversation"][1]["content"], disallowed_special=())))
            self.assertEqual(set(sample), {"metadata", "prompt_features", "history_features", "target", "diagnostics"})
            self.assertIn("70%/15%/15%", (tmp / "first/preparation.md").read_text())
            # Rerunning into a populated directory fails instead of corrupting artifacts.
            rerun = subprocess.run([sys.executable, str(root / "wildchat/prepare.py"), "--input", str(source),
                                    "--output", str(tmp / "first")], capture_output=True, text=True, cwd=root)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("must be empty", rerun.stderr)
            # A file with no valid pairs must still produce a complete zero-count report.
            invalid = tmp / "invalid.parquet"
            invalid_rows = [conversation("empty", 0)]
            invalid_rows[0]["conversation"][0]["content"] = " "
            pq.write_table(pa.Table.from_pylist(invalid_rows), invalid)
            empty_run = subprocess.run([sys.executable, str(root / "wildchat/prepare.py"),
                                        "--input", str(invalid), "--output", str(tmp / "empty")],
                                       capture_output=True, text=True, cwd=root)
            self.assertEqual(empty_run.returncode, 0, empty_run.stderr)
            empty_manifest = json.loads((tmp / "empty/manifest.json").read_text())
            self.assertEqual(empty_manifest["written_samples"], 0)
            self.assertEqual(empty_manifest["group_exclusions"]["no_valid_samples"]["groups"], 1)


if __name__ == "__main__":
    unittest.main()
