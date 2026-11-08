"""Synthetic checks for current-request input features."""
import importlib.util
import unittest

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("pandas", "pyarrow"))
if HAS_DEPS:
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from freeinference.features import current_input_features


@unittest.skipUnless(HAS_DEPS, "pandas and pyarrow are required")
class CurrentInputFeatureTests(unittest.TestCase):
    def test_current_response_changes_do_not_change_features(self):
        rows = pd.DataFrame([{
            "prompt_tokens": 20, "cache_read_tokens": 3, "n_messages": 2,
            "model": "m", "provider": "p", "stream": True,
            "role_tokens": {"system": 1, "user": 2, "assistant": 3, "tool": 4,
                            "tool_definitions": 5},
            "tool_definition_names": ["one", "two"], "completion_tokens": 10,
            "end_ms": 50, "ttft_ms": 5, "finish_reason": "stop",
            "tool_calls": [{"category": "one"}],
        }], index=[7])
        changed = rows.copy(deep=True)
        changed.loc[7, ["completion_tokens", "end_ms", "ttft_ms", "finish_reason"]] = [900, 999, 99, "tool_calls"]
        changed.at[7, "tool_calls"] = [{"category": "other"}]
        features = current_input_features(rows)
        assert_frame_equal(features, current_input_features(changed))
        self.assertEqual(features.loc[7, "tool_definition_count"], 2)
        self.assertEqual(features.loc[7, "role_tokens_tool"], 4)
        self.assertEqual(features.index.tolist(), [7])

    def test_missing_and_empty_inputs_are_distinct(self):
        rows = pd.DataFrame({"prompt_tokens": [1, 2], "cache_read_tokens": [0, 0],
                             "n_messages": [1, 1], "model": ["m", "m"],
                             "provider": ["p", "p"], "stream": [True, True],
                             "role_tokens": [None, {"user": 2}],
                             "tool_definition_names": [None, np.array([])]})
        result = current_input_features(rows)
        self.assertTrue(pd.isna(result.loc[0, "tool_definition_count"]))
        self.assertEqual(result.loc[1, "tool_definition_count"], 0)
        self.assertTrue(pd.isna(result.loc[0, "role_tokens_user"]))
        self.assertTrue(pd.isna(result.loc[1, "role_tokens_system"]))


@unittest.skipUnless(HAS_DEPS, "pandas and pyarrow are required")
class HistoryFeatureTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([{
            "session_id": "parent", "step": step, "start_ms": step * 100,
            "end_ms": step * 100 + 50, "completion_tokens": 10 * (step + 1),
            "prompt_tokens": 100 * (step + 1), "finish_reason": "stop",
            "link_reason": None if step == 0 else "tool_call_id",
            "step_trigger": "user" if step in (0, 3) else "tool",
            "waited_for": None if step == 0 else "tool", "transition": "append",
            "cache_read_tokens": 0, "n_messages": step + 1, "model": "m",
            "provider": "p", "stream": True, "role_tokens": {"user": 10},
            "tool_definition_names": [], "ttft_ms": 10,
        } for step in range(5)])

    def calls(self):
        return pd.DataFrame([
            {"session_id": "parent", "step": 0, "category": "Read", "outcome": "ok", "result_tokens": 4},
            {"session_id": "parent", "step": 0, "category": "Read", "outcome": "error", "result_tokens": 6},
            {"session_id": "parent", "step": 1, "category": "Edit", "outcome": "timeout", "result_tokens": 8},
        ])

    def build(self, rows=None, calls=None, sessions=None):
        from freeinference.features import build_features
        if rows is None:
            rows = self.rows()
        if calls is None:
            calls = self.calls()
        if sessions is None:
            sessions = pd.DataFrame({"session_id": ["parent"], "harness": ["h"],
                                     "parent_session_id": [None]})
        return build_features(rows, sessions, calls)

    def test_first_and_incomplete_or_equal_predecessor_have_no_history(self):
        from freeinference.features import A_FEATURES
        for end in (100, 101, None):
            rows = self.rows()
            rows.loc[0, "end_ms"] = end
            with self.subTest(end=end):
                result = self.build(rows)
                history = result["B"].drop(columns=list(A_FEATURES))
                self.assertTrue(history.loc[[0, 1]].isna().all().all())
                self.assertTrue(pd.isna(result["L"].loc[1, "completion_lag_1"]))
                # Later valid rows can still use older strictly completed results.
                self.assertEqual(result["L"].loc[2, "completion_lag_1"], 20)

    def test_overlapping_earlier_step_and_child_session_are_excluded(self):
        rows = self.rows()
        # This older step is still running; the immediate predecessor is done.
        rows.loc[0, "end_ms"] = 999
        child = rows.iloc[[0]].copy()
        child.loc[:, "session_id"] = "child"
        child.loc[:, "end_ms"] = 199
        child.loc[:, "completion_tokens"] = 9999
        rows = pd.concat([rows, child], ignore_index=True)
        sessions = pd.DataFrame({"session_id": ["parent", "child"], "harness": ["h", "h"],
                                 "parent_session_id": [None, "parent"]})
        b = self.build(rows, sessions=sessions)["B"]
        self.assertEqual(b.loc[2, "completion_lag_1"], 20)
        self.assertTrue(pd.isna(b.loc[2, "completion_lag_2"]))
        self.assertEqual(b.loc[2, "completion_ewma"], 20)
        self.assertEqual(b.loc[2, "previous_tool_result_tokens"], 8)
        self.assertTrue(b.loc[5].drop(labels=list(self.build()["A"].columns)).isna().all())

    def test_latest_completed_is_ordered_by_end_and_boundary_is_strict(self):
        rows = self.rows()
        rows.loc[0, "end_ms"] = 180
        b = self.build(rows)["B"]
        self.assertEqual(b.loc[2, "completion_lag_1"], 10)
        self.assertEqual(b.loc[2, "completion_lag_2"], 20)
        self.assertEqual(b.loc[2, "previous_end_gap_ms"], 20)
        rows.loc[0, "end_ms"] = 200
        b = self.build(rows)["B"]
        self.assertEqual(b.loc[2, "completion_lag_1"], 20)
        self.assertTrue(pd.isna(b.loc[2, "completion_lag_2"]))

    def test_current_and_future_responses_and_tools_do_not_change_target_row(self):
        rows, calls = self.rows(), self.calls()
        original = self.build(rows, calls)
        rows.loc[2:, ["completion_tokens", "end_ms", "ttft_ms", "finish_reason"]] = [9999, 9999, 9999, "changed"]
        calls = pd.concat([calls, pd.DataFrame([{"session_id": "parent", "step": step,
                    "category": "Subagent", "outcome": "error", "result_tokens": 9999}
                    for step in (2, 3, 4)])], ignore_index=True)
        changed = self.build(rows, calls)
        for condition in ("A", "L", "B"):
            with self.subTest(condition=condition):
                assert_frame_equal(original[condition].iloc[[2]], changed[condition].iloc[[2]])

    def test_lags_window_differences_turns_and_tools(self):
        result = self.build()
        b = result["B"]
        self.assertEqual(b.loc[3, "completion_lag_1"], 30)
        self.assertEqual(b.loc[3, "completion_lag_3"], 10)
        self.assertEqual(b.loc[3, "prompt_lag_3"], 100)
        self.assertEqual(b.loc[3, "completion_ewma"], 22.5)
        self.assertEqual(b.loc[3, "prompt_ewma"], 225)
        self.assertEqual(b.loc[3, "completion_diff"], 10)
        self.assertEqual(b.loc[3, "prompt_diff"], 100)
        self.assertEqual(b.loc[4, "completion_ewma"], 32.5)
        self.assertEqual(b.loc[3, "turn_index"], 1)
        self.assertEqual(b.loc[3, "turn_step"], 0)
        self.assertEqual(b.loc[4, "turn_step"], 1)
        self.assertEqual(b.loc[1, "previous_tool_call_count"], 2)
        self.assertEqual(b.loc[1, "previous_tool_category_Read"], 2)
        self.assertEqual(b.loc[1, "previous_tool_outcome_error"], 1)
        self.assertEqual(b.loc[1, "previous_tool_result_tokens"], 10)
        self.assertEqual(b.loc[3, "previous_tool_call_count"], 0)
        self.assertEqual(b.loc[3, "previous_tool_result_tokens"], 0)
        self.assertEqual(b.loc[3, "session_elapsed_ms"], 300)
        self.assertEqual(b.loc[3, "previous_end_gap_ms"], 50)
        self.assertEqual(b.loc[3, "previous_finish_reason"], "stop")
        assert_frame_equal(result["A"], result["L"].drop(columns="completion_lag_1"))
        assert_frame_equal(result["L"], b.loc[:, result["L"].columns])
        self.assertEqual(result["A"].loc[1, "harness"], "h")
        self.assertEqual(result["A"].loc[1, "step_trigger"], "tool")
        self.assertEqual(result["A"].loc[1, "waited_for"], "tool")
        self.assertEqual(result["A"].loc[1, "transition"], "append")

    def test_missing_unknown_link_and_missing_step_disable_history(self):
        from freeinference.features import A_FEATURES
        for reason in (None, pd.NA, "unverified"):
            rows = self.rows()
            rows.loc[2, "link_reason"] = reason
            b = self.build(rows)["B"]
            self.assertTrue(b.loc[2].drop(labels=list(A_FEATURES)).isna().all())
        rows = self.rows().drop(index=1)
        calls = self.calls().query("step != 1")
        b = self.build(rows, calls)["B"]
        self.assertTrue(b.loc[2].drop(labels=list(A_FEATURES)).isna().all())

    def test_missing_tool_results_and_unknown_vocabulary(self):
        calls = self.calls()
        calls.loc[0, ["category", "outcome", "result_tokens"]] = ["unknown", None, None]
        b = self.build(calls=calls)["B"]
        self.assertTrue(pd.isna(b.loc[1, "previous_tool_result_tokens"]))
        self.assertEqual(b.loc[1, "previous_tool_category_other/missing"], 1)
        self.assertEqual(b.loc[1, "previous_tool_outcome_missing"], 1)

    def test_reordering_duplicate_index_and_release_weeks_are_isolated(self):
        rows = self.rows()
        expected = self.build(rows)["B"]
        order = [3, 1, 4, 0, 2]
        shuffled = rows.iloc[order].copy()
        shuffled.index = [7, 7, 2, 9, 1]
        actual = self.build(shuffled)["B"]
        target = expected.iloc[order].copy()
        target.index = shuffled.index
        assert_frame_equal(actual, target)
        from freeinference.features import build_features
        sessions = pd.DataFrame({"session_id": ["parent"], "harness": ["h"]})
        copies = []
        for week in (1, 2):
            r, s, t = rows.copy(), sessions.copy(), self.calls()
            for frame in (r, s, t):
                frame["week"] = week
            if week == 2:
                r["completion_tokens"] *= 10
            copies.append((r, s, t))
        r, s, t = (pd.concat([item[i] for item in copies], ignore_index=True) for i in range(3))
        b = build_features(r, s, t)["B"]
        self.assertEqual(b.loc[2, "completion_lag_1"], 20)
        self.assertEqual(b.loc[7, "completion_lag_1"], 200)

    def test_invalid_keys_fail_closed(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            self.build(pd.concat([rows, rows.iloc[[0]]]))
        with self.assertRaises(ValueError):
            self.build(sessions=pd.DataFrame({"session_id": ["other"], "harness": ["h"]}))
        calls = self.calls()
        calls.loc[0, "step"] = 999
        with self.assertRaises(ValueError):
            self.build(calls=calls)

    def test_unfinished_predecessor_does_not_fall_back_to_older_results(self):
        from freeinference.features import A_FEATURES
        rows = self.rows()
        rows.loc[1, "end_ms"] = 250
        b = self.build(rows)["B"]
        self.assertTrue(b.loc[2].drop(labels=list(A_FEATURES)).isna().all())

    def test_completed_future_steps_do_not_enter_earlier_step_history(self):
        rows = self.rows()
        rows.loc[4, ["start_ms", "end_ms", "completion_tokens"]] = [75, 90, 9999]
        b = self.build(rows)["B"]
        self.assertEqual(b.loc[1, "completion_lag_1"], 10)
        self.assertTrue(pd.isna(b.loc[1, "completion_lag_2"]))
        self.assertEqual(b.loc[1, "turn_index"], 0)

    def test_tool_results_wait_for_successor_input_and_latency_is_ignored(self):
        from freeinference.features import TOOL_FEATURES
        rows = self.rows()
        rows.loc[0, "end_ms"] = 180
        rows.loc[1, ["start_ms", "end_ms"]] = [500, 550]
        rows.loc[2, ["start_ms", "end_ms"]] = [100, 150]
        rows.loc[3, ["start_ms", "end_ms"]] = [200, 250]
        b = self.build(rows)["B"]
        self.assertEqual(b.loc[3, "completion_lag_1"], 10)
        self.assertTrue(b.loc[3, list(TOOL_FEATURES)].isna().all())
        calls = self.calls()
        calls["latency_ms"] = 99999999
        assert_frame_equal(self.build()["B"], self.build(calls=calls)["B"])

    def test_missing_turn_trigger_is_missing_position(self):
        rows = self.rows()
        rows.loc[:2, "step_trigger"] = pd.NA
        b = self.build(rows)["B"]
        self.assertTrue(pd.isna(b.loc[2, "turn_index"]))
        self.assertTrue(pd.isna(b.loc[2, "turn_step"]))
        self.assertEqual(b.loc[2, "completion_lag_1"], 20)

    def test_generation_refuses_test_and_incomplete_weeks_before_writing(self):
        from freeinference.features import run
        from freeinference.loader import REVISION
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "REVISION").write_text(REVISION, encoding="utf-8")
            (root / "release").mkdir()
            with self.assertRaisesRegex(ValueError, "1–13"):
                run(root, root / "output.json")
            (root / "release" / "2026-08-16_2026-08-22").mkdir()
            with self.assertRaisesRegex(ValueError, "Held-out"):
                run(root, root / "output.json")
            self.assertFalse((root / "output.json").exists())


if __name__ == "__main__":
    unittest.main()
