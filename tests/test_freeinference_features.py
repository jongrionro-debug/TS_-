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


@unittest.skipUnless(HAS_DEPS, "pandas and pyarrow are required")
class CrossSessionFeatureTests(unittest.TestCase):
    def fixture(self):
        from freeinference.features import OBSERVATION_START_MS
        base = OBSERVATION_START_MS
        # Query t=900000; lower 10m boundary=300000. Session IDs deliberately
        # repeat across weeks. The current session's earlier result is excluded
        # from account history, but remains a gateway observation.
        records = [
            (1, "old", 0, 100, 200, 10),
            (2, "current", 0, 300000, 300000, 20),
            (2, "child", 0, 300000, 400000, 30),
            (2, "child", 1, 500000, 700000, 40),
            (2, "other", 0, 600000, 800000, 50),
            (2, "equal", 0, 650000, 900000, 60),
            (2, "running", 0, 700000, 1000000, 70),
            (2, "unknown", 0, 800000, None, 80),
            (2, "current", 1, 900000, 900100, 90),
            (2, "at-start", 0, 900000, 900000, 100),
            (2, "future", 0, 1000000, 1100000, 110),
            (2, "old", 0, 200000, 250000, 120),
        ]
        rows = pd.DataFrame(records, columns=["week", "session_id", "step", "start_ms", "end_ms", "completion_tokens"])
        rows.start_ms += base
        rows.end_ms += base
        sessions = rows[["week", "session_id"]].drop_duplicates().assign(
            user_id="account", parent_session_id=None, harness="h")
        sessions.loc[sessions.session_id.eq("other"), "user_id"] = "other-account"
        sessions.loc[sessions.session_id.eq("child"), "parent_session_id"] = "current"
        return rows, sessions

    def build(self, rows, sessions):
        from freeinference.features import build_cross_session_features
        return build_cross_session_features(rows, sessions)

    def test_hand_computed_history_gateway_and_boundaries(self):
        rows, sessions = self.fixture()
        result = self.build(rows, sessions)
        query = result.iloc[8]
        expected = {"account_prev_completed_requests": 4, "account_prev_sessions": 3,
                    "account_completion_mean": 50, "account_completion_last": 40,
                    "account_last_end_gap_ms": 200000, "gateway_inflight_requests": 2,
                    "gateway_recent_sessions_10m": 6, "gateway_completion_10m": 140,
                    "gateway_completed_requests_10m": 4}
        self.assertEqual(query.to_dict(), expected)
        first = result.iloc[0]
        self.assertEqual(first.account_prev_completed_requests, 0)
        self.assertEqual(first.account_prev_sessions, 0)
        self.assertTrue(first[["account_completion_mean", "account_completion_last",
                               "account_last_end_gap_ms"]].isna().all())
        self.assertTrue(first[["gateway_recent_sessions_10m", "gateway_completion_10m",
                               "gateway_completed_requests_10m"]].isna().all())
        self.assertEqual(first.gateway_inflight_requests, 0)
        diagnostics = result.attrs["diagnostics"]["overall"]
        self.assertEqual(diagnostics["missing_end_requests"], 1)
        self.assertEqual(diagnostics["inflight_excluded_unknown_end_max"], 1)
        self.assertGreater(diagnostics["linked_parent_child_fraction"], 0)

    def test_current_uncompleted_equal_end_and_future_outcomes_cannot_leak(self):
        rows, sessions = self.fixture()
        expected = self.build(rows, sessions).iloc[[8]]
        changed = rows.copy()
        base = rows.loc[8, "start_ms"]
        changed.loc[[5, 6, 7, 8, 9, 10], "completion_tokens"] = 999999
        changed.loc[8, "end_ms"] = base + 99999
        changed.loc[10, ["start_ms", "end_ms"]] = [base + 200000, base + 300000]
        assert_frame_equal(expected, self.build(changed, sessions).iloc[[8]])
        # Altering a running end while it remains running cannot affect features.
        changed.loc[6, "end_ms"] = base + 200000
        assert_frame_equal(expected, self.build(changed, sessions).iloc[[8]])
        # No response at an exactly equal start/end is a completed observation.
        changed.loc[9, "end_ms"] = base + 10
        assert_frame_equal(expected, self.build(changed, sessions).iloc[[8]])

    def test_account_excludes_current_session_and_other_accounts_but_crosses_weeks(self):
        rows, sessions = self.fixture()
        expected = self.build(rows, sessions).iloc[[8], :5]
        changed = rows.copy()
        changed.loc[[1, 4], "completion_tokens"] = 999999
        assert_frame_equal(expected, self.build(changed, sessions).iloc[[8], :5])
        # Previous public-week session (same ID as a separate session in week 2)
        # is a distinct, valid historical session.
        changed = rows.copy()
        changed.loc[0, "completion_tokens"] = 50
        query = self.build(changed, sessions).iloc[8]
        self.assertEqual(query.account_completion_mean, 60)
        self.assertEqual(query.account_prev_sessions, 3)

    def test_order_index_and_ALB_are_preserved(self):
        from freeinference.features import FEATURE_SETS, build_features
        rows, sessions = self.fixture()
        expected = self.build(rows, sessions)
        order = [8, 11, 3, 0, 10, 2, 9, 1, 4, 5, 7, 6]
        shuffled = rows.iloc[order].copy()
        shuffled.index = [7, 7, 2, 1, 9, 5, 8, 3, 2, 0, 6, 4]
        target = expected.iloc[order].copy()
        target.index = shuffled.index
        assert_frame_equal(self.build(shuffled, sessions), target)
        # C is assembled independently; the existing API and column/value
        # contracts for A/L/B must remain unchanged.
        old_rows = HistoryFeatureTests().rows().assign(week=1)
        old_sessions = pd.DataFrame({"week": [1], "session_id": ["parent"], "harness": ["h"],
                                     "user_id": ["account"], "parent_session_id": [None]})
        calls = HistoryFeatureTests().calls().assign(week=1)
        before = build_features(old_rows, old_sessions, calls)
        cross = self.build(old_rows, old_sessions)
        combined = pd.concat([before["B"], cross], axis=1)
        for condition in ("A", "L", "B"):
            assert_frame_equal(before[condition], combined.loc[:, list(FEATURE_SETS[condition])])
        self.assertEqual(tuple(combined), FEATURE_SETS["C"])

    def test_randomized_matches_slow_online_oracle_and_end_ties(self):
        from freeinference.features import C_EXTRA, OBSERVATION_START_MS
        rng = np.random.default_rng(42)
        for seed in range(3):
            n = 140
            starts = OBSERVATION_START_MS + rng.integers(0, 1800000, n) // 10000 * 10000
            ends = starts + rng.integers(0, 150000, n) // 10000 * 10000
            rows = pd.DataFrame({"week": rng.integers(1, 3, n), "session_id": rng.integers(0, 7, n).astype(str),
                                 "step": np.arange(n), "start_ms": starts, "end_ms": ends.astype(float),
                                 "completion_tokens": rng.integers(1, 100, n).astype(float)})
            rows.loc[rng.choice(n, 8, replace=False), "end_ms"] = np.nan
            sessions = rows[["week", "session_id"]].drop_duplicates()
            sessions["user_id"] = sessions.session_id.astype(int) % 3
            sessions["parent_session_id"] = None
            merged = rows.merge(sessions, on=["week", "session_id"], sort=False)
            expected = []
            for row in merged.itertuples():
                past = merged.loc[(merged.end_ms < row.start_ms) & merged.user_id.eq(row.user_id)
                                  & ~(merged.week.eq(row.week) & merged.session_id.eq(row.session_id))]
                past = past.sort_values(["end_ms", "week", "session_id", "step"])
                completed = merged.loc[(merged.end_ms >= row.start_ms - 600000) & (merged.end_ms < row.start_ms)]
                recent = merged.loc[(merged.start_ms >= row.start_ms - 600000) & (merged.start_ms < row.start_ms)]
                window = [len(recent[["week", "session_id"]].drop_duplicates()), completed.completion_tokens.sum(), len(completed)]
                if row.start_ms < OBSERVATION_START_MS + 600000:
                    window = [np.nan] * 3
                expected.append([len(past), len(past[["week", "session_id"]].drop_duplicates()),
                                 past.completion_tokens.mean(), past.completion_tokens.iloc[-1] if len(past) else np.nan,
                                 row.start_ms - past.end_ms.iloc[-1] if len(past) else np.nan,
                                 ((merged.start_ms < row.start_ms) & (merged.end_ms >= row.start_ms)).sum(), *window])
            expected = pd.DataFrame(expected, columns=C_EXTRA).astype(float)
            expected.gateway_inflight_requests = expected.gateway_inflight_requests.astype(int)
            assert_frame_equal(self.build(rows, sessions), expected)

    def test_diagnostics_parent_child_counts_and_complete_window_start(self):
        from freeinference.features import OBSERVATION_START_MS
        rows, sessions = self.fixture()
        # Only query + a completed parent and child: the parent and child are
        # included, while the query's own earlier request is excluded.
        rows = rows.iloc[[1, 2, 8]].copy()
        result = self.build(rows, sessions)
        diag = result.attrs["diagnostics"]["overall"]
        self.assertEqual(diag["account_history_request_exposures"], 1)
        self.assertEqual(diag["linked_parent_child_request_exposures"], 1)
        self.assertEqual(diag["linked_parent_child_fraction"], 1)
        rows.loc[8, "start_ms"] = OBSERVATION_START_MS + 600000
        query = self.build(rows, sessions).loc[8]
        self.assertEqual(query.gateway_recent_sessions_10m, 2)
        self.assertEqual(query.gateway_completion_10m, 50)

    def test_real_week_boundary_retains_history_and_gateway_window(self):
        from freeinference.features import OBSERVATION_START_MS
        boundary = OBSERVATION_START_MS + 7 * 24 * 60 * 60 * 1000
        rows = pd.DataFrame({"week": [1, 1, 2], "session_id": ["same", "parallel", "same"],
                             "step": [0, 0, 0], "start_ms": [boundary - 600000, boundary - 100, boundary],
                             "end_ms": [boundary - 50, boundary, boundary + 50],
                             "completion_tokens": [10., 20., 30.]})
        sessions = rows[["week", "session_id"]].assign(user_id="account", parent_session_id=None)
        query = self.build(rows, sessions).iloc[2]
        self.assertEqual(query.account_prev_completed_requests, 1)
        self.assertEqual(query.account_prev_sessions, 1)
        self.assertEqual(query.account_completion_last, 10)
        self.assertEqual(query.gateway_recent_sessions_10m, 2)
        self.assertEqual(query.gateway_completed_requests_10m, 1)
        self.assertEqual(query.gateway_completion_10m, 10)
        self.assertEqual(query.gateway_inflight_requests, 1)

    def test_missing_completion_does_not_become_partial_sum_or_reduce_request_count(self):
        rows, sessions = self.fixture()
        rows.loc[2, "completion_tokens"] = np.nan
        query = self.build(rows, sessions).iloc[8]
        self.assertEqual(query.account_prev_completed_requests, 4)
        self.assertTrue(pd.isna(query.account_completion_mean))
        self.assertEqual(query.account_completion_last, 40)
        self.assertEqual(query.gateway_completed_requests_10m, 4)
        self.assertTrue(pd.isna(query.gateway_completion_10m))
        # Missing output in the current session is excluded from account mean.
        rows.loc[2, "completion_tokens"] = 30
        rows.loc[1, "completion_tokens"] = np.nan
        self.assertEqual(self.build(rows, sessions).iloc[8].account_completion_mean, 50)

    def test_invalid_metadata_fails_closed_and_unknown_account_has_no_history(self):
        rows, sessions = self.fixture()
        with self.assertRaises(ValueError):
            self.build(rows, sessions.iloc[1:])
        with self.assertRaises(ValueError):
            self.build(rows, pd.concat([sessions, sessions.iloc[[0]]]))
        sessions.user_id = None
        result = self.build(rows, sessions)
        self.assertTrue(result.account_prev_completed_requests.eq(0).all())
        self.assertTrue(result.account_prev_sessions.eq(0).all())
        self.assertTrue(result.account_completion_mean.isna().all())


if __name__ == "__main__":
    unittest.main()
