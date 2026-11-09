"""Prediction-time A/L/B/C features from the pinned processed tables.

No identifiers or targets are returned. Call once per release week, or supply
``week`` on all three tables to keep reconstructed sessions isolated by week.
"""
import bisect
import hashlib
import json
import shlex
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .loader import FIRST_WEEK, REVISION, data_dir, load_table, weeks

REQUEST_INPUTS = (
    "prompt_tokens", "cache_read_tokens", "n_messages", "model", "provider", "stream",
    "harness", "step_trigger", "waited_for", "transition",
)
ROLE_NAMES = ("system", "user", "assistant", "tool", "tool_definitions")
LINK_REASONS = {"tool_call_id", "assistant_message"}
LAG_COUNT = 3
EWMA_ALPHA = .5
TOOL_CATEGORIES = (
    "Read", "Other", "*Network", "*Interpreter", "*File Ops", "*Search",
    "*Wait/Poll", "Edit", "Write", "Browser", "*Git/VCS", "*Build/Test",
    "*Packages", "MCP", "Ask User", "Subagent",
)
TOOL_OUTCOMES = ("ok", "error", "timeout", "denied")
TOOL_FEATURES = (
    "previous_tool_call_count", "previous_tool_result_tokens",
    *(f"previous_tool_category_{category}" for category in (*TOOL_CATEGORIES, "other/missing")),
    *(f"previous_tool_outcome_{outcome}" for outcome in (*TOOL_OUTCOMES, "other", "missing")),
)
A_FEATURES = (*REQUEST_INPUTS, *(f"role_tokens_{role}" for role in ROLE_NAMES), "tool_definition_count")
L_FEATURES = (*A_FEATURES, "completion_lag_1")
B_EXTRA = (
    "step", "turn_index", "turn_step",
    *(f"{kind}_lag_{lag}" for kind in ("completion", "prompt") for lag in range(1, LAG_COUNT + 1)
      if (kind, lag) != ("completion", 1)),
    "completion_ewma", "prompt_ewma", "completion_diff", "prompt_diff",
    "previous_finish_reason", *TOOL_FEATURES, "session_elapsed_ms", "previous_end_gap_ms",
)
C_EXTRA = (
    "account_prev_completed_requests", "account_prev_sessions", "account_completion_mean",
    "account_completion_last", "account_last_end_gap_ms", "gateway_inflight_requests",
    "gateway_recent_sessions_10m", "gateway_completion_10m", "gateway_completed_requests_10m",
)
FEATURE_SETS = {"A": A_FEATURES, "L": L_FEATURES, "B": (*L_FEATURES, *B_EXTRA)}
FEATURE_SETS["C"] = (*FEATURE_SETS["B"], *C_EXTRA)
GATEWAY_WINDOW_MS = 10 * 60 * 1000
OBSERVATION_START_MS = int(datetime.combine(FIRST_WEEK, datetime.min.time(), timezone.utc).timestamp() * 1000)
SESSION_COLUMNS = ["session_id", "harness", "user_id", "parent_session_id"]
REQUEST_COLUMNS = list(dict.fromkeys([
    "session_id", "step", "start_ms", "end_ms", "link_reason", "completion_tokens",
    "finish_reason", "role_tokens", "tool_definition_names",
    *(name for name in REQUEST_INPUTS if name != "harness"),
]))
OUTPUT = Path(__file__).parent / "outputs" / "features" / "features.json"


def current_input_features(requests):
    """Return input-derived A features, preserving row order and index.

    Absent input fields, role breakdowns and tool-definition lists are missing;
    an explicitly empty list of definitions has count zero.
    """
    result = requests.reindex(columns=REQUEST_INPUTS).copy()
    for role in ROLE_NAMES:
        result[f"role_tokens_{role}"] = requests["role_tokens"].map(
            lambda value: value.get(role) if isinstance(value, dict) else None
        )
    result["tool_definition_count"] = requests["tool_definition_names"].map(
        lambda value: len(value) if isinstance(value, (list, tuple, np.ndarray)) else None
    )
    return result


def _tool_summaries(calls, keys):
    summaries = {}
    for key, frame in calls.groupby([*keys, "step"], sort=False):
        values = dict.fromkeys(TOOL_FEATURES, 0)
        values["previous_tool_call_count"] = len(frame)
        # A partial sum would turn unknown result sizes into apparent zeros.
        tokens = frame.result_tokens
        values["previous_tool_result_tokens"] = tokens.sum() if tokens.notna().all() else np.nan
        for category, count in frame.category.fillna("other/missing").value_counts().items():
            category = category if category in TOOL_CATEGORIES else "other/missing"
            values[f"previous_tool_category_{category}"] += int(count)
        for outcome, count in frame.outcome.fillna("missing").value_counts().items():
            outcome = outcome if outcome in TOOL_OUTCOMES or outcome == "missing" else "other"
            values[f"previous_tool_outcome_{outcome}"] += int(count)
        summaries[key] = values
    return summaries


def _window_ewma(values):
    """adjust=False EWMA over up to three completed requests, oldest first."""
    value = np.nan
    for item in reversed(values):
        if pd.notna(item):
            value = item if pd.isna(value) else EWMA_ALPHA * item + (1 - EWMA_ALPHA) * value
    return value


def history_allowed(row, predecessor):
    """Shared current-input link and strict predecessor-completion gate."""
    return (row["step"] > 0 and row["link_reason"] in LINK_REASONS
            and predecessor is not None and pd.notna(predecessor["end_ms"])
            and predecessor["end_ms"] < row["start_ms"])


def build_features(requests, sessions, tool_calls):
    """Return {A, L, B} DataFrames aligned positionally with requests.

    A current link must name an allowed input-based reason and have an existing
    step-1 with end < current start. Otherwise *all* history features are NA.
    History is earlier steps in the same week/session, strictly completed at
    the current start, ordered by (end_ms, step); a tie uses the larger step.
    Tool results are visible only once their successor input has arrived.
    """
    keys = ["week", "session_id"] if "week" in requests else ["session_id"]
    for frame, label, unique in ((requests, "requests", [*keys, "step"]),
                                 (sessions, "sessions", keys)):
        if frame[unique].isna().any().any() or frame.duplicated(unique).any():
            raise ValueError(f"{label} require unique non-null keys")
    if requests.start_ms.isna().any() or requests.step.lt(0).any() or (requests.step % 1 != 0).any():
        raise ValueError("requests require start_ms and non-negative integer steps")
    if (requests.end_ms < requests.start_ms).any():
        raise ValueError("request end must not precede start")
    if tool_calls[[*keys, "step"]].isna().any().any():
        raise ValueError("tool calls require non-null request keys")
    request_keys = pd.MultiIndex.from_frame(requests[[*keys, "step"]])
    if not pd.MultiIndex.from_frame(tool_calls[[*keys, "step"]]).isin(request_keys).all():
        raise ValueError("tool call without a matching request")
    rows = requests.drop(columns="harness", errors="ignore").merge(
        sessions[[*keys, "harness"]], on=keys, how="left", sort=False,
        validate="many_to_one", indicator=True,
    )
    if rows["_merge"].ne("both").any():
        raise ValueError("request without a matching session")
    rows = rows.drop(columns="_merge")
    a = current_input_features(rows)
    history_columns = ["completion_lag_1", *B_EXTRA]
    history = {name: [np.nan] * len(rows) for name in history_columns}
    # finish_reason is categorical, not a float array.
    history["previous_finish_reason"] = [None] * len(rows)
    tools = _tool_summaries(tool_calls, keys)
    no_tools = dict.fromkeys(TOOL_FEATURES, 0)
    records = rows.to_dict("records")
    for key, group in rows.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        by_step = {int(records[pos]["step"]): pos for pos in group.index}
        # A sweep by start makes completed outcomes available by end time,
        # independently of step order and independently of caller row order.
        ended = sorted((records[pos]["end_ms"], int(records[pos]["step"]), pos)
                       for pos in group.index if pd.notna(records[pos]["end_ms"]))
        available, user_steps = [], []
        cursor = 0
        first = by_step.get(0)
        for pos in group.sort_values(["start_ms", "step"], kind="stable").index:
            row = records[pos]
            start, step = row["start_ms"], int(row["step"])
            while cursor < len(ended) and ended[cursor][0] < start:
                available.append(ended[cursor])
                cursor += 1
            if pd.notna(row["step_trigger"]) and row["step_trigger"] == "user":
                bisect.insort(user_steps, step)
            predecessor = by_step.get(step - 1)
            if not history_allowed(row, records[predecessor] if predecessor is not None else None):
                continue
            recent = []
            for _, old_step, old_pos in reversed(available):
                if old_step < step:
                    recent.append(records[old_pos])
                    if len(recent) == LAG_COUNT:
                        break
            previous = recent[0]
            history["step"][pos] = step
            turn = bisect.bisect_right(user_steps, step)
            if turn:
                history["turn_index"][pos] = turn - 1
                history["turn_step"][pos] = step - user_steps[turn - 1]
            for kind in ("completion", "prompt"):
                values = [old[f"{kind}_tokens"] for old in recent]
                for lag, value in enumerate(values, 1):
                    history[f"{kind}_lag_{lag}"][pos] = value
                history[f"{kind}_ewma"][pos] = _window_ewma(values)
                if len(values) > 1:
                    history[f"{kind}_diff"][pos] = values[0] - values[1]
            history["previous_finish_reason"][pos] = previous["finish_reason"]
            if first is not None and records[first]["start_ms"] <= start:
                history["session_elapsed_ms"][pos] = start - records[first]["start_ms"]
            history["previous_end_gap_ms"][pos] = start - previous["end_ms"]
            successor = by_step.get(int(previous["step"]) + 1)
            if (successor is not None and records[successor]["start_ms"] <= start
                    and records[successor]["link_reason"] in LINK_REASONS):
                summary = tools.get((*key, int(previous["step"])), no_tools)
                for name, value in summary.items():
                    history[name][pos] = value
    b = pd.concat([a, pd.DataFrame(history)], axis=1)
    b.index = requests.index
    return {condition: b.loc[:, list(columns)].copy() for condition, columns in FEATURE_SETS.items() if condition != "C"}


def _prefix_queries(ends, values, source_groups, starts, query_groups):
    """Grouped cumulative sums/searchsorted, O((sources + queries) log n).

    Return completed count, token sum and missing-token count. Negative group
    codes represent an unknown account/parent and have no observable history.
    """
    result = np.zeros((len(starts), 3))
    sources = pd.DataFrame({"group": source_groups, "end": ends, "value": values})
    sources = sources.loc[sources.end.notna() & sources.group.ge(0)].sort_values(
        ["group", "end"], kind="stable")
    queries = pd.Series(np.arange(len(starts))).groupby(query_groups, sort=False)
    by_group = sources.groupby("group", sort=False).indices
    for group, positions in queries:
        if group not in by_group:
            continue
        old = sources.iloc[by_group[group]]
        positions = positions.to_numpy()
        count = np.searchsorted(old.end.to_numpy(), starts[positions], side="left")
        tokens = np.r_[0., old.value.fillna(0).to_numpy().cumsum()]
        missing = np.r_[0, old.value.isna().to_numpy().cumsum()]
        result[positions] = np.column_stack([count, tokens[count], missing[count]])
    return result


def build_cross_session_features(requests, sessions):
    """Return the nine C additions in caller row/index order, across weeks.

    Strictly completed outcomes only; current session excluded from account
    history. Equal-end ties use (week, session_id, step) lexicographic order.
    Only the three 10-minute window features are NA before a complete window
    since the pinned release start. Aggregate diagnostics live in ``attrs``;
    they are never model inputs and contain no identifiers or row values.
    """
    keys = ["week", "session_id"] if "week" in requests else ["session_id"]
    for frame, label, unique in ((requests, "requests", [*keys, "step"]),
                                 (sessions, "sessions", keys)):
        if frame[unique].isna().any().any() or frame.duplicated(unique).any():
            raise ValueError(f"{label} require unique non-null keys")
    if requests.start_ms.isna().any() or (requests.end_ms < requests.start_ms).any():
        raise ValueError("requests require start_ms and end >= start when known")
    metadata = sessions[[*keys, "user_id", "parent_session_id"]].copy()
    metadata["sid"] = np.arange(len(metadata))
    rows = requests[[*keys, "step", "start_ms", "end_ms", "completion_tokens"]].merge(
        metadata, on=keys, how="left", sort=False, validate="many_to_one", indicator=True)
    if rows._merge.ne("both").any():
        raise ValueError("request without a matching session")
    starts, ends, tokens = (rows[name].to_numpy(dtype=float) for name in
                            ("start_ms", "end_ms", "completion_tokens"))
    sid = rows.sid.to_numpy()
    accounts = pd.factorize(rows.user_id, sort=False)[0]
    own = _prefix_queries(ends, tokens, sid, starts, sid)
    account = _prefix_queries(ends, tokens, accounts, starts, accounts) - own
    # Unknown account IDs cannot link separate sessions (even to each other).
    account[accounts < 0] = 0
    result = pd.DataFrame(np.nan, index=np.arange(len(rows)), columns=C_EXTRA)
    result.account_prev_completed_requests = account[:, 0]
    result.account_completion_mean = np.divide(
        account[:, 1], account[:, 0], out=np.full(len(rows), np.nan),
        where=(account[:, 0] > 0) & (account[:, 2] == 0))
    result.account_prev_sessions = 0.
    completed = rows.loc[rows.end_ms.notna()].copy()
    completed["account"] = accounts[completed.index]
    completed = completed.sort_values(["end_ms", *keys, "step"], kind="stable")
    by_account = completed.groupby("account", sort=False)
    queries = pd.Series(np.arange(len(rows))).groupby(accounts, sort=False)
    for account_id, positions in queries:
        if account_id < 0 or account_id not in by_account.indices:
            continue
        old = completed.iloc[by_account.indices[account_id]]
        positions = positions.to_numpy()
        first_ends = np.sort(old.drop_duplicates("sid").end_ms.to_numpy())
        result.loc[positions, "account_prev_sessions"] = (
            np.searchsorted(first_ends, starts[positions], side="left") - (own[positions, 0] > 0))
        count = np.searchsorted(old.end_ms.to_numpy(), starts[positions], side="left")
        # A contiguous run of one session: skip the whole run when it is the
        # query session. Its predecessor is the most recent OTHER session.
        old_sid = old.sid.to_numpy()
        run_start = np.maximum.accumulate(np.where(
            np.r_[True, old_sid[1:] != old_sid[:-1]], np.arange(len(old)), 0))
        last = count - 1
        same = (count > 0) & (old_sid[np.maximum(last, 0)] == sid[positions])
        last[same] = run_start[last[same]] - 1
        valid = last >= 0
        target, source = positions[valid], last[valid]
        result.loc[target, "account_completion_last"] = old.completion_tokens.to_numpy()[source]
        result.loc[target, "account_last_end_gap_ms"] = starts[target] - old.end_ms.to_numpy()[source]

    known = np.isfinite(ends)
    result.gateway_inflight_requests = (
        np.searchsorted(np.sort(starts[known]), starts, side="left")
        - np.searchsorted(np.sort(ends[known]), starts, side="left"))
    # Each session contributes a union of (start, start+10m] intervals. Counting
    # these disjoint intervals gives distinct sessions without a per-row scan.
    ordered = rows.sort_values(["sid", "start_ms"], kind="stable")
    times = ordered.start_ms.to_numpy()
    ids = ordered.sid.to_numpy()
    new = np.r_[True, (ids[1:] != ids[:-1]) | (np.diff(times) > GATEWAY_WINDOW_MS)]
    first = times[new]
    last = times[np.r_[new[1:], True]] + GATEWAY_WINDOW_MS
    result.gateway_recent_sessions_10m = (
        np.searchsorted(np.sort(first), starts, side="left")
        - np.searchsorted(np.sort(last), starts, side="left"))
    order = np.argsort(ends[known], kind="stable")
    sorted_ends, sorted_tokens = ends[known][order], tokens[known][order]
    lo = np.searchsorted(sorted_ends, starts - GATEWAY_WINDOW_MS, side="left")
    hi = np.searchsorted(sorted_ends, starts, side="left")
    total = np.r_[0., np.nan_to_num(sorted_tokens, nan=0).cumsum()]
    missing = np.r_[0, np.isnan(sorted_tokens).cumsum()]
    result.gateway_completed_requests_10m = hi - lo
    result.gateway_completion_10m = np.where(missing[hi] == missing[lo], total[hi] - total[lo], np.nan)
    partial = starts - GATEWAY_WINDOW_MS < OBSERVATION_START_MS
    result.loc[partial, ["gateway_recent_sessions_10m", "gateway_completion_10m",
                         "gateway_completed_requests_10m"]] = np.nan

    # Direct parent/child edges are resolved within the public week. Only
    # same-account completed outcomes contribute; no final session ends used.
    parent_keys = metadata[keys].copy()
    parent_keys["session_id"] = metadata.parent_session_id
    lookup = pd.MultiIndex.from_frame(metadata[keys])
    parent = lookup.get_indexer(pd.MultiIndex.from_frame(parent_keys))
    metadata_accounts = pd.factorize(metadata.user_id, sort=False)[0]
    linked = (parent >= 0) & (parent != np.arange(len(parent)))
    linked[linked] &= ((metadata_accounts[linked] >= 0) & (metadata_accounts[linked] == metadata_accounts[parent[linked]]))
    parent[~linked] = -1
    row_parent = parent[sid]
    parent_history = _prefix_queries(ends, tokens, sid, starts, row_parent)[:, 0]
    child_history = _prefix_queries(ends, tokens, row_parent, starts, sid)[:, 0]
    linked_history = parent_history + child_history
    unknown = np.searchsorted(np.sort(starts[~known]), starts, side="left")

    def diagnostics(mask):
        denominator = float(account[mask, 0].sum())
        numerator = float(linked_history[mask].sum())
        return {"rows": int(mask.sum()), "missing_end_requests": int((~known & mask).sum()),
                "inflight_excluded_unknown_end_mean": float(unknown[mask].mean()),
                "inflight_excluded_unknown_end_max": int(unknown[mask].max()),
                "account_history_request_exposures": denominator,
                "linked_parent_child_request_exposures": numerator,
                "linked_parent_child_fraction": numerator / denominator if denominator else None}

    result.index = requests.index
    result.attrs["diagnostics"] = {
        "definition": "direct same-account parent/child within public week; exposures sum histories over query rows",
        "overall": diagnostics(np.ones(len(rows), dtype=bool)),
        "weeks": {str(week): diagnostics(rows.week.eq(week).to_numpy())
                  for week in pd.unique(rows.week)} if "week" in rows else {}}
    return result


def run(root=None, output=OUTPUT):
    """Generate aggregate timing/missingness only; never export request rows."""
    paths = weeks(root)
    if len(paths) != 13 or any(p.name[:10] != (FIRST_WEEK + timedelta(days=7*i)).isoformat()
                               for i, p in enumerate(paths)):
        raise ValueError("Features require consecutive train/validation weeks 1–13")
    started = perf_counter()
    weekly, cross_rows, cross_sessions = [], [], []
    total_missing = {name: 0 for name in FEATURE_SETS["C"]}
    for number, week in enumerate(paths, 1):
        tick = perf_counter()
        requests = load_table(week, "requests", REQUEST_COLUMNS)
        sessions = load_table(week, "sessions", SESSION_COLUMNS)
        calls = load_table(week, "tool_calls", ["session_id", "step", "category", "outcome", "result_tokens"])
        loaded = perf_counter()
        features = build_features(requests, sessions, calls)
        cross_rows.append(requests[["session_id", "step", "start_ms", "end_ms", "completion_tokens"]].assign(week=number))
        cross_sessions.append(sessions.assign(week=number))
        built = perf_counter()
        b = features["B"]
        missing = {name: int(value) for name, value in b.isna().sum().items()}
        for name, value in missing.items():
            total_missing[name] += value
        weekly.append({"week": week.name, "rows": len(b), "load_seconds": loaded - tick,
                       "generation_seconds": built - loaded,
                       "history_available_rows": int(b.completion_lag_1.notna().sum()),
                       "missing_count": missing,
                       "missing_rate": {name: value / len(b) for name, value in missing.items()}})
        print(f"{week.name}: {len(b):,} rows, generation {built - loaded:.3f}s", flush=True)
    tick = perf_counter()
    cross = build_cross_session_features(pd.concat(cross_rows, ignore_index=True),
                                         pd.concat(cross_sessions, ignore_index=True))
    cross_seconds = perf_counter() - tick
    offset = 0
    for week in weekly:
        missing = cross.iloc[offset:offset + week["rows"]].isna().sum()
        for name, value in missing.items():
            week["missing_count"][name] = int(value)
            total_missing[name] += int(value)
        week["missing_rate"] = {name: value / week["rows"] for name, value in week["missing_count"].items()}
        offset += week["rows"]
    count = sum(week["rows"] for week in weekly)
    result = {"revision": REVISION, "scope": "weeks 1–13 only; no split, training or evaluation",
              "command": 'FREEINFERENCE_DATA_DIR="$FREEINFERENCE_DATA_DIR" OPENBLAS_NUM_THREADS=1 '
                         f"{shlex.quote(sys.executable)} -m freeinference.features",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cross_session": {"window_ms": GATEWAY_WINDOW_MS,
                                "observation_start_ms": OBSERVATION_START_MS,
                                "generation_seconds": cross_seconds,
                                "diagnostics": cross.attrs["diagnostics"]},
              "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                              "platform": platform.platform()},
              "feature_sets": FEATURE_SETS, "lag_count": LAG_COUNT, "ewma_alpha": EWMA_ALPHA,
              "ewma_window": "up to 3 completed requests, end order, adjust=False",
              "rows": count, "elapsed_seconds": perf_counter() - started,
              "generation_seconds": sum(week["generation_seconds"] for week in weekly) + cross_seconds,
              "missing_count": total_missing,
              "missing_rate": {name: value / count for name, value in total_missing.items()},
              "weeks": weekly}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run(data_dir())
