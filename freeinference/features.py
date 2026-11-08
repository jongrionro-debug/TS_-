"""Prediction-time A/L/B features from the pinned processed tables.

No identifiers or targets are returned. Call once per release week, or supply
``week`` on all three tables to keep reconstructed sessions isolated by week.
"""
import bisect
import json
import platform
import sys
from datetime import timedelta
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
FEATURE_SETS = {"A": A_FEATURES, "L": L_FEATURES, "B": (*L_FEATURES, *B_EXTRA)}
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
            if (step == 0 or row["link_reason"] not in LINK_REASONS or predecessor is None
                    or pd.isna(records[predecessor]["end_ms"])
                    or records[predecessor]["end_ms"] >= start):
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
    return {condition: b.loc[:, list(columns)].copy() for condition, columns in FEATURE_SETS.items()}


def run(root=None, output=OUTPUT):
    """Generate aggregate timing/missingness only; never export request rows."""
    paths = weeks(root)
    if len(paths) != 13 or any(p.name[:10] != (FIRST_WEEK + timedelta(days=7*i)).isoformat()
                               for i, p in enumerate(paths)):
        raise ValueError("Features require consecutive train/validation weeks 1–13")
    started = perf_counter()
    weekly, total_missing = [], {name: 0 for name in FEATURE_SETS["B"]}
    for week in paths:
        tick = perf_counter()
        requests = load_table(week, "requests", REQUEST_COLUMNS)
        sessions = load_table(week, "sessions", ["session_id", "harness"])
        calls = load_table(week, "tool_calls", ["session_id", "step", "category", "outcome", "result_tokens"])
        loaded = perf_counter()
        features = build_features(requests, sessions, calls)
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
    count = sum(week["rows"] for week in weekly)
    result = {"revision": REVISION, "scope": "weeks 1–13 only; no split, training or evaluation",
              "command": f"{sys.executable} -m freeinference.features",
              "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                              "platform": platform.platform()},
              "feature_sets": FEATURE_SETS, "lag_count": LAG_COUNT, "ewma_alpha": EWMA_ALPHA,
              "ewma_window": "up to 3 completed requests, end order, adjust=False",
              "rows": count, "elapsed_seconds": perf_counter() - started,
              "generation_seconds": sum(week["generation_seconds"] for week in weekly),
              "missing_count": total_missing,
              "missing_rate": {name: value / count for name, value in total_missing.items()},
              "weeks": weekly}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run(data_dir())
