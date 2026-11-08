"""Create descriptive train/validation EDA from the guarded processed tables."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .loader import FIRST_WEEK, REVISION, data_dir, load_table, weeks

OUTPUT = Path(__file__).parent / "outputs" / "eda"
BIN_MS = 600_000
CATEGORIES = {"Read", "Other", "*Network", "*Interpreter", "*File Ops", "*Search",
              "*Wait/Poll", "Edit", "Write", "Browser", "*Git/VCS", "*Build/Test",
              "*Packages", "MCP", "Ask User", "Subagent"}
FINISH = {"stop", "length", "tool_calls", "content_filter", "function_call"}


def aggregate_10min(requests, start_ms=None, end_ms=None):
    """Sum tokens into UTC [start, end) bins, filling gaps; optional bounds are exclusive at end."""
    required = ["start_ms", "prompt_tokens", "completion_tokens"]
    if requests[required].isna().any().any():
        raise ValueError("start and token counts must be non-null")
    if ((requests[["prompt_tokens", "completion_tokens"]] < 0).any().any()):
        raise ValueError("token counts must be non-negative")
    if requests.empty and (start_ms is None or end_ms is None):
        raise ValueError("empty requests require both bounds")
    first = int(requests.start_ms.min() // BIN_MS * BIN_MS) if start_ms is None else int(start_ms)
    end = int(requests.start_ms.max() // BIN_MS * BIN_MS + BIN_MS) if end_ms is None else int(end_ms)
    if first % BIN_MS or end % BIN_MS or end <= first:
        raise ValueError("bounds must be aligned UTC 10-minute boundaries")
    if not requests.empty and ((requests.start_ms < first) | (requests.start_ms >= end)).any():
        raise ValueError("requests outside aggregation bounds")
    bins = (requests.start_ms // BIN_MS) * BIN_MS
    sums = requests.groupby(bins)[["prompt_tokens", "completion_tokens"]].sum()
    sums = sums.reindex(pd.Index(range(first, end, BIN_MS), name="start_ms"), fill_value=0)
    sums.index = pd.to_datetime(sums.index, unit="ms", utc=True)
    return sums


def adjacent_pairs(rows):
    rows = rows.sort_values(["week", "session_id", "step"], kind="stable").copy()
    groups = rows.groupby(["week", "session_id"], sort=False)
    rows["next_step"] = groups.step.shift(-1)
    rows["next_start"] = groups.start_ms.shift(-1)
    rows["next_completion"] = groups.completion_tokens.shift(-1)
    return rows[(rows.next_step == rows.step + 1) & (rows.end_ms < rows.next_start)].copy()


def active_sessions(rows):
    """Number of session spans containing each request start, including its own session."""
    spans = rows.groupby(["week", "session_id"]).agg(start=("start_ms", "min"), end=("end_ms", "max"))
    if spans.isna().any().any() or (spans.end < spans.start).any():
        raise ValueError("invalid session time span")
    starts = np.sort(spans.start.to_numpy())
    ends = np.sort(spans.end.to_numpy())
    # Closed spans ensure a zero-duration session is active at its own start.
    times = rows.start_ms.to_numpy()
    return np.searchsorted(starts, times, side="right") - np.searchsorted(ends, times, side="left")


def distribution_table(frame, group_columns, value):
    grouped = frame.groupby(group_columns, dropna=False)[value]
    return grouped.agg(n="count", mean="mean", p10=lambda x: x.quantile(.1),
                       p50="median", p90=lambda x: x.quantile(.9)).reset_index()


def markdown_table(frame):
    def fmt(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{v:.2f}"
        return str(v).replace("|", "\\|")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    return "\n".join([header, divider] + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def diagram(path):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(.5, .94, "Requests: start (UTC), prompt, completion", ha="center", fontsize=13, weight="bold")
    for x, label in ((.16, "00:02  (10, 4)"), (.40, "00:09  (20, 6)"), (.76, "00:10  (7, 3)")):
        ax.text(x, .68, label, ha="center", fontsize=12)
    for x, target in ((.16, .28), (.40, .28), (.76, .76)):
        ax.annotate("", xy=(target, .32), xytext=(x, .60), arrowprops={"arrowstyle": "->", "lw": 1.3})
    ax.text(.28, .15, "[00:00, 00:10): (30, 10)", ha="center", fontsize=12)
    ax.text(.76, .15, "[00:10, 00:20): (7, 3)", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(series, path):
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True, constrained_layout=True)
    boundary = pd.Timestamp("2026-07-26", tz="UTC")
    for ax, (column, title) in zip(axes, [("prompt_tokens", "Input (prompt)"), ("completion_tokens", "Output (completion)")]):
        ax.plot(series.index, series[column], lw=.45, color="#235789")
        ax.axvspan(boundary, series.index[-1] + pd.Timedelta(minutes=10), color="#f0c987", alpha=.35)
        ax.axvline(boundary, color="#9b5d00", lw=1)
        ax.set_title(title + " tokens per 10 minutes")
        ax.set_ylabel("Tokens")
        ax.grid(alpha=.2)
    axes[0].text(.02, .92, "Train: weeks 1–10", transform=axes[0].transAxes)
    axes[0].text(.78, .92, "Validation: weeks 11–13", transform=axes[0].transAxes)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))
    axes[-1].set_xlabel("2026, UTC")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root=None, output=OUTPUT):
    paths = weeks(root)
    if len(paths) != 13 or any(p.name[:10] != (FIRST_WEEK + timedelta(days=7*i)).isoformat() for i, p in enumerate(paths)):
        raise ValueError("EDA requires consecutive train/validation weeks 1–13")
    frames, tools = [], []
    for i, week in enumerate(paths):
        rows = load_table(week, "requests", ["session_id", "step", "start_ms", "end_ms", "prompt_tokens", "completion_tokens", "finish_reason"])
        rows["week"] = i + 1
        rows["split"] = "train" if i < 10 else "validation"
        calls = load_table(week, "tool_calls", ["session_id", "step", "category"])
        calls["week"] = i + 1
        frames.append(rows)
        tools.append(calls)
    rows = pd.concat(frames, ignore_index=True)
    calls = pd.concat(tools, ignore_index=True)
    if rows[["session_id", "step", "start_ms", "end_ms", "prompt_tokens", "completion_tokens"]].isna().any().any() or rows.duplicated(["week", "session_id", "step"]).any():
        raise ValueError("requests require unique session steps and complete time/token values")
    start = int(datetime(2026, 5, 17, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(datetime(2026, 8, 16, tzinfo=timezone.utc).timestamp() * 1000)
    series = aggregate_10min(rows, start, end)
    pairs = adjacent_pairs(rows)
    pairs["finish"] = pairs.finish_reason.where(pairs.finish_reason.isin(FINISH), "other/missing")
    calls["category"] = calls.category.where(calls.category.isin(CATEGORIES), "other/missing")
    calls = calls.drop_duplicates(["week", "session_id", "step", "category"])
    tool_pairs = pairs.merge(calls, on=["week", "session_id", "step"], how="left")
    tool_pairs["category"] = tool_pairs.category.fillna("no tool")
    rows["active"] = active_sessions(rows)
    rows["active_group"] = pd.cut(rows.active, bins=[0, 1, 5, 10, 25, 50, 100, np.inf],
                                  labels=["1", "2–5", "6–10", "11–25", "26–50", "51–100", "101+"]).astype(str)
    correlations = []
    for split in ("train", "validation"):
        p = pairs[pairs.split == split]
        correlations.append({"split": split, "pairs": len(p), "Pearson r": p.completion_tokens.corr(p.next_completion)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    diagram(output / "aggregation.png")
    plot_timeseries(series, output / "timeseries.png")
    tables = [
        ("Adjacent step completion correlation", pd.DataFrame(correlations)),
        ("Previous finish reason → next completion", distribution_table(pairs, ["split", "finish"], "next_completion")),
        ("Previous step tool category → next completion", distribution_table(tool_pairs, ["split", "category"], "next_completion")),
        ("Active sessions at request start → completion", distribution_table(rows, ["split", "active_group"], "completion_tokens")),
        ("Weekly completion distribution", distribution_table(rows, ["split", "week"], "completion_tokens")),
    ]
    text = ["# FreeInference EDA: weeks 1–13", "", f"- Run: `{shlex.quote(sys.executable)} -m freeinference.eda` with `FREEINFERENCE_DATA_DIR` set to the prepared data.",
            f"- Revision: `{REVISION}`. Train: weeks 1–10; validation: weeks 11–13. Test is refused by the loader.",
            f"- Requests: {len(rows):,}; strictly completed adjacent step pairs: {len(pairs):,}. Descriptive associations are not prediction results.", "",
            "## Request aggregation", "", "UTC 10-minute intervals are left-closed and right-open. Both token columns are summed and empty bins are zero-filled; the full 13-week interval is shown. Zero bins indicate no recorded requests, not evidence of continuous service availability.", "",
            "![Request aggregation](aggregation.png)", "", "![Train and validation time series](timeseries.png)", "",
            "## Definitions", "", "Adjacent means consecutive numeric steps within the same released week and session, with previous end strictly before next start. Correlation is Pearson's r over request pairs. Finish reason and tool category belong to the previous step; each distinct category contributes once per pair, so multi-category pairs appear in multiple rows. `no tool` means no matching tool call.", "",
            "A session is active from its first request start through its last request end, both inclusive; it includes gaps and the current session. Concurrent counts include spans across released weeks. This retrospective span uses the eventual last end and is not an online feature. Token distributions are request-weighted; n counts non-null completions and p10/p50/p90 are empirical quantiles. Categories outside the fixed public vocabulary are grouped to avoid exposing identifiers.", ""]
    for title, table in tables:
        text += [f"## {title}", "", markdown_table(table), ""]
    (output / "eda.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Wrote EDA to {output}")


if __name__ == "__main__":
    run(data_dir())
