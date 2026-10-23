"""Build review figures from existing EDA tables; leave the proposal PDF unchanged."""

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ts-proposal-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
TABLES = ROOT / "eda/results/tables"
TARGETS = ["next10_input_tokens", "next10_final_output_tokens"]
COLORS = ["#267485", "#bd6547"]

minute = pd.read_csv(TABLES / "04_series_1min.csv")
target = pd.read_csv(TABLES / "08_active_sessions_and_demand.csv")
coverage = pd.read_csv(TABLES / "03_coverage.csv").iloc[0]
followup = pd.read_csv(TABLES / "06_followup_gap_thresholds.csv")
minute_times = minute.timestamp.to_numpy()
times = target.timestamp.to_numpy()
positions = np.searchsorted(minute_times, times)
assert (np.diff(minute_times) == 60).all()
assert (np.diff(times) == 60).all()

checks = {"target_timestamps": len(target), "step_seconds": 60, "exact_matches": {}}
for source, saved in [
    ("requests", "next10_requests"),
    ("input_tokens", "next10_input_tokens"),
    ("arrival_final_output_tokens", "next10_final_output_tokens"),
]:
    cumulative = np.r_[0, np.cumsum(minute[source].to_numpy(dtype=np.int64))]
    reconstructed = cumulative[positions + 10] - cumulative[positions]
    assert np.array_equal(reconstructed, target[saved].to_numpy())
    checks["exact_matches"][saved] = True

origin = int(coverage.conversation_min_timestamp // 86400) * 86400
target["day"] = (target.timestamp // 86400 - origin // 86400).astype(int)
target["hour"] = (target.timestamp % 86400 // 3600).astype(int)
target["elapsed_days"] = (target.timestamp - origin) / 86400
daily = target.groupby("day").agg(
    n=("timestamp", "size"),
    last=("timestamp", "max"),
    input_mean=(TARGETS[0], "mean"),
    output_mean=(TARGETS[1], "mean"),
)

# This implements the PRD's 60% rule for selecting an illustration only.
# It does not persist or finalize model split boundaries or label eligibility.
cutoff = coverage.conversation_min_timestamp + 0.6 * (
    coverage.conversation_max_timestamp - coverage.conversation_min_timestamp
)
train_days = daily[(daily.n == 1440) & (daily["last"] + 600 < cutoff)]
selected_day = int((train_days.input_mean - train_days.input_mean.median()).abs().idxmin())
one_day = target[target.day == selected_day]
checks["day_selection"] = {
    "rule": "Complete day in the first 60% of time span nearest median daily input-target mean",
    "zero_based_day": selected_day,
    "display_day": selected_day + 1,
    "timestamp_start": int(one_day.timestamp.min()),
    "timestamp_end_exclusive": int(one_day.timestamp.max()) + 60,
    "provisional_complete_days": len(train_days),
    "split_status": "Illustration only; model boundaries and label eligibility are not finalized",
}
day_night = []
for day in train_days.index:
    group = target[target.day == day]
    daytime = group[(group.hour >= 9) & (group.hour < 18)][TARGETS].mean()
    night = group[group.hour < 8][TARGETS].mean()
    day_night.append((daytime > night).to_numpy())
checks["daytime_vs_night"] = {
    "daytime_hours": "09:00-18:00",
    "night_hours": "00:00-08:00",
    "days": len(train_days),
    "input_days_with_higher_daytime_mean": int(np.array(day_night)[:, 0].sum()),
    "output_days_with_higher_daytime_mean": int(np.array(day_night)[:, 1].sum()),
}
checks["period_means"] = {}
for first, end in [(0, 30), (80, 110)]:
    group = target[(target.day >= first) & (target.day < end)]
    checks["period_means"][f"days_{first + 1}_to_{end}"] = group[TARGETS].mean().to_dict()

hourly = target.groupby(target.timestamp // 3600).agg(
    timestamp=("timestamp", "mean"),
    n=("timestamp", "size"),
    **{name: (name, "mean") for name in TARGETS},
)
hourly = hourly[hourly.n == 60]
full_days = daily[daily.n == 1440]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#ccc6be", "axes.labelcolor": "#433c38",
    "text.color": "#352f2c", "xtick.color": "#655c55", "ytick.color": "#655c55",
    "axes.titleweight": "bold", "savefig.facecolor": "#fffaf3",
})

for mode in ["raw", "hourly"]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), sharex="col", sharey="row",
                             gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor("#fffaf3")
    fig.suptitle("Forecasting targets: within-day variation and the full time span",
                 fontsize=19, x=0.07, ha="left", y=0.97, weight="bold")
    axes[0, 0].set_title(f"One complete day | Day {selected_day + 1}", loc="left", pad=14)
    overview_title = "Raw values + daily mean" if mode == "raw" else "Hourly means + daily mean"
    axes[0, 1].set_title(f"About 110 days | {overview_title}", loc="left", pad=14)
    for row, (name, color, label) in enumerate(zip(TARGETS, COLORS, ["Input", "Final output"])):
        left, right = axes[row]
        left.plot((one_day.timestamp % 86400) / 3600, one_day[name], color=color, lw=1)
        if mode == "raw":
            right.plot(target.elapsed_days, target[name], color=color, lw=0.35, alpha=0.5,
                       label="10-min target, every minute")
        else:
            right.plot((hourly.timestamp - origin) / 86400, hourly[name], color=color,
                       lw=0.8, alpha=0.7, label="Hourly mean of 10-min targets")
        mean_name = "input_mean" if row == 0 else "output_mean"
        right.plot(full_days.index + 0.5, full_days[mean_name], color="#352f2c", lw=1.8,
                   label="Daily mean of 10-min targets")
        right.legend(loc="upper left", frameon=False, fontsize=8.5)
        left.set_ylabel(f"{label} tokens / next 10 min")
        for ax in [left, right]:
            ax.set_facecolor("white")
            ax.grid(axis="y", color="#ece8e2", lw=0.7)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        left.set_xlim(0, 24)
        left.set_xticks([0, 6, 12, 18, 24])
        right.set_xlim(0, 110)
    axes[1, 0].set_xlabel("Hour from original Timestamp (UTC+8)")
    axes[1, 1].set_xlabel("Relative day (first observed day starts at 0)")
    fig.text(0.07, 0.04,
             "Targets sum tokens of requests arriving in [t, t+10 min); output is final response length. Adjacent targets share 9 min.\n"
             "Day selection uses median daily input demand in the first 60% of the time span. "
             "Overview means summarize targets; they are not hourly/daily token totals.",
             fontsize=9, color="#655c55", linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.19, hspace=0.18, wspace=0.16)
    fig.savefig(OUT / f"12_target_timeseries_{mode}.png", dpi=180)
    plt.close(fig)

# Compare all horizons on identical, fully observed timestamps.
horizons = [5, 10, 15]
last_full = int(minute.loc[minute.full_observation, "timestamp"].max())
common = times[times + max(horizons) * 60 <= last_full + 60]
common_positions = np.searchsorted(minute_times, common)
rows = []
for horizon in horizons:
    row = {"horizon_minutes": horizon, "common_timestamps": len(common)}
    threshold = followup[followup.threshold_seconds == horizon * 60].iloc[0]
    row["observed_followup_gap_coverage_pct"] = float(threshold.iloc[1])
    for source, key in [("input_tokens", "input"), ("arrival_final_output_tokens", "output")]:
        cumulative = np.r_[0, np.cumsum(minute[source].to_numpy(dtype=np.int64))]
        values = cumulative[common_positions + horizon] - cumulative[common_positions]
        row[f"zero_{key}_target_pct"] = 100 * float(np.mean(values == 0))
    rows.append(row)
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "horizon_context.csv", index=False)
checks["horizon_comparison"] = {
    "common_timestamps": len(common),
    "first_timestamp": int(common.min()), "last_timestamp": int(common.max()),
    "note": "Observed follow-up pairs and forecast timestamps are different denominators",
    "results": rows,
}
(OUT / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(checks, ensure_ascii=False, indent=2))
