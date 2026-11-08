"""Four train-only quantile baselines and prediction-time history summaries."""
import bisect

import numpy as np
import pandas as pd

from .features import history_allowed
from .split import SESSION_COLUMNS

QUANTILES = (.1, .5, .9)
PREDICTION_COLUMNS = ("P10", "P50", "P90")
MIN_GROUP_ROWS = 100
BASELINES = ("fixed", "model", "previous_completion", "session_mean")


def build_history(requests):
    """Lag and cumulative mean with the same gate as build_features.

    A completion sweep plus prefix sums over steps excludes unfinished and
    higher-step requests even when caller order and start order differ.
    Missing completion values are omitted from the cumulative mean.
    """
    keys = list(SESSION_COLUMNS)
    if (requests[[*keys, "step"]].isna().any().any()
            or requests.duplicated([*keys, "step"]).any()
            or requests.start_ms.isna().any() or requests.step.lt(0).any()
            or (requests.step % 1 != 0).any()
            or (requests.end_ms < requests.start_ms).any()):
        raise ValueError("Invalid request keys, steps or times")
    rows = requests.reset_index(drop=True)
    result = np.full((len(rows), 2), np.nan)
    records = rows.to_dict("records")
    for _, group in rows.groupby(keys, sort=False):
        by_step = {int(records[i]["step"]): i for i in group.index}
        steps = sorted(by_step)
        ranks = {step: i + 1 for i, step in enumerate(steps)}
        totals, counts = np.zeros(len(steps) + 1), np.zeros(len(steps) + 1)
        ended = sorted((records[i]["end_ms"], int(records[i]["step"]), i)
                       for i in group.index if pd.notna(records[i]["end_ms"]))
        available, cursor = [], 0
        for pos in group.sort_values(["start_ms", "step"], kind="stable").index:
            row = records[pos]
            while cursor < len(ended) and ended[cursor][0] < row["start_ms"]:
                _, old_step, old_pos = ended[cursor]
                available.append(old_pos)
                value = records[old_pos]["completion_tokens"]
                if pd.notna(value):
                    rank = ranks[old_step]
                    while rank < len(totals):
                        totals[rank] += value
                        counts[rank] += 1
                        rank += rank & -rank
                cursor += 1
            step = int(row["step"])
            prev_pos = by_step.get(step - 1)
            if not history_allowed(row, records[prev_pos] if prev_pos is not None else None):
                continue
            for old_pos in reversed(available):
                if records[old_pos]["step"] < step:
                    result[pos, 0] = records[old_pos]["completion_tokens"]
                    break
            rank = bisect.bisect_left(steps, step)
            total, count = 0., 0.
            while rank:
                total += totals[rank]
                count += counts[rank]
                rank -= rank & -rank
            if count:
                result[pos, 1] = total / count
    return pd.DataFrame(result, columns=["completion_lag_1", "session_completion_mean"],
                        index=requests.index)


class Baselines:
    """Fit accepts a label mask; predict never reads current target values."""

    def fit(self, requests, history, train_target):
        if not requests.index.equals(history.index):
            raise ValueError("History must align with requests")
        mask = np.asarray(train_target, dtype=bool)
        if mask.shape != (len(requests),) or not mask.any():
            raise ValueError("A nonempty aligned training mask is required")
        train = requests.iloc[np.flatnonzero(mask)]
        y = train.completion_tokens.to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Training targets must be finite and non-negative")
        self.fixed = np.quantile(y, QUANTILES, method="linear")
        self.by_model = {model: np.quantile(group.completion_tokens, QUANTILES, method="linear")
                         for model, group in train.groupby("model", sort=False)
                         if len(group) >= MIN_GROUP_ROWS}
        self.ratios = {}
        for name in history:
            values = history[name].to_numpy(dtype=float)[mask]
            valid = np.isfinite(values)
            self.ratios[name] = (np.quantile(y[valid] / np.maximum(values[valid], 1), QUANTILES,
                                           method="linear") if valid.any() else None)
        return self

    def predict(self, requests, history):
        if not requests.index.equals(history.index):
            raise ValueError("History must align with requests")
        model_fallback = ~requests.model.isin(self.by_model).to_numpy()
        model = np.array([self.by_model.get(value, self.fixed) for value in requests.model], dtype=float)
        predictions = {"fixed": np.tile(self.fixed, (len(requests), 1)), "model": model}
        fallback = {"fixed": np.zeros(len(requests), dtype=bool), "model": model_fallback}
        for name, column in (("previous_completion", "completion_lag_1"),
                             ("session_mean", "session_completion_mean")):
            values = history[column].to_numpy(dtype=float)
            use_model = ~np.isfinite(values) | (self.ratios[column] is None)
            result = model.copy()
            if self.ratios[column] is not None:
                result[~use_model] = values[~use_model, None] * self.ratios[column]
            predictions[name] = result
            fallback[name] = use_model
        return ({name: pd.DataFrame(np.sort(values, axis=1), index=requests.index,
                                    columns=PREDICTION_COLUMNS)
                 for name, values in predictions.items()},
                pd.DataFrame(fallback, index=requests.index))
