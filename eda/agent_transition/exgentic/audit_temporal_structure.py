"""Diagnose within-session order and run a fixed, group-held-out temporal pilot.

The local shard is small and already inspected: results are exploratory, not a
confirmation test. No raw text, provider calls, or original files are written.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source/train-00000-of-00039.parquet"
RESULTS = ROOT / "results"
EXPECTED_HASH = "5cfde1c01460d2f7d90fdc6c245278cee20f32101d5275b6cf3d06f30a5f8f60"
SEED = 20261029
PERMUTATIONS = 999
SHUFFLE_SEEDS = [101, 102, 103, 104, 105]
STATIC = ["input_chars", "input_message_count", "input_text_chars", "latest_user_text_chars",
          "tool_def_count", "has_tool_defs", "input_tool_response_count", "input_tool_response_chars",
          "input_assistant_message_count"]
BAG = ["history_count", "history_mean", "history_median", "history_std", "history_min", "history_max"]
ORDERED = ["lag1", "lag2", "lag3", "last_completed", "recent3_mean", "recent3_delta"]
PARAMETERS = dict(loss="absolute_error", max_iter=100, max_leaf_nodes=7,
                  min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)


def utc(value):
    time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return time.replace(tzinfo=timezone.utc) if time.tzinfo is None else time.astimezone(timezone.utc)


def corr(left, right):
    left, right = np.asarray(left), np.asarray(right)
    good = np.isfinite(left) & np.isfinite(right)
    if good.sum() < 3 or np.std(left[good]) == 0 or np.std(right[good]) == 0:
        return None
    return float(np.corrcoef(left[good], right[good])[0, 1])


def pair_correlation(series, lag=1, center=False):
    arrays = [values - np.nanmean(values) if center else values for values in series]
    return corr(np.concatenate([v[:-lag] for v in arrays if len(v) > lag]),
                np.concatenate([v[lag:] for v in arrays if len(v) > lag]))


def diagnostics(series):
    # Keep failed-call positions as NaN: do not bridge failures or session boundaries.
    per_session = [corr(v[:-1], v[1:]) for v in series]
    per_session = [v for v in per_session if v is not None]
    observed = pair_correlation(series, center=True)
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = []
        for values in series:
            values = values.copy()
            finite = np.isfinite(values)
            values[finite] = rng.permutation(values[finite])
            shuffled.append(values)
        null.append(pair_correlation(shuffled, center=True))
    positive_p = (1 + sum(v >= observed for v in null)) / (PERMUTATIONS + 1)
    valid_only = [v[np.isfinite(v)] for v in series]
    differences = [np.diff(v) for v in valid_only]
    return {
        "pooled_within_session_pairs_lag1": pair_correlation(series),
        "pooled_within_session_pairs_lag2": pair_correlation(series, lag=2),
        "within_session_demeaned_lag1": observed,
        "per_session_lag1_median": float(np.median(per_session)),
        "per_session_lag1_count": len(per_session),
        "valid_call_sequence_lag1": pair_correlation(valid_only),
        "valid_call_sequence_demeaned_lag1": pair_correlation(valid_only, center=True),
        "valid_call_difference_lag1": pair_correlation(differences),
        "permutation_null_95_interval": np.quantile(null, [.025, .975]).tolist(),
        "permutation_one_sided_p": positive_p,
        "permutations": PERMUTATIONS,
        "warning": "Diagnostic after seeing this shard; global per-session centering is not a prediction feature.",
    }


def ordered_features(index, known, values):
    by_index = {position: float(value) for (position, _), value in zip(known, values)}
    return {
        "lag1": by_index.get(index - 1, np.nan),
        "lag2": by_index.get(index - 2, np.nan),
        "lag3": by_index.get(index - 3, np.nan),
        "last_completed": values[-1] if len(values) else np.nan,
        "recent3_mean": float(np.mean(values[-3:])) if len(values) else np.nan,
        "recent3_delta": values[-1] - values[-2] if len(values) >= 2 else np.nan,
    }


def build_frame(rows, shuffle_seed=None):
    old = pd.read_parquet(RESULTS / "next_output_feature_audit.parquet").set_index("span_id")
    rng = np.random.default_rng(shuffle_seed)
    records, input_series, output_series = [], [], []
    for row in rows:
        spans = sorted(row["spans"], key=lambda s: (utc(s["start_time"]), s["span_id"]))
        series = {name: [] for name in ["input_tokens", "output_tokens"]}
        initial_hash = hashlib.sha256(spans[0]["attributes"]["gen_ai.input.messages"].encode()).hexdigest()
        for index, span in enumerate(spans):
            a = span["attributes"]
            valid = span["status"]["code"] == 1 and a.get("gen_ai.output.messages") is not None
            for name in series:
                series[name].append(a["gen_ai.usage." + name] if valid else np.nan)
            if not valid:
                continue
            known = sorted(
                [(j, s) for j, s in enumerate(spans[:index])
                 if s["status"]["code"] == 1 and utc(s["end_time"]) < utc(span["start_time"])],
                key=lambda pair: (utc(pair[1]["end_time"]), pair[1]["span_id"]),
            )
            for _, previous in known:
                assert utc(previous["end_time"]) < utc(span["start_time"])
            values = np.array([s["attributes"]["gen_ai.usage.output_tokens"] for _, s in known], dtype=float)
            feature = old.loc[span["span_id"]]
            assert feature.input_chars == len(a["gen_ai.input.messages"])
            record = {k: float(feature[k]) for k in STATIC}
            record.update({
                "session_id": row["session_id"], "span_id": span["span_id"], "group": initial_hash,
                "model": a["gen_ai.request.model"], "target": a["gen_ai.usage.output_tokens"],
                "history_count": len(values),
                "history_mean": float(np.mean(values)) if len(values) else np.nan,
                "history_median": float(np.median(values)) if len(values) else np.nan,
                "history_std": float(np.std(values)) if len(values) else np.nan,
                "history_min": float(np.min(values)) if len(values) else np.nan,
                "history_max": float(np.max(values)) if len(values) else np.nan,
            })
            if shuffle_seed is not None:
                # Same known values and missing-lag positions; only their order changes.
                values = rng.permutation(values)
            record.update(ordered_features(index, known, values))
            records.append(record)
        input_series.append(np.array(series["input_tokens"], dtype=float))
        output_series.append(np.array(series["output_tokens"], dtype=float))
    return pd.DataFrame(records), input_series, output_series


def evaluate(frame, rows):
    groups = frame.group.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(frame, frame.target, groups))
    specifications = {"current_structure": STATIC, "unordered_history": STATIC + BAG,
                      "ordered_history": STATIC + BAG + ORDERED}
    predictions = frame[["session_id", "span_id", "group", "model", "target", "history_count"]].copy()
    fold_metadata = []
    for name in ["model_median", "last_completed_baseline", *specifications]:
        predictions[name] = np.nan
    for fold, (train, test) in enumerate(folds):
        training, testing = frame.iloc[train], frame.iloc[test]
        assert set(training.group).isdisjoint(testing.group)
        assert set(training.session_id).isdisjoint(testing.session_id)
        fallback = float(training.target.median())
        medians = training.groupby("model").target.median()
        predictions.loc[test, "model_median"] = testing.model.map(medians).fillna(fallback)
        predictions.loc[test, "last_completed_baseline"] = testing.last_completed.fillna(
            testing.model.map(medians).fillna(fallback)
        )
        fold_metadata.append({"fold": fold, "train_sessions": training.session_id.nunique(),
                              "test_sessions": testing.session_id.nunique(), "train_rows": len(train),
                              "test_rows": len(test)})
        for name, columns in specifications.items():
            # Fit model vocabulary on training data; unknown held-out models map to all zeros.
            train_model = pd.get_dummies(training.model, dtype=float)
            test_model = pd.get_dummies(testing.model, dtype=float).reindex(columns=train_model.columns, fill_value=0)
            xtrain = np.column_stack([training[columns].to_numpy(), train_model.to_numpy()])
            xtest = np.column_stack([testing[columns].to_numpy(), test_model.to_numpy()])
            estimator = HistGradientBoostingRegressor(**PARAMETERS)
            estimator.fit(xtrain, training.target)
            predictions.loc[test, name] = np.maximum(0, estimator.predict(xtest))
    for seed in SHUFFLE_SEEDS:
        shuffled, _, _ = build_frame(rows, shuffle_seed=seed)
        assert frame.span_id.tolist() == shuffled.span_id.tolist()
        assert frame[STATIC + BAG].equals(shuffled[STATIC + BAG])
        assert frame.target.equals(shuffled.target)
        assert frame[["lag1", "lag2", "lag3"]].isna().equals(shuffled[["lag1", "lag2", "lag3"]].isna())
        name = f"shuffled_order_{seed}"
        predictions[name] = np.nan
        for train, test in folds:
            columns = STATIC + BAG + ORDERED
            train_model = pd.get_dummies(shuffled.iloc[train].model, dtype=float)
            test_model = pd.get_dummies(shuffled.iloc[test].model, dtype=float).reindex(columns=train_model.columns, fill_value=0)
            xtrain = np.column_stack([shuffled.iloc[train][columns], train_model])
            xtest = np.column_stack([shuffled.iloc[test][columns], test_model])
            estimator = HistGradientBoostingRegressor(**PARAMETERS)
            estimator.fit(xtrain, frame.iloc[train].target)
            predictions.loc[test, name] = np.maximum(0, estimator.predict(xtest))
    names = [c for c in predictions if c not in ["session_id", "span_id", "group", "model", "target", "history_count"]]
    metrics = {}
    for name in names:
        error = (predictions[name] - predictions.target).abs()
        metrics[name] = {"mae": float(error.mean()),
                         "session_macro_mae": float(error.groupby(predictions.session_id).mean().mean()),
                         "group_macro_mae": float(error.groupby(predictions.group).mean().mean()),
                         "history3_mae": float(error[predictions.history_count >= 3].mean())}
    delta = ((predictions.unordered_history - predictions.target).abs()
             - (predictions.ordered_history - predictions.target).abs())
    group_deltas = delta.groupby(predictions.group).mean().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = rng.choice(group_deltas, size=(5000, len(group_deltas)), replace=True).mean(axis=1)
    comparison = {"positive_means_ordered_better": True,
                  "group_macro_mae_gain_ordered_vs_unordered": float(group_deltas.mean()),
                  "exploratory_group_bootstrap_95_interval": np.quantile(boot, [.025, .975]).tolist(),
                  "groups": len(group_deltas), "bootstrap_draws": 5000,
                  "shuffled_order_mae_range": [min(metrics[f"shuffled_order_{s}"]["mae"] for s in SHUFFLE_SEEDS),
                                               max(metrics[f"shuffled_order_{s}"]["mae"] for s in SHUFFLE_SEEDS)]}
    assert predictions[names].notna().all().all()
    return predictions, metrics, comparison, fold_metadata


if __name__ == "__main__":
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_HASH
    ROWS = pq.read_table(SOURCE).to_pylist()
    frame, input_series, output_series = build_frame(ROWS)
    input_diagnostics = diagnostics(input_series)
    output_diagnostics = diagnostics(output_series)
    predictions, metrics, comparison, folds = evaluate(frame, ROWS)
    summary = {
        "source_sha256": EXPECTED_HASH, "scope": "Shard 0, appworld/claude_code only; exploratory",
        "target": "Same-call successful output usage, before response generation",
        "seed": SEED, "sessions": len(ROWS), "labels": len(frame), "initial_input_groups": frame.group.nunique(),
        "successful_session_lengths": np.quantile([np.isfinite(v).sum() for v in output_series], [0,.25,.5,.75,1]).tolist(),
        "diagnostic_definition": "Pearson correlation of lagged within-session pairs; no session-boundary pairs. Failures stay NaN. Not proof of a specific ARIMA process.",
        "input": input_diagnostics, "output": output_diagnostics,
        "observed_lag_counts": {name: int(frame[name].notna().sum()) for name in ["lag1", "lag2", "lag3"]},
        "metrics": metrics, "comparison": comparison, "folds": folds,
        "model_parameters": PARAMETERS, "feature_sets": {"static": STATIC, "bag": BAG, "ordered": ORDERED},
        "feature_audit_sha256": hashlib.sha256((RESULTS / "next_output_feature_audit.parquet").read_bytes()).hexdigest(),
        "limits": ["Timezones and usage availability inherit the previous audit assumptions.",
                   "Static baseline uses numeric request structure, not full semantic text representations.",
                   "This is group-held-out prefix evaluation, not training at successive real calendar dates.",
                   "Fixed one-model configuration without tuning; not a proof of absence or presence of all temporal predictability.",
                   "Shuffle preserves observed-history values and available-lag masks; it does not simulate plausible agent trajectories."]}
    (RESULTS / "temporal_structure_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    predictions.to_csv(RESULTS / "temporal_pilot_predictions.csv", index=False)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["feature_sets","folds","model_parameters","limits"]}, indent=2))
