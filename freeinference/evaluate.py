"""Reusable grouped quantile metrics and paired cluster bootstrap; baseline CLI."""
import hashlib
import json
import platform
import sys
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow

from .baselines import (BASELINES, MIN_GROUP_ROWS, PREDICTION_COLUMNS, QUANTILES,
                        Baselines, build_history)
from .loader import FIRST_WEEK, REVISION, data_dir, load_table, weeks
from .split import (FIT_WEEKS, HOLDOUT_START_MS, HOLDOUT_WEEK, SESSION_COLUMNS,
                    TRAIN_WEEKS, VALIDATION_START_MS, VALIDATION_WEEKS, split_masks)

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261109
METRICS = ("pinball_p10", "pinball_p50", "pinball_p90", "mean_pinball",
           "mae_p50", "coverage", "mean_width")
WEIGHTINGS = ("request", "session")
OUTPUT = Path(__file__).parent / "outputs" / "baselines"
REQUEST_COLUMNS = ["session_id", "step", "start_ms", "end_ms", "link_reason",
                   "completion_tokens", "model"]


def _codes(frame, columns):
    if not columns or frame[list(columns)].isna().any().any():
        raise ValueError("Cluster/session keys must be non-null")
    return pd.factorize(pd.MultiIndex.from_frame(frame[list(columns)]), sort=False)[0]


def _losses(y, prediction):
    values = prediction.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=float)
    if values.shape != (len(y), 3) or not np.isfinite(values).all():
        raise ValueError("Predictions must be finite P10/P50/P90 values")
    if (np.diff(values, axis=1) < 0).any():
        raise ValueError("Quantile crossing refused")
    residual = y[:, None] - values
    pinball = np.maximum(np.array(QUANTILES) * residual,
                         (np.array(QUANTILES) - 1) * residual)
    return np.column_stack((pinball, pinball.mean(axis=1), np.abs(residual[:, 1]),
                            (values[:, 0] <= y) & (y <= values[:, 2]),
                            values[:, 2] - values[:, 0]))


def _prepare(rows, predictions):
    if rows.empty or not predictions:
        raise ValueError("Nonempty evaluation rows and predictions are required")
    y = rows.completion_tokens.to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Evaluation targets must be finite and non-negative")
    losses = {}
    for name, prediction in predictions.items():
        if not rows.index.equals(prediction.index):
            raise ValueError("Predictions must align with evaluation rows")
        losses[name] = _losses(y, prediction)
    return losses


def _groups(rows):
    yield "overall", np.ones(len(rows), dtype=bool)
    yield "step/0", rows.step.eq(0).to_numpy()
    yield "step/followup", rows.step.gt(0).to_numpy()
    for column in ("harness", "model"):
        # None has a distinct label even if a real category is '<missing>'.
        for value in rows[column].drop_duplicates():
            missing = pd.isna(value)
            mask = rows[column].isna() if missing else rows[column].eq(value).fillna(False)
            label = "missing" if missing else f"value={value}"
            yield f"{column}/{label}", mask.to_numpy(dtype=bool)


def _group_weights(mask, sessions):
    counts = np.bincount(sessions[mask], minlength=int(sessions.max()) + 1)
    return np.column_stack((np.ones(mask.sum()), 1 / counts[sessions[mask]]))


def evaluate_predictions(rows, predictions):
    """Metrics by group, with each session totaling one *within that group*."""
    losses = _prepare(rows, predictions)
    sessions = _codes(rows, SESSION_COLUMNS)
    result = {}
    for group, mask in _groups(rows):
        if not mask.any():
            continue
        weights = _group_weights(mask, sessions)
        metrics = {}
        for name, loss in losses.items():
            metrics[name] = {
                weighting: dict(zip(METRICS, map(float, np.average(loss[mask], axis=0,
                                                                 weights=weights[:, i]))))
                for i, weighting in enumerate(WEIGHTINGS)}
        result[group] = {"rows": int(mask.sum()), "sessions": int(np.unique(sessions[mask]).size),
                         "metrics": metrics}
    return result


def cluster_draws(n_clusters, *, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, batch_size=64):
    """Shared multinomial cluster multiplicities; entire clusters move together."""
    if n_clusters < 1 or B < 1 or batch_size < 1:
        raise ValueError("Cluster count, B and batch size must be positive")
    rng = np.random.default_rng(seed)
    for offset in range(0, B, batch_size):
        yield rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters),
                              size=min(batch_size, B - offset))


def paired_bootstrap(rows, predictions, reference="fixed", *, cluster_columns=SESSION_COLUMNS,
                     B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Percentile 95% CIs of predictor minus reference, without refitting.

    One cluster draw is shared across every predictor, group, and weighting.
    Session weights are computed within groups before resampling; duplicating a
    session duplicates its unit weight. Other cluster columns can define account
    or day blocks without changing this session-weighting definition.
    """
    losses = _prepare(rows, predictions)
    if reference not in losses:
        raise ValueError("Reference predictor is absent")
    clusters = _codes(rows, cluster_columns)
    sessions = _codes(rows, SESSION_COLUMNS)
    n_clusters = int(clusters.max()) + 1
    names = list(predictions)
    groups = [(name, mask) for name, mask in _groups(rows) if mask.any()]
    denominators, numerators = [], []
    for _, mask in groups:
        weights = _group_weights(mask, sessions)
        denominator = np.zeros((n_clusters, 2))
        np.add.at(denominator, clusters[mask], weights)
        denominators.append(denominator)
        for name in names:
            numerator = np.zeros((n_clusters, 2, len(METRICS)))
            diff = losses[name][mask] - losses[reference][mask]
            np.add.at(numerator, clusters[mask], weights[:, :, None] * diff[:, None, :])
            numerators.append(numerator.reshape(n_clusters, -1))
    denominator_matrix = np.concatenate(denominators, axis=1)
    numerator_matrix = np.concatenate(numerators, axis=1)
    samples = []
    for draws in cluster_draws(n_clusters, B=B, seed=seed):
        denom = (draws @ denominator_matrix).reshape(-1, len(groups), 1, 2, 1)
        numer = (draws @ numerator_matrix).reshape(-1, len(groups), len(names), 2, len(METRICS))
        samples.append(np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0))
    samples = np.concatenate(samples)
    intervals = {}
    for g, (group, _) in enumerate(groups):
        valid = np.isfinite(samples[:, g, 0, 0, 0])
        intervals[group] = {
            "valid_replicates": int(valid.sum()),
            "differences": {
                name: {weighting: {metric: (list(map(float, np.percentile(
                    samples[valid, g, m, w, k], [2.5, 97.5], method="linear")))
                                           if valid.any() else [None, None])
                                   for k, metric in enumerate(METRICS)}
                       for w, weighting in enumerate(WEIGHTINGS)}
                for m, name in enumerate(names)}}
    return {"reference": reference, "cluster_columns": list(cluster_columns),
            "clusters": n_clusters, "B": B, "seed": seed,
            "ci": "95% percentile, linear interpolation; predictor minus reference",
            "empty_group_draws": "omit and report valid_replicates per group",
            "groups": intervals}


def _set_counts(rows, mask):
    selected = rows.loc[mask]
    return {"rows": len(selected), "sessions": len(selected[list(SESSION_COLUMNS)].drop_duplicates())}


def _markdown(result):
    lines = ["# 기준선 validation 평가", "",
             f"- revision: `{result['revision']}`; 코드 HEAD: `{result['code_revision']}`",
             f"- 실행: `{result['command']}`",
             f"- 환경: `{json.dumps(result['environment'], ensure_ascii=False)}`",
             f"- 소요: {result['elapsed_seconds']:.3f}초; 1~13주만 읽음. test 미사용.",
             f"- 분할: `{json.dumps(result['split'], ensure_ascii=False)}`",
             f"- 세트별 행·세션 수: `{json.dumps(result['sets'])}`",
             f"- train 경계 제외 정답: {result['excluded_train_targets']}; fit 경계 제외: {result['excluded_fit_targets']}",
             f"- bootstrap: 세션 단위 짝지은 B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED}, 차이(예측기−fixed)의 95% 백분위 CI.",
             "- 요청 가중은 행마다 1, 세션 가중은 해당 구간 안에서 세션마다 합계 1. 재학습 없음.",
             "- pinball·MAE·폭 단위는 토큰, coverage·대체율은 0~1 비율. 수치는 표시 자릿수만 반올림.",
             "- 결측 lag/평균은 모델별로, 부족·미관측 모델은 고정 분위수로 대체. 연쇄 대체율도 JSON에 기록.",
             "- 전체 수치·설정: [baselines.json](baselines.json)", "",
             "## 대체 사용 비율", "", "| 기준선 | train | validation |", "|---|---:|---:|"]
    for name in BASELINES:
        lines.append(f"| {name} | {result['fallback_rates']['train'][name]['fallback_rate']:.6f} | "
                     f"{result['fallback_rates']['validation'][name]['fallback_rate']:.6f} |")
    lines += ["", "## 구간별 지표", "",
              "| 구간 | 행 | 세션 | 기준선 | 가중 | " + " | ".join(METRICS) + " |",
              "|---|---:|---:|---|---|" + "---:|" * len(METRICS)]
    for group, values in result["metrics"].items():
        label = group.replace("|", "\\|")
        for name, metrics in values["metrics"].items():
            for weighting, scores in metrics.items():
                lines.append(f"| {label} | {values['rows']} | {values['sessions']} | {name} | {weighting} | "
                             + " | ".join(f"{scores[key]:.6f}" for key in METRICS) + " |")
    lines += ["", "## fixed 대비 차이의 95% CI", "",
              "음수 pinball·MAE는 오차 감소다. coverage·폭 차이는 개선 여부를 단독 판정하지 않는다.", "",
              "| 구간 | 유효 반복 | 기준선 | 가중 | " + " | ".join(METRICS) + " |",
              "|---|---:|---|---|" + "---|" * len(METRICS)]
    for group, values in result["bootstrap"]["groups"].items():
        label = group.replace("|", "\\|")
        for name, metrics in values["differences"].items():
            for weighting, scores in metrics.items():
                lines.append(f"| {label} | {values['valid_replicates']} | {name} | {weighting} | "
                             + " | ".join(f"[{scores[k][0]:.6f}, {scores[k][1]:.6f}]" for k in METRICS) + " |")
    return "\n".join(lines) + "\n"


def run(root=None, output=OUTPUT):
    started = perf_counter()
    paths = weeks(root)
    if len(paths) != 13 or any(p.name[:10] != (FIRST_WEEK + timedelta(weeks=i)).isoformat()
                             for i, p in enumerate(paths)):
        raise ValueError("Evaluation requires consecutive train/validation weeks 1–13")
    weekly, histories = [], []
    for number, path in enumerate(paths, 1):
        requests = load_table(path, "requests", REQUEST_COLUMNS)
        sessions = load_table(path, "sessions", ["session_id", "harness"])
        requests = requests.merge(sessions, on="session_id", how="left", sort=False,
                                  validate="many_to_one", indicator=True)
        if requests["_merge"].ne("both").any():
            raise ValueError("Request without matching session")
        requests = requests.drop(columns="_merge")
        requests["week"] = number
        histories.append(build_history(requests))
        weekly.append(requests)
        print(f"week {number}: {len(requests):,} rows; history built", flush=True)
    rows = pd.concat(weekly, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    masks = split_masks(rows)
    model = Baselines().fit(rows, history, masks.train_target)
    predictions, fallback = model.predict(rows, history)
    validation = rows.loc[masks.validation].copy()
    val_predictions = {name: frame.loc[masks.validation] for name, frame in predictions.items()}
    metrics = evaluate_predictions(validation, val_predictions)
    print("validation metrics ready; paired bootstrap", flush=True)
    bootstrap = paired_bootstrap(validation, val_predictions)
    fallback_rates = {}
    for subset in ("train", "validation"):
        mask = masks[f"{subset}_target"]
        fallback_rates[subset] = {}
        for name in BASELINES:
            used = fallback.loc[mask, name]
            model_used = fallback.loc[mask, "model"]
            fallback_rates[subset][name] = {
                "rows": int(mask.sum()), "fallback_rows": int(used.sum()),
                "fallback_rate": float(used.mean()),
                "fixed_fallback_rate": float((used & model_used).mean()) if name != "fixed" else 0.}
    # Git metadata only; never read local configuration or credentials.
    import subprocess
    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {"revision": REVISION, "code_revision": code_revision,
              "code_state": "working tree implementation; harness records final commit",
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("loader.py", "features.py", "split.py", "baselines.py", "evaluate.py")},
              "command": f"{sys.executable} -m freeinference.evaluate",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
                              "platform": platform.platform()},
              "split": {"train_weeks": TRAIN_WEEKS, "validation_weeks": VALIDATION_WEEKS,
                        "fit_weeks": FIT_WEEKS, "holdout_week": HOLDOUT_WEEK,
                        "validation_start_ms": VALIDATION_START_MS, "holdout_start_ms": HOLDOUT_START_MS,
                        "session_assignment": "first start_ms week; identity=(public week, session_id)",
                        "target_boundary": "end_ms < next split start; missing end excluded; history unchanged"},
              "sets": {name: _set_counts(rows, masks[name]) for name in masks if name != "session_week"},
              "excluded_train_targets": int((masks.train & ~masks.train_target).sum()),
              "excluded_fit_targets": int((masks.fit & ~masks.fit_target).sum()),
              "baseline_settings": {"quantiles": QUANTILES, "quantile_method": "linear",
                                    "MIN_GROUP_ROWS": MIN_GROUP_ROWS,
                                    "ratio": "train target / max(history, 1), empirical quantiles; prediction=history*r_q",
                                    "no_ratio_training_rows": "use model quantiles",
                                    "crossing": "sort P10/P50/P90"},
              "weighting": "request=1 per row; session=1 per session within each group",
              "fallback_rates": fallback_rates, "metrics": metrics, "bootstrap": bootstrap,
              "elapsed_seconds": perf_counter() - started}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "baselines.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    (output / "baselines.md").write_text(_markdown(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(data_dir())
