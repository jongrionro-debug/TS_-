#!/usr/bin/env python3
"""Compare prompt-only and history LightGBM quantiles; NEVER open test.jsonl."""
import sys
from pathlib import Path

# Local inspect.py must not shadow the stdlib module imported by NumPy/LightGBM.
if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import shlex
import time

from wildchat.metrics import QUANTILES, baseline, empirical_quantile, evaluate
from wildchat.prepare import timestamp_us

PROMPT_FEATURES = ("input_tokens", "input_chars", "input_utf8_bytes")
SEARCH_SPACE = (
    {"num_leaves": 15, "min_data_in_leaf": 100},
    {"num_leaves": 31, "min_data_in_leaf": 100},
    {"num_leaves": 31, "min_data_in_leaf": 300},
)
TEST_REASON = "사용자/오케스트레이터 지시로 최종 시험 평가를 보류함; test.jsonl은 읽지 않음"
BOOTSTRAP_METRICS = ("mean_pinball_loss", "mae_p50", "p10_p90_coverage", "mean_interval_width")
COMPARISONS = ("baseline", "A", "B")


def feature_names(lags, interval_lags=()):
    """Explicit numeric allowlist; metadata/diagnostics/targets cannot enter X."""
    history = ["previous_user_turns"]
    for lag in range(1, lags + 1):
        history += [f"lag_{lag}_{side}_{unit}" for side in ("input", "output")
                    for unit in ("tokens", "chars", "utf8_bytes")]
        history += [f"lag_{lag}_{kind}_missing" for kind in ("turn", "input", "output")]
        if lag in interval_lags:
            history += [f"lag_{lag}_turn_interval_seconds", f"lag_{lag}_turn_interval_missing"]
    return {"A": list(PROMPT_FEATURES), "B": [*PROMPT_FEATURES, *history]}


def feature_vector(row, names):
    values = []
    for name in names:
        source = row["prompt_features"] if name in PROMPT_FEATURES else row["history_features"]
        value = source[name]
        if value is None:
            if name in PROMPT_FEATURES or name == "previous_user_turns" or name.endswith("_missing"):
                raise ValueError(f"required feature is null: {name}")
            value = math.nan  # Native LightGBM missing value; no fitted imputation.
        elif name.endswith("_missing"):
            if value not in (False, True):
                raise ValueError(f"invalid missing indicator: {name}")
            value = float(value)
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"feature must be a finite nonnegative number: {name}")
        values.append(value)
    return values


@dataclass
class Partition:
    x: object
    y: object
    summary: dict
    groups: set
    conversations: set
    row_groups: object = None
    turn_indices: object = None


def load_partition(prepared, split, manifest, names):
    """Stream only the two allowed files into bounded numeric arrays."""
    import numpy as np
    if split not in ("train", "validation"):
        raise ValueError("only train and validation may be read; test is forbidden")
    expected = manifest["partitions"][split]
    n = expected["samples"]
    if n <= 0:
        raise ValueError(f"empty {split} partition")
    x, y = np.empty((n, len(names)), dtype=np.float64), np.empty(n, dtype=np.float64)
    row_groups, turn_indices = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    groups, conversations, identities = set(), set(), set()
    first_turns, count = 0, 0
    time_min = time_max = None
    lower = timestamp_us(manifest["boundaries"]["train_validation_cutoff"])
    upper = timestamp_us(manifest["boundaries"]["validation_test_cutoff"])
    if lower is None or upper is None or lower >= upper:
        raise ValueError("valid chronological preparation boundaries required")
    fingerprint = hashlib.sha256()
    with (prepared / f"{split}.jsonl").open("rb") as handle:
        for raw in handle:
            fingerprint.update(raw)
            row = json.loads(raw)
            meta = row["metadata"]
            stamp = timestamp_us(meta["recorded_at"])
            if meta["split"] != split or stamp is None or not (
                stamp < lower if split == "train" else lower <= stamp < upper
            ):
                raise ValueError(f"invalid split or recorded_at boundary: {split}")
            if count >= n:
                raise ValueError(f"{split} sample count differs from manifest")
            target = row["target"]["output_tokens"]
            if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or target < 0:
                raise ValueError("output target must be finite and nonnegative")
            # An interval must have verified submission semantics AND valid slot diagnostics.
            for name in names:
                if name.endswith("_interval_seconds") and row["history_features"][name] is not None:
                    lag = name.split("_")[1]
                    if (not manifest["verified_user_timestamps"] or
                        row["history_features"][f"lag_{lag}_turn_interval_missing"] or
                        lag in row["diagnostics"]["interval_exclusion_reasons"]):
                        raise ValueError("unverified or invalid turn interval")
            identity = (meta["source_row_id"], meta["user_message_index"])
            if identity in identities:
                raise ValueError("duplicate prepared sample identity")
            identities.add(identity)
            if meta["turn_index"] != row["history_features"]["previous_user_turns"]:
                raise ValueError("turn index disagrees with past-turn count")
            if any(isinstance(meta[key], bool) or not isinstance(meta[key], int) or meta[key] < 0
                   for key in ("group_id", "turn_index")):
                raise ValueError("nonnegative integer group_id and turn_index required")
            row_groups[count], turn_indices[count] = meta["group_id"], meta["turn_index"]
            first_turns += int(meta["turn_index"] == 0)
            groups.add(meta["group_id"])
            conversations.add(meta["source_row_id"])
            time_min = min(time_min, stamp) if time_min is not None else stamp
            time_max = max(time_max, stamp) if time_max is not None else stamp
            x[count], y[count] = feature_vector(row, names), target
            count += 1
    if (count != n or len(groups) != expected["groups"] or
        len(conversations) != expected["conversations"] or
        time_min != timestamp_us(expected["recorded_at_min"]) or
        time_max != timestamp_us(expected["recorded_at_max"])):
        raise ValueError(f"{split} counts/time range differ from preparation manifest")
    summary = {**expected, "first_turn_samples": first_turns, "sha256": fingerprint.hexdigest()}
    return Partition(x, y, summary, groups, conversations, row_groups, turn_indices)


def check_disjoint(train, validation):
    if train.groups & validation.groups or train.conversations & validation.conversations:
        raise ValueError("train/validation group or source-row intersection")
    if timestamp_us(train.summary["recorded_at_max"]) >= timestamp_us(validation.summary["recorded_at_min"]):
        raise ValueError("train/validation recorded time overlap")


def select_interval_lags(train, names, lags, verified):
    """Decide optional columns from train only, never validation or manifest totals."""
    import numpy as np
    return [lag for lag in range(1, lags + 1) if verified and
            np.isfinite(train.x[:, names.index(f"lag_{lag}_turn_interval_seconds")]).any()]


def order_quantiles(predictions):
    """Record raw crossings, clip negative values, then sort each row ascending."""
    import numpy as np
    raw = np.asarray(predictions, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("finite nonempty (n, 3) predictions required")
    crossings = int(np.any(raw[:, :-1] > raw[:, 1:], axis=1).sum())
    negative = int(np.any(raw < 0, axis=1).sum())
    return np.sort(np.maximum(raw, 0), axis=1), {
        "raw_crossing_rows": crossings, "raw_crossing_rate": crossings / len(raw),
        "negative_prediction_rows": negative, "n": len(raw),
    }


def average_ranks(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values))
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return ranks


def pearson(left, right):
    import numpy as np
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    a, b = left - left.mean(), right - right.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else None


def descriptive_stats(inputs, outputs):
    import numpy as np
    def distribution(values):
        ordered = sorted(values.tolist())
        return {"n": len(values), "mean": float(values.mean()), "std_population": float(values.std()),
                "min": ordered[0], **{f"p{int(q * 100)}": empirical_quantile(ordered, q)
                for q in (.1, .5, .9, .99)}, "max": ordered[-1]}
    return {"input_tokens": distribution(inputs), "output_tokens": distribution(outputs),
            "correlations": {"pearson_raw": pearson(inputs, outputs),
                             "spearman_average_ties": pearson(average_ranks(inputs), average_ranks(outputs)),
                             "pearson_log1p": pearson(np.log1p(inputs), np.log1p(outputs))}}


def metric_difference(left, right):
    """Signed right-minus-left; coverage/width are descriptive, not ranked scores."""
    result = {k: right[k] - left[k] for k in left if k not in ("n", "pinball_loss")}
    result["pinball_loss"] = {k: right["pinball_loss"][k] - left["pinball_loss"][k]
                             for k in left["pinball_loss"]}
    return result


def cohort_masks(turn_indices):
    import numpy as np
    turns = np.asarray(turn_indices)
    if turns.ndim != 1 or not np.issubdtype(turns.dtype, np.integer) or np.any(turns < 0):
        raise ValueError("one-dimensional nonnegative integer turn indices required")
    return {"all": np.ones(len(turns), dtype=bool), "first_turn": turns == 0, "followup_turn": turns > 0}


def cluster_bootstrap(targets, predictions, group_ids, turn_indices, seed, replicates):
    """Paired #5-group percentile CIs, conditional on fixed selected predictions.

    Each draw samples G of the G validation groups with replacement and keeps
    every selected group's rows (including multiplicities). The same draw is
    shared by all models and cohorts. Metrics remain row-weighted ratios of
    group sums to row counts, rather than unweighted averages of group metrics.
    """
    import numpy as np
    y, groups = np.asarray(targets, dtype=float), np.asarray(group_ids)
    masks = cohort_masks(turn_indices)
    if (y.ndim != 1 or not len(y) or groups.shape != y.shape or
        len(next(iter(masks.values()))) != len(y) or set(predictions) != set(COMPARISONS)):
        raise ValueError("aligned nonempty targets/groups/turns and exactly baseline/A/B required")
    if replicates <= 0 or seed < 0:
        raise ValueError("positive bootstrap replicates and nonnegative seed required")
    unique_groups, codes = np.unique(groups, return_inverse=True)
    g = len(unique_groups)
    # Each metric is an additive row contribution; grouping these avoids storing
    # or repeatedly materializing resampled rows/predictions or a huge draw matrix.
    contributions = np.empty((len(y), len(COMPARISONS), len(BOOTSTRAP_METRICS)))
    points = {cohort: {} for cohort in masks}
    for index, name in enumerate(COMPARISONS):
        pred = np.asarray(predictions[name], dtype=float)
        if pred.shape != (len(y), 3):
            raise ValueError("each prediction array must have shape (n, 3)")
        # Reuse #6 validation/formulas for point estimates, including rejecting crossings.
        for cohort, mask in masks.items():
            points[cohort][name] = evaluate(y[mask].tolist(), pred[mask].tolist()) if mask.any() else None
        error = y[:, None] - pred
        contributions[:, index, 0] = np.maximum(np.asarray(QUANTILES) * error,
                                               (np.asarray(QUANTILES) - 1) * error).mean(axis=1)
        contributions[:, index, 1] = np.abs(y - pred[:, 1])
        contributions[:, index, 2] = (pred[:, 0] <= y) & (y <= pred[:, 2])
        contributions[:, index, 3] = pred[:, 2] - pred[:, 0]
    group_sums = np.zeros((g, len(masks), len(COMPARISONS), len(BOOTSTRAP_METRICS)))
    group_counts = np.zeros((g, len(masks)), dtype=np.int64)
    for index, mask in enumerate(masks.values()):
        np.add.at(group_sums[:, index], codes[mask], contributions[mask])
        np.add.at(group_counts[:, index], codes[mask], 1)
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.full((replicates, len(masks), len(COMPARISONS), len(BOOTSTRAP_METRICS)), np.nan)
    for iteration in range(replicates):
        weights = np.bincount(rng.integers(0, g, size=g), minlength=g)
        counts = weights @ group_counts
        totals = np.einsum("g,gk->k", weights, group_sums.reshape(g, -1)).reshape(group_sums.shape[1:])
        np.divide(totals, counts[:, None, None], out=samples[iteration], where=counts[:, None, None] > 0)
    def intervals(values):
        if not len(values):
            return {key: None for key in BOOTSTRAP_METRICS}
        bounds = np.quantile(values, (.025, .975), axis=0, method="linear")
        return {key: bounds[:, index].tolist() for index, key in enumerate(BOOTSTRAP_METRICS)}
    results = {}
    for index, (cohort, mask) in enumerate(masks.items()):
        valid = np.isfinite(samples[:, index]).all(axis=(1, 2))
        draws = samples[valid, index]
        results[cohort] = {
            "n_samples": int(mask.sum()), "n_groups": int(np.unique(groups[mask]).size),
            "bootstrap_effective_replicates": int(valid.sum()),
            "bootstrap_empty_replicates": int((~valid).sum()),
            "models": {name: {"metrics": points[cohort][name], "ci95": intervals(draws[:, model_index])}
                       for model_index, name in enumerate(COMPARISONS)},
            "difference_b_minus_a": {
                "metrics": metric_difference(points[cohort]["A"], points[cohort]["B"]) if mask.any() else None,
                "ci95": intervals(draws[:, 2] - draws[:, 1]),
            },
        }
    return {"method": "paired group percentile bootstrap; row-weighted metrics",
            "group_source": "validation metadata.group_id from #5", "groups_per_draw": g,
            "replicates": replicates, "seed": seed, "rng": "numpy.random.PCG64", "confidence_level": .95,
            "percentile_method": "numpy.quantile linear; 2.5% and 97.5%",
            "shared_draws": "same validation group draw for baseline/A/B and all/first/followup cohorts",
            "empty_cohort_rule": "omit draws with zero cohort rows; report effective/empty counts; null if unavailable",
            "refit_or_reselect": False, "metrics": list(BOOTSTRAP_METRICS)}, results


def fit_model(train_x, train_y, validation_x, validation_y, names, seed, max_rounds, patience, threads):
    import lightgbm as lgb
    import numpy as np
    common = {"objective": "quantile", "metric": "quantile", "learning_rate": .05,
              "max_bin": 255, "feature_fraction": 1.0, "bagging_fraction": 1.0,
              "bagging_freq": 0, "lambda_l1": 0.0, "lambda_l2": 0.0,
              "seed": seed, "data_random_seed": seed, "feature_fraction_seed": seed,
              "bagging_seed": seed, "deterministic": True, "force_col_wise": True,
              "num_threads": threads, "verbosity": -1, "feature_pre_filter": False}
    train_set = lgb.Dataset(train_x, label=train_y, feature_name=names, free_raw_data=False)
    valid_set = lgb.Dataset(validation_x, label=validation_y, reference=train_set, free_raw_data=False)
    trials, best, best_predictions = [], None, None
    for trial_index, config in enumerate(SEARCH_SPACE):
        predictions, fits = [], []
        for q in QUANTILES:
            learning_curve = {}
            model = lgb.train({**common, **config, "alpha": q}, train_set,
                              num_boost_round=max_rounds, valid_sets=[valid_set], valid_names=["validation"],
                              callbacks=[lgb.record_evaluation(learning_curve),
                                         lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
            predictions.append(model.predict(validation_x, num_iteration=model.best_iteration, num_threads=threads))
            fits.append({"quantile": q, "best_iteration": model.best_iteration,
                         "retained_trees": model.current_iteration(),
                         "rounds_evaluated": len(learning_curve["validation"]["quantile"]),
                         "validation_raw_pinball_at_best": model.best_score["validation"]["quantile"]})
        ordered, crossing = order_quantiles(np.column_stack(predictions))
        metrics = evaluate(validation_y.tolist(), ordered.tolist())
        trial = {"index": trial_index, "parameters": config, "quantile_fits": fits,
                 "validation_metrics": metrics, "crossing_before_repair": crossing}
        trials.append(trial)
        print(f'  trial {trial_index}: validation mean pinball={metrics["mean_pinball_loss"]:.6f}', flush=True)
        if best is None or metrics["mean_pinball_loss"] < best["validation_metrics"]["mean_pinball_loss"]:
            best = trial  # Exact ties retain the first configuration; no refit on validation.
            best_predictions = ordered
    return {"common_parameters": common, "trials": trials, "selected_trial": best["index"],
            "selected_parameters": best["parameters"], "selected_quantile_fits": best["quantile_fits"],
            "validation_metrics": best["validation_metrics"],
            "crossing_before_repair": best["crossing_before_repair"]}, best_predictions


def render_report(r):
    scope = (f'전체 14개 샤드 중 첫 샤드 {r["source"]["input_files"][0]["rows"]:,}개 원본 대화 기록 행의 결정적 부분 자료이며 전체 자료를 대표하지 않는다.'
             if r["source"]["first_shard_only"] else
             f'manifest에 지정된 {len(r["source"]["input_files"])}개 입력 파일에서 준비된 표본 전체를 사용했다.')
    interval_note = ("실제 제출시각 의미가 검증되고 train에서 유효 관측이 있는 슬롯만 포함했다."
                     if r["interval_lags"] else
                     "제출시각 의미가 검증되고 train에서 유효 관측이 있는 슬롯이 없어 간격 수치와 상수인 간격 결측 지표를 제외했다.")
    lines = ["# WildChat 프롬프트 대 이력 GBDT 분위수 실험 (#7)", "",
             "핵심 가설: 길이+대화이력 GBDT(B)가 길이만 GBDT(A)보다 현재 응답의 출력 토큰을 더 잘 예측하는가? 비교 대상은 고정 분위수 기준선·A·B 세 개다. 의미 특징은 사용하지 않는다.", "",
             f'- 기록시각(UTC): {r["recorded_at_utc"]}; 실행 {r["elapsed_seconds"]:.2f}초; seed {r["seed"]}.',
             f'- **test 미평가**: {r["test"]["reason"]}. 최종 test 1회 평가 시점 승인이 필요하다.',
             "- 아래 성능은 튜닝·조기 종료에 사용한 validation 결과다. 독립 시험 성능이나 일반화 입증으로 해석하지 않는다.",
             "", "## 표본과 분할", "",
             f'- 실제 전처리 경로: `{r["prepared"]}`. manifest의 /tmp 경로는 과거 생성 위치다.',
             f'- WildChat-1M revision `{r["source"]["revision"]}`. #5의 준비된 표본을 그대로 사용했다. 추가 추출·행 제거는 없다.',
             f'- {scope}',
             '- 이 이슈의 실제 실행은 운영자의 자원·범위 결정으로 고정된 issue5-shard0를 사용한다.',
             "- 동일 hash·가시 대화 전체·첫 user 텍스트의 정확 일치를 추이적으로 묶은 #5 그룹을 유지한다. 그룹 최초 기록시각의 70%/85% 경계, 같은 시각은 뒤 구간, 경계 걸침 그룹 전체 제외 규칙을 재사용한다.",
             f'- 경계(UTC): {r["boundaries"]["train_validation_cutoff"]} / {r["boundaries"]["validation_test_cutoff"]}.',
             "- 실제 읽은 train·validation의 그룹/원본 행 교집합은 0이고 기록시각 범위는 겹치지 않는다. test의 수·범위는 manifest 메타데이터만 인용하며 파일 검증도 수행하지 않는다.",
             "", "| 분할 | 그룹 | 대화 기록 행 | 쌍 표본 | 첫 user 턴 | 기록시각 최소~최대(UTC) |",
             "|---|---:|---:|---:|---:|---|"]
    for split, p in r["partitions"].items():
        first = f'{p["first_turn_samples"]:,}' if "first_turn_samples" in p else "미열람"
        lines.append(f'| {split} | {p["groups"]:,} | {p["conversations"]:,} | {p["samples"]:,} | {first} | {p["recorded_at_min"]} ~ {p["recorded_at_max"]} |')
    lines += ["", "A·B·단순 기준은 validation의 모든 동일 행을 같은 순서로 평가한다. 첫 턴과 이력 부족 행도 유지하며, 불완전한 이전 user 턴은 lag 위치를 차지한다.",
              "", "## 입력·출력 분포와 관계", "",
              "분위수는 #6과 같은 경험적 CDF 역함수 ceil(n×q) 방식이다. 상관은 입력/출력 쌍 단위 기술 통계이며 인과관계가 아니다. Spearman 동률에는 평균 순위를 사용한다. 상수 열의 상관은 JSON null로 기록한다.", "",
              "| 분할·턴·항목 | 수 | 평균 | 표준편차(모집단) | 최소 | P10 | P50 | P90 | P99 | 최대 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, cohorts in r["descriptive_statistics_by_cohort"].items():
        for cohort, stats in cohorts.items():
            if stats is None:
                continue
            for side in ("input_tokens", "output_tokens"):
                d = stats[side]
                numbers = " | ".join(f"{d[key]:,.4f}" for key in ("n", "mean", "std_population", "min", "p10", "p50", "p90", "p99", "max"))
                lines.append(f"| {split} {cohort} {side} | {numbers} |")
    lines += ["", "| 분할·턴 | Pearson(원척도) | Spearman | Pearson(log1p 입력, log1p 출력) |", "|---|---:|---:|---:|"]
    for split, cohorts in r["descriptive_statistics_by_cohort"].items():
        for cohort, stats in cohorts.items():
            if stats is None:
                continue
            values = [stats["correlations"][key] for key in ("pearson_raw", "spearman_average_ties", "pearson_log1p")]
            lines.append(f'| {split} {cohort} | ' + ' | '.join("정의 불가" if v is None else f"{v:.6f}" for v in values) + ' |')
    lines += ["", "## 특징 및 누출 방지", ""]
    for model in ("A", "B"):
        lines.append(f'- {model} ({len(r["features"][model])}개): ' + ", ".join(f"`{name}`" for name in r["features"][model]) + ".")
    lines += [f'- 간격 lag에 사용한 슬롯: {r["interval_lags"]}. {interval_note} assistant 시각이나 행 시각으로 대체하지 않는다.',
              "- 현재 프롬프트 길이는 제출 직후 알려진 숫자다. B의 31개 특징은 모두 길이·이력 수치와 결측 지표다. 이전 인접 assistant 응답은 현재 프롬프트 전에 관측됐다고 가정한다. 원문·프롬프트 유형·주제·언어·임베딩·현재 응답·미래 턴·전체 대화 집계·메타데이터는 특징에 넣지 않는다.",
              "- 결측 lag는 NaN과 기존 결측 지표로 유지한다. 정규화·대체값 통계는 학습하지 않으며, LightGBM의 bin 경계는 train Dataset에서만 만든다. validation Dataset은 train을 reference로 사용한다.",
              "", "## 학습·선택과 분위수 교차", "",
              f'- A·B 각각 P10/P50/P90, 같은 seed {r["seed"]}, 같은 탐색 공간 {json.dumps(r["training"]["search_space"])}.',
              f'- 모델별 {r["training"]["candidates_per_model"]}설정 × 3분위수 = {r["training"]["fits_per_model"]}회. 각 최대 {r["training"]["max_rounds"]}회 boosting, learning_rate 0.05, validation 분위수 손실 {r["training"]["early_stopping_rounds"]}회 무개선 조기 종료(min_delta=0), 스레드 {r["training"]["threads"]}.',
              "- 각 분위수의 best iteration으로 validation을 예측한 뒤 음수를 0으로 자르고 행별 오름차순 정렬한다. 정렬 후 P10≤P50≤P90을 #6 metrics.evaluate에 전달한다. 교차율은 보정 전 원예측 기준이다. 정렬은 개별 분위수의 의미와 P50도 바꿀 수 있으며 보정 후 지표로 선택한다.",
              "- validation의 3분위수 평균 pinball 최소 설정을 A·B 각각 선택한다. 동률은 탐색 순서의 첫 설정을 선택한다. validation을 학습 데이터에 합쳐 재학습하지 않는다. 선택된 모델의 시험 예측·평가와 모델 바이너리 저장은 수행하지 않는다.",
              "- 단순 기준은 train 출력만으로 계산한 고정 P10/P50/P90: " + str(r["baseline"]["quantiles"]) + ".",
              "", "| 모델 | 후보 | 설정 | best iteration P10/P50/P90 | 검증 평균 pinball | 보정 전 교차율 | 선택 |",
              "|---|---:|---|---|---:|---:|---|"]
    for name, model in r["models"].items():
        for trial in model["trials"]:
            fits = "/".join(str(f["best_iteration"]) for f in trial["quantile_fits"])
            lines.append(f'| {name} | {trial["index"]} | {trial["parameters"]} | {fits} | {trial["validation_metrics"]["mean_pinball_loss"]:.6f} | {trial["crossing_before_repair"]["raw_crossing_rate"]:.6f} | {"예" if trial["index"] == model["selected_trial"] else ""} |')
    boot = r["bootstrap"]
    lines += ["", "## 전체·첫 턴·후속 턴의 validation 성능과 95% CI", "",
              "주 지표는 평균 pinball이다. pinball은 max(q(y−p), (q−1)(y−p))이며 P10/P50/P90의 산술평균을 사용한다. MAE는 P50 기준이다. P10~P90 커버리지는 양 끝을 포함하고 평균 폭은 P90−P10이다.", "",
              f'- #5 validation의 {boot["groups_per_draw"]:,}개 group_id에서 같은 수의 그룹을 복원 재표집한다. seed {boot["seed"]}, {boot["rng"]}, {boot["replicates"]:,}회, 2.5%/97.5% percentile(선형 보간)로 95% CI를 계산한다.',
              "- 선택된 그룹의 모든 행을 중복 횟수만큼 유지하며 표본별 평균(그룹 합/표본 수)을 계산한다. 세 모델과 세 턴 구분에 같은 재표집을 사용하고, B−A는 각 반복 내 차이로 CI를 계산한다. 그룹별 지표를 동일 가중 평균하거나 행을 독립 재표집하지 않는다.",
              "- 설정 선택은 전체 validation만 사용한다. 같은 선택 모델의 예측을 turn_index==0(첫 턴), >0(후속 턴)으로 분리한다. 부트스트랩 중 재학습·재선택은 없다. 따라서 CI는 선택된 예측과 이 검증 표본에 조건부이며 선택 불확실성·낙관 편향을 보정하지 않는다.",
              "- 특정 턴 표본이 0인 반복은 해당 구분에서 제외하고 유효 반복 수를 기록한다. 관측 표본 자체가 없으면 지표·CI는 null이다. 각 셀은 점추정 [95% CI]이며 B−A 행은 네 번째 모델이 아닌 대응 차이다."]
    labels = {"all": "전체", "first_turn": "첫 턴 (turn_index==0)", "followup_turn": "후속 턴 (turn_index>0)"}
    model_labels = {"baseline": "고정 분위수 기준선", "A": "A(길이만)", "B": "B(길이+이력)"}
    for cohort, result in r["validation_by_cohort"].items():
        lines += ["", f'### {labels[cohort]}', "",
                  f'{result["n_samples"]:,}쌍 · {result["n_groups"]:,}그룹 · 유효 부트스트랩 {result["bootstrap_effective_replicates"]:,}/{boot["replicates"]:,}회(빈 표본 {result["bootstrap_empty_replicates"]:,}회).', "",
                  "| 비교 | 평균 pinball (주 지표) | P50 MAE | P10~P90 커버리지 | 평균 구간 폭 |",
                  "|---|---:|---:|---:|---:|"]
        entries = [(model_labels[name], model) for name, model in result["models"].items()]
        entries.append(("B−A", result["difference_b_minus_a"]))
        for name, entry in entries:
            cells = []
            for metric in BOOTSTRAP_METRICS:
                if entry["metrics"] is None:
                    cells.append("해당 표본 없음")
                elif entry["ci95"][metric] is None:
                    cells.append(f'{entry["metrics"][metric]:.6f} [CI 산출 불가]')
                else:
                    low, high = entry["ci95"][metric]
                    cells.append(f'{entry["metrics"][metric]:.6f} [{low:.6f}, {high:.6f}]')
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    a = r["models"]["A"]["validation_metrics"]["mean_pinball_loss"]
    delta = r["difference_b_minus_a"]["mean_pinball_loss"]
    percent = f"({delta / a * 100:+.2f}%)" if a else "(A=0으로 비율 정의 불가)"
    lines += ["", f"B−A 평균 pinball 차이는 {delta:+.6f} {percent}이다. " +
              ("이 표본의 검증 분할에서 이력 추가 후 손실이 감소했다." if delta < 0 else "이 표본의 검증 분할에서 이력 추가의 손실 개선은 관측되지 않았다.")]
    for cohort in ("first_turn", "followup_turn"):
        difference = r["validation_by_cohort"][cohort]["difference_b_minus_a"]
        bounds = difference["ci95"]["mean_pinball_loss"]
        if difference["metrics"] is not None and bounds is not None:
            delta = difference["metrics"]["mean_pinball_loss"]
            lines += [f'{labels[cohort]}의 B−A 평균 pinball은 {delta:+.6f}, 95% CI [{bounds[0]:.6f}, {bounds[1]:.6f}]이다. ' +
                      ("손실 감소가 관측됐다. " if delta < 0 else "손실 감소는 관측되지 않았다. ") +
                      ("CI는 0을 포함한다." if bounds[0] <= 0 <= bounds[1] else "CI는 0을 포함하지 않는다.")]
    lines += ["MAE·pinball 차이는 음수일수록 오차 감소다. 커버리지는 명목 0.8과 비교하고 폭은 커버리지와 함께 해석한다. 원래 #6의 log 오차와 분위수별 pinball 점추정도 JSON의 각 턴 구분에 보존한다. 후보 탐색 표는 전체 validation의 선택 과정 기록이다.",
              "", "## 재현 정보와 한계", "",
              "- 실행 패키지: " + ", ".join(f"{k}={v}" for k, v in r["versions"].items()) + ".",
              "- SciPy는 LightGBM 4.7.0 패키지 메타데이터의 필수 런타임 의존성이므로 버전 고정을 유지한다. 부트스트랩은 NumPy만 사용한다.",
              "- 전처리 패키지: " + json.dumps(r["source"]["preparation_versions"], ensure_ascii=False) + ".",
              "- 전체 LightGBM 파라미터·각 분위수 best iteration·후보별 지표·파일 SHA-256은 [experiment.json](experiment.json)에 기록한다. best iteration의 트리만 예측에 사용한다.",
              "- 재현 명령(실행에 사용한 Python 경로; 다른 환경에서는 의존성이 설치된 Python으로 대체):", "", "```sh",
              r["command"], "python3 -m unittest discover -s wildchat/tests",
              "/workspace/venvs/wildchat/bin/python -m unittest discover -s wildchat/tests",
              "python3 scripts/verify.py", "```", "",
              "- o200k_base로 user/assistant 가시 텍스트를 독립 계수한 재구성 토큰이다. 시스템·역할·툴·숨은 토큰, 제공자별 tokenizer와 서비스의 실제 사용량을 재현하지 않는다.",
              "- 기록시각은 행의 마지막 턴 시각이며 실제 개별 요청 도착시각 순서를 보장하지 않는다. user 제출시각이 없어 실제 턴 간격의 효과는 평가하지 못한다. 다른 샤드와 연결되는 중복도 확인하지 못했다.",
              "- 튜닝에 재사용한 검증 결과, 첫 샤드의 시기·분포·길이 편향, 분할 내 중복의 반복 가중 효과가 있다. 전체 데이터·독립 시험의 효과, 실제 GPU/비용 개선, 인과 효과로 확장하지 않는다.",
              "- 하이퍼파라미터 선택·조기 종료·보고가 모두 같은 validation이라 절대 수치는 낙관적일 수 있음(A·B 동일 조건). 그룹 부트스트랩 CI도 이 한계를 제거하지 않는다.",
              "- 미결정: 최종 test 1회 평가 시점 승인. 이 구현은 승인 전 test를 읽을 실행 경로를 제공하지 않는다.", ""]
    return "\n".join(lines)


def run(args):
    import numpy as np
    started = time.monotonic()
    if (args.max_rounds <= 0 or args.early_stopping_rounds <= 0 or not 1 <= args.threads <= 8 or
        args.bootstrap_replicates <= 0 or args.seed < 0):
        raise ValueError("positive rounds/patience/bootstrap replicates, nonnegative seed and 1..8 threads required")
    prepared, output = args.prepared.resolve(), args.output.resolve()
    if output == prepared or output in prepared.parents or prepared in output.parents:
        raise ValueError("output and prepared data directories must be separate")
    manifest_raw = (prepared / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    lags = manifest["lags"]
    if not isinstance(lags, int) or lags <= 0:
        raise ValueError("positive preparation lag count required")
    verified = manifest["verified_user_timestamps"]
    candidates = feature_names(lags, range(1, lags + 1) if verified else ())["B"]
    print("Loading train (test.jsonl is forbidden)", flush=True)
    train = load_partition(prepared, "train", manifest, candidates)
    interval_lags = select_interval_lags(train, candidates, lags, verified)
    names = feature_names(lags, interval_lags)
    train.x = train.x[:, [candidates.index(name) for name in names["B"]]]
    print("Loading validation", flush=True)
    validation = load_partition(prepared, "validation", manifest, names["B"])
    check_disjoint(train, validation)
    models, predictions = {}, {}
    for name in ("A", "B"):
        columns = [names["B"].index(feature) for feature in names[name]]
        print(f"Training {name}: {len(columns)} numeric features", flush=True)
        models[name], predictions[name] = fit_model(
            train.x[:, columns], train.y, validation.x[:, columns], validation.y,
            names[name], args.seed, args.max_rounds, args.early_stopping_rounds, args.threads)
    quantiles = baseline(train.y.tolist())
    baseline_metrics = evaluate(validation.y.tolist(), itertools.repeat(quantiles, len(validation.y)))
    predictions["baseline"] = np.tile(quantiles, (len(validation.y), 1))
    print(f"Paired #5-group bootstrap: {args.bootstrap_replicates} replicates, seed {args.seed}", flush=True)
    bootstrap_info, by_cohort = cluster_bootstrap(validation.y, predictions, validation.row_groups,
                                                 validation.turn_indices, args.seed, args.bootstrap_replicates)
    descriptive_by_cohort = {
        split: {cohort: descriptive_stats(p.x[mask, 0], p.y[mask]) if mask.any() else None
                for cohort, mask in cohort_masks(p.turn_indices).items()}
        for split, p in (("train", train), ("validation", validation))}
    result = {
        "hypothesis": "length+history GBDT B predicts current output tokens better than length-only GBDT A",
        "comparison_models": list(COMPARISONS), "semantic_features": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "elapsed_seconds": time.monotonic() - started, "prepared": str(prepared),
        "command": shlex.join([sys.executable, "wildchat/train.py", "--prepared", str(prepared),
                               "--output", str(args.output), "--seed", str(args.seed),
                               "--max-rounds", str(args.max_rounds), "--early-stopping-rounds",
                               str(args.early_stopping_rounds), "--threads", str(args.threads),
                               "--bootstrap-replicates", str(args.bootstrap_replicates)]),
        "versions": {"python": sys.version.split()[0], **{name: importlib.metadata.version(name)
                      for name in ("lightgbm", "numpy", "scipy")}},
        "source": {"revision": manifest["revision"], "input_files": manifest["input_files"],
                   "first_shard_only": manifest["first_shard_only"],
                   "sampling_decision": "entire supplied #5 train/validation; no additional sampling",
                   "preparation_versions": manifest["versions"],
                   "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                   "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__), Path(__file__).with_name("metrics.py"))}},
        "target_definition": {"field": "target.output_tokens", "unit": "reconstructed visible assistant text tokens",
                              "tokenizer": "tiktoken/o200k_base", "actual_provider_usage": False,
                              "prompt_scope": "current user text only; not full provider context"},
        "boundaries": manifest["boundaries"],
        "partitions": {"train": train.summary, "validation": validation.summary,
                       "test": {**manifest["partitions"]["test"], "source": "manifest metadata only; not opened"}},
        "split_checks": {"train_validation_group_intersection": 0, "train_validation_source_row_intersection": 0,
                         "recorded_time_ranges_disjoint": True, "test_checked": False},
        "test": {"evaluated": False, "read": False, "reason": TEST_REASON},
        "features": names, "interval_lags": interval_lags,
        "descriptive_statistics": {split: cohorts["all"] for split, cohorts in descriptive_by_cohort.items()},
        "descriptive_statistics_by_cohort": descriptive_by_cohort,
        "training": {"fit_split": "train", "selection_split": "validation", "evaluation_split": "validation",
                     "search_space": list(SEARCH_SPACE), "candidates_per_model": len(SEARCH_SPACE),
                     "fits_per_model": len(SEARCH_SPACE) * len(QUANTILES), "max_rounds": args.max_rounds,
                     "early_stopping_rounds": args.early_stopping_rounds, "early_stopping_min_delta": 0,
                     "threads": args.threads, "selection_metric": "mean_pinball_loss after quantile repair",
                     "tie_rule": "first candidate", "refit_on_validation": False,
                     "missing_values": "native NaN; no imputation/scaling", "bin_fit_split": "train",
                     "quantiles": list(QUANTILES), "crossing_rule": "clip negative to zero, row-wise ascending sort",
                     "models_saved": False},
        "baseline": {"fit_split": "train", "quantile_method": "inverse empirical CDF, ceil(n*q)",
                     "quantiles": dict(zip(("p10", "p50", "p90"), quantiles)), "validation_metrics": baseline_metrics},
        "models": models,
        "difference_b_minus_a": metric_difference(models["A"]["validation_metrics"], models["B"]["validation_metrics"]),
        "bootstrap": bootstrap_info, "validation_by_cohort": by_cohort,
        "limitations": ["하이퍼파라미터 선택·조기 종료·보고가 모두 같은 validation이라 절대 수치는 낙관적일 수 있음(A·B 동일 조건)",
                        "Group bootstrap CIs condition on fixed selected predictions; no refit or reselection; selection uncertainty is not covered",
                        "Fixed first-shard sample; reconstructed o200k_base visible text tokens, not provider usage"],
        "decisions_needed": ["최종 test 1회 평가 시점 승인"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(result), encoding="utf-8")
    print(f"Wrote experiment.json and report.md to {output}; test was not opened", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000,
                        help="paired validation #5-group bootstrap draws; uses --seed")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, ImportError, OSError, KeyError) as error:
        parser.exit(2, f"train: {error}\n")


if __name__ == "__main__":
    main()
