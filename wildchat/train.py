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
    return Partition(x, y, summary, groups, conversations)


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
    trials, best = [], None
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
    return {"common_parameters": common, "trials": trials, "selected_trial": best["index"],
            "selected_parameters": best["parameters"], "selected_quantile_fits": best["quantile_fits"],
            "validation_metrics": best["validation_metrics"],
            "crossing_before_repair": best["crossing_before_repair"]}


def render_report(r):
    scope = (f'전체 14개 샤드 중 첫 샤드 {r["source"]["input_files"][0]["rows"]:,}개 원본 대화 기록 행의 결정적 부분 자료이며 전체 자료를 대표하지 않는다.'
             if r["source"]["first_shard_only"] else
             f'manifest에 지정된 {len(r["source"]["input_files"])}개 입력 파일에서 준비된 표본 전체를 사용했다.')
    interval_note = ("실제 제출시각 의미가 검증되고 train에서 유효 관측이 있는 슬롯만 포함했다."
                     if r["interval_lags"] else
                     "제출시각 의미가 검증되고 train에서 유효 관측이 있는 슬롯이 없어 간격 수치와 상수인 간격 결측 지표를 제외했다.")
    lines = ["# WildChat 프롬프트 대 이력 GBDT 분위수 실험 (#7)", "",
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
              "| 분할·항목 | 수 | 평균 | 표준편차(모집단) | 최소 | P10 | P50 | P90 | P99 | 최대 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, stats in r["descriptive_statistics"].items():
        for side in ("input_tokens", "output_tokens"):
            d = stats[side]
            numbers = " | ".join(f"{d[key]:,.4f}" for key in ("n", "mean", "std_population", "min", "p10", "p50", "p90", "p99", "max"))
            lines.append(f"| {split} {side} | {numbers} |")
    lines += ["", "| 분할 | Pearson(원척도) | Spearman | Pearson(log1p 입력, log1p 출력) |", "|---|---:|---:|---:|"]
    for split, stats in r["descriptive_statistics"].items():
        values = [stats["correlations"][key] for key in ("pearson_raw", "spearman_average_ties", "pearson_log1p")]
        lines.append(f'| {split} | ' + ' | '.join("정의 불가" if v is None else f"{v:.6f}" for v in values) + ' |')
    lines += ["", "## 특징 및 누출 방지", ""]
    for model in ("A", "B"):
        lines.append(f'- {model} ({len(r["features"][model])}개): ' + ", ".join(f"`{name}`" for name in r["features"][model]) + ".")
    lines += [f'- 간격 lag에 사용한 슬롯: {r["interval_lags"]}. {interval_note} assistant 시각이나 행 시각으로 대체하지 않는다.',
              "- 현재 프롬프트 길이는 제출 직후 알려진 숫자다. 이전 인접 assistant 응답은 현재 프롬프트 전에 관측됐다고 가정한다. 원문·임베딩·현재 응답·미래 턴·전체 대화 집계·메타데이터는 특징에 넣지 않는다.",
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
    lines += ["", "## 동일 validation 표본의 성능", "",
              "MAE와 log 오차는 P50 기준이며 log 오차는 mean(|log1p(y)−log1p(P50)|)다. pinball은 max(q(y−p), (q−1)(y−p)), 평균은 3분위수 산술평균이다. 커버리지는 양 끝을 포함하고 평균 폭은 P90−P10이다.",
              "", "| 모델 | MAE | log 오차 | pinball P10 | P50 | P90 | 평균 pinball | P10~P90 커버리지 | 평균 폭 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    metrics = {"단순 기준": r["baseline"]["validation_metrics"],
               **{name: model["validation_metrics"] for name, model in r["models"].items()},
               "B−A": r["difference_b_minus_a"]}
    for name, m in metrics.items():
        numbers = [m["mae_p50"], m["mean_absolute_log1p_error_p50"], *m["pinball_loss"].values(),
                   m["mean_pinball_loss"], m["p10_p90_coverage"], m["mean_interval_width"]]
        lines.append(f'| {name} | ' + ' | '.join(f"{v:.6f}" for v in numbers) + ' |')
    a = r["models"]["A"]["validation_metrics"]["mean_pinball_loss"]
    delta = r["difference_b_minus_a"]["mean_pinball_loss"]
    percent = f"({delta / a * 100:+.2f}%)" if a else "(A=0으로 비율 정의 불가)"
    lines += ["", f"B−A 평균 pinball 차이는 {delta:+.6f} {percent}이다. " +
              ("이 표본의 검증 분할에서 이력 추가 후 손실이 감소했다." if delta < 0 else "이 표본의 검증 분할에서 이력 추가의 손실 개선은 관측되지 않았다."),
              "MAE·log·pinball 차이는 음수일수록 오차 감소다. 커버리지는 명목 0.8과 비교하고 폭은 커버리지와 함께 해석한다. 반복·연결 표본이 남아 있으므로 행을 독립 관측으로 보거나 통계적 유의성을 주장하지 않는다.",
              "", "## 재현 정보와 한계", "",
              "- 실행 패키지: " + ", ".join(f"{k}={v}" for k, v in r["versions"].items()) + ".",
              "- 전처리 패키지: " + json.dumps(r["source"]["preparation_versions"], ensure_ascii=False) + ".",
              "- 전체 LightGBM 파라미터·각 분위수 best iteration·후보별 지표·파일 SHA-256은 [experiment.json](experiment.json)에 기록한다. best iteration의 트리만 예측에 사용한다.",
              "- 재현 명령(실행에 사용한 Python 경로; 다른 환경에서는 의존성이 설치된 Python으로 대체):", "", "```sh",
              r["command"], "python3 -m unittest discover -s wildchat/tests",
              "/workspace/venvs/wildchat/bin/python -m unittest discover -s wildchat/tests",
              "python3 scripts/verify.py", "```", "",
              "- o200k_base로 user/assistant 가시 텍스트를 독립 계수한 재구성 토큰이다. 시스템·역할·툴·숨은 토큰, 제공자별 tokenizer와 서비스의 실제 사용량을 재현하지 않는다.",
              "- 기록시각은 행의 마지막 턴 시각이며 실제 개별 요청 도착시각 순서를 보장하지 않는다. user 제출시각이 없어 실제 턴 간격의 효과는 평가하지 못한다. 다른 샤드와 연결되는 중복도 확인하지 못했다.",
              "- 튜닝에 재사용한 검증 결과, 첫 샤드의 시기·분포·길이 편향, 분할 내 중복의 반복 가중 효과가 있다. 전체 데이터·독립 시험의 효과, 실제 GPU/비용 개선, 인과 효과로 확장하지 않는다.",
              "- 미결정: 최종 test 1회 평가 시점 승인. 이 구현은 승인 전 test를 읽을 실행 경로를 제공하지 않는다.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    if args.max_rounds <= 0 or args.early_stopping_rounds <= 0 or not 1 <= args.threads <= 8:
        raise ValueError("positive rounds/patience and 1..8 threads required")
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
    models = {}
    for name in ("A", "B"):
        columns = [names["B"].index(feature) for feature in names[name]]
        print(f"Training {name}: {len(columns)} numeric features", flush=True)
        models[name] = fit_model(train.x[:, columns], train.y, validation.x[:, columns], validation.y,
                                 names[name], args.seed, args.max_rounds, args.early_stopping_rounds, args.threads)
    quantiles = baseline(train.y.tolist())
    baseline_metrics = evaluate(validation.y.tolist(), itertools.repeat(quantiles, len(validation.y)))
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "elapsed_seconds": time.monotonic() - started, "prepared": str(prepared),
        "command": shlex.join([sys.executable, "wildchat/train.py", "--prepared", str(prepared),
                               "--output", str(args.output), "--seed", str(args.seed),
                               "--max-rounds", str(args.max_rounds), "--early-stopping-rounds",
                               str(args.early_stopping_rounds), "--threads", str(args.threads)]),
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
        "descriptive_statistics": {split: descriptive_stats(p.x[:, 0], p.y)
                                   for split, p in (("train", train), ("validation", validation))},
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
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, ImportError, OSError, KeyError) as error:
        parser.exit(2, f"train: {error}\n")


if __name__ == "__main__":
    main()
