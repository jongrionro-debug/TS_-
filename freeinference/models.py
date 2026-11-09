"""Train-only A/L/B/C quantile selection, refit and aggregate validation report."""
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow

from .baselines import Baselines, PREDICTION_COLUMNS, QUANTILES, build_history
from .evaluate import (BOOTSTRAP_B, BOOTSTRAP_SEED, METRICS, _set_counts,
                       evaluate_predictions, paired_bootstrap, _losses)
from .features import (C_EXTRA, FEATURE_SETS, REQUEST_COLUMNS, SESSION_COLUMNS as FEATURE_SESSION_COLUMNS,
                       build_cross_session_features, build_features)
from .loader import FIRST_WEEK, REVISION, data_dir, load_table, weeks
from .split import (FIT_WEEKS, HOLDOUT_START_MS, HOLDOUT_WEEK, SESSION_COLUMNS,
                    TRAIN_WEEKS, VALIDATION_START_MS, VALIDATION_WEEKS, split_masks)

SEED = 20261109
NUM_THREADS = 4
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 30
TOP_GAIN = 10
CATEGORICAL = ("model", "provider", "harness", "stream", "step_trigger",
               "waited_for", "transition", "previous_finish_reason")
# Identical, predeclared candidates for every condition and quantile. Ties use
# the first candidate; validation is never used to select or stop a model.
GRID = tuple({"num_leaves": leaves, "min_data_in_leaf": minimum,
              "learning_rate": .05, "target_transform": transform}
             for leaves, minimum in ((15, 100), (31, 300))
             for transform in ("identity", "log1p"))
PARAMETERS = {"objective": "quantile", "metric": "None", "seed": SEED,
              "deterministic": True, "force_col_wise": True,
              "num_threads": NUM_THREADS, "verbosity": -1}
OUTPUT = Path(__file__).parent / "outputs" / "models"
PREVIOUS_REVISION = "f2da61698358ac3a47171a6a8b133bd3c6f36e82"  # PR #39
REGRESSION_REVISION = "39dedfd"  # PR #41: unchanged A/L/B 3000-round reference
PREVIOUS_PATH = "freeinference/outputs/models/models.json"


def pinball(y, prediction, q):
    residual = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.maximum(q * residual, (q - 1) * residual).mean())


def original_scale(prediction, transform):
    values = np.asarray(prediction, dtype=float)
    if transform == "log1p":
        values = np.expm1(values)
    elif transform != "identity":
        raise ValueError("Unknown target transform")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite original-scale prediction")
    return values


class FeatureEncoder:
    """Stage-training vocabulary only; unknown and absent categories stay NA."""

    def fit(self, frame):
        self.columns = tuple(frame.columns)
        self.categories = {name: list(pd.unique(frame[name].dropna()))
                           for name in self.columns if name in CATEGORICAL}
        return self

    def transform(self, frame):
        if tuple(frame.columns) != self.columns:
            raise ValueError("Input columns must exactly match FEATURE_SETS")
        encoded = pd.DataFrame(index=frame.index)
        for name in self.columns:
            if name in self.categories:
                known = frame[name].where(frame[name].isin(self.categories[name]))
                encoded[name] = pd.Categorical(known, categories=self.categories[name])
            else:
                encoded[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        return encoded


def _features(features, index=None):
    if set(features) != set(FEATURE_SETS):
        raise ValueError("Exactly A/L/B/C features are required")
    for condition, columns in FEATURE_SETS.items():
        if tuple(features[condition].columns) != columns:
            raise ValueError("Input columns must exactly match FEATURE_SETS")
        if index is not None and not features[condition].index.equals(index):
            raise ValueError("Features must align with requests")


def _targets(rows, mask):
    values = rows.loc[mask, "completion_tokens"].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Nonempty finite non-negative stage targets are required")
    return values


class QuantileModels:
    """Select on fit/holdout; refit using only train_target; predict without y."""

    def fit(self, rows, features):
        _features(features, rows.index)
        masks = split_masks(rows)
        y_fit = _targets(rows, masks.fit_target)
        y_holdout = _targets(rows, masks.holdout_target)
        y_train = _targets(rows, masks.train_target)
        self.models, self.encoders, self.selection, self.importance = {}, {}, {}, {}
        for condition, frame in features.items():
            encoder = FeatureEncoder().fit(frame.loc[masks.fit_target])
            x_fit = encoder.transform(frame.loc[masks.fit_target])
            x_holdout = encoder.transform(frame.loc[masks.holdout_target])
            final_encoder = FeatureEncoder().fit(frame.loc[masks.train_target])
            x_train = final_encoder.transform(frame.loc[masks.train_target])
            self.encoders[condition] = final_encoder
            self.models[condition], self.selection[condition], self.importance[condition] = {}, {}, {}
            # Safe internal names: some feature names contain spaces or '*'.
            names = [f"f{i}" for i in range(len(frame.columns))]
            categorical = [i for i, name in enumerate(frame.columns) if name in CATEGORICAL]
            for q, label in zip(QUANTILES, PREDICTION_COLUMNS):
                candidates = []
                for setting in GRID:
                    transform = setting["target_transform"]
                    params = {**PARAMETERS, **{k: v for k, v in setting.items()
                                              if k != "target_transform"}, "alpha": q}
                    target = np.log1p(y_fit) if transform == "log1p" else y_fit
                    holdout_target = np.log1p(y_holdout) if transform == "log1p" else y_holdout
                    training = lgb.Dataset(x_fit, label=target, feature_name=names,
                                           categorical_feature=categorical)
                    holdout = lgb.Dataset(x_holdout, label=holdout_target, reference=training,
                                          feature_name=names, categorical_feature=categorical)

                    def score(prediction, dataset):
                        # The original labels are retained without float32 Dataset
                        # roundoff. Only this holdout is passed to lgb.train.
                        return "original_pinball", pinball(
                            y_holdout, original_scale(prediction, transform), q), False

                    evaluated_rounds = 0

                    def record_round(env):
                        # lgb.train returns a booster truncated to best_iteration;
                        # count evaluations before early_stopping (order=30) runs.
                        nonlocal evaluated_rounds
                        evaluated_rounds = env.iteration + 1

                    record_round.order = 20
                    model = lgb.train(params, training, num_boost_round=MAX_ROUNDS,
                                      valid_sets=[holdout], feval=score,
                                      callbacks=[record_round, lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                                                   first_metric_only=True, verbose=False)])
                    iterations = int(model.best_iteration)
                    loss = pinball(y_holdout, original_scale(model.predict(
                        x_holdout, num_iteration=iterations, num_threads=NUM_THREADS), transform), q)
                    candidates.append({"settings": dict(setting), "iterations": iterations,
                                       "best_iteration": iterations,
                                       "evaluated_rounds": evaluated_rounds,
                                       "hit_max_rounds": evaluated_rounds >= MAX_ROUNDS,
                                       "holdout_pinball": loss})
                best = min(candidates, key=lambda candidate: candidate["holdout_pinball"])
                self.selection[condition][label] = {**best, "candidates": candidates}
                setting = best["settings"]
                target = np.log1p(y_train) if setting["target_transform"] == "log1p" else y_train
                params = {**PARAMETERS, **{k: v for k, v in setting.items()
                                          if k != "target_transform"}, "alpha": q}
                model = lgb.train(params, lgb.Dataset(x_train, label=target, feature_name=names,
                                                     categorical_feature=categorical),
                                  num_boost_round=best["iterations"])
                self.models[condition][label] = model
                gain = model.feature_importance(importance_type="gain")
                total = float(gain.sum())
                ranked = sorted(zip(frame.columns, map(float, gain)), key=lambda item: -item[1])
                self.importance[condition][label] = {
                    "total_gain": total,
                    "top": [{"feature": name, "gain": value,
                             "fraction": value / total if total else 0.}
                            for name, value in ranked[:TOP_GAIN]]}
                print(f"{condition}/{label}: {setting}, rounds={best['iterations']}, "
                      f"holdout pinball={best['holdout_pinball']:.6f}", flush=True)
        return self

    def predict(self, features):
        _features(features)
        predictions, crossing = {}, {}
        for condition, frame in features.items():
            encoded = self.encoders[condition].transform(frame)
            values = np.column_stack([
                original_scale(self.models[condition][label].predict(encoded, num_threads=NUM_THREADS),
                               self.selection[condition][label]["settings"]["target_transform"])
                for label in PREDICTION_COLUMNS])
            crossing[condition] = float((np.diff(values, axis=1) < 0).any(axis=1).mean())
            predictions[condition] = pd.DataFrame(np.sort(values, axis=1), index=frame.index,
                                                   columns=PREDICTION_COLUMNS)
        return predictions, crossing


def comparisons(rows, predictions):
    """Every call uses the existing shared seeded session draws and metrics."""
    return {name: paired_bootstrap(rows, {key: predictions[key] for key in (reference, *targets)},
                                    reference=reference, cluster_columns=SESSION_COLUMNS,
                                    B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
            for name, reference, targets in (
                ("B_minus_L", "L", ("B",)), ("L_minus_A", "A", ("L",)),
                ("C_minus_B", "B", ("C",)), ("C_minus_L", "L", ("C",)),
                ("conditions_minus_fixed", "fixed", ("A", "L", "B", "C")),
                ("conditions_minus_previous_completion", "previous_completion", ("A", "L", "B", "C")))}


COMPARISONS = (("B_minus_L", "B", "L"), ("L_minus_A", "L", "A"),
               ("C_minus_B", "C", "B"), ("C_minus_L", "C", "L"))


def account_subsets(rows, masks):
    """Membership and ranks depend only on train membership and validation counts."""
    if rows.account.isna().any():
        raise ValueError("Account keys must be non-null")
    validation = rows.loc[masks.validation]
    seen = validation.account.isin(rows.loc[masks.train, "account"])
    # Stable first-occurrence tie break uses neither targets nor predictions.
    counts = validation.account.value_counts(sort=False).sort_values(ascending=False, kind="stable")
    return {"train_seen": seen, "new_account": ~seen,
            "exclude_top_1": ~validation.account.isin(counts.index[:1]),
            "exclude_top_5": ~validation.account.isin(counts.index[:5])}


def account_mean_pinball(rows, predictions):
    counts = rows.groupby("account", sort=False).account.transform("size").to_numpy()
    return {name: float(np.average(_losses(rows.completion_tokens.to_numpy(dtype=float), frame)[:, 3],
                                   weights=1 / counts)) for name, frame in predictions.items()}


def robustness_results(rows, masks, predictions, session_bootstrap):
    validation = rows.loc[masks.validation].copy()
    validation["utc_day"] = pd.to_datetime(validation.start_ms, unit="ms", utc=True).dt.floor("D")
    subsets = account_subsets(rows, masks)
    def calculate(frame, frames, clusters):
        return {name: paired_bootstrap(frame, {reference: frames[reference], target: frames[target]},
                                       reference=reference, cluster_columns=(clusters,),
                                       B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
                for name, target, reference in COMPARISONS}
    account = calculate(validation, predictions, "account")
    day = calculate(validation, predictions, "utc_day")
    scores = evaluate_predictions(validation, predictions)["overall"]["metrics"]
    result = {"definition": "UTC start day; account and session weights are inverse group row counts; no refitting",
              "account_mean_pinball": account_mean_pinball(validation, predictions),
              "comparisons": {}, "subsets": {}}
    for name, target, reference in COMPARISONS:
        entry = {"difference": {w: scores[target][w]["mean_pinball"] - scores[reference][w]["mean_pinball"]
                                for w in ("request", "session")}, "clusters": {}}
        for label, source in (("session", session_bootstrap), ("account", account), ("utc_day", day)):
            data = source[name]
            group = data["groups"]["overall"]
            entry["clusters"][label] = {"count": data["clusters"], "valid_replicates": group["valid_replicates"],
                "ci95": {w: group["differences"][target][w]["mean_pinball"] for w in ("request", "session")}}
        entry["interpretation"] = ("군집 방식에 견고" if all(v["ci95"]["request"][1] < 0
                                     for v in entry["clusters"].values()) else "군집 방식에 따라 차이 미확인")
        result["comparisons"][name] = entry
    for label, mask in subsets.items():
        frame = validation.loc[mask]
        if frame.empty:
            result["subsets"][label] = {"rows": 0, "excluded_row_fraction": 1.0, "comparisons": {}}
            continue
        frames = {name: pred.loc[mask] for name, pred in predictions.items()}
        metrics = evaluate_predictions(frame, frames)["overall"]
        boots = calculate(frame, frames, "account")
        comparisons = {}
        for name, target, reference in COMPARISONS:
            boot = boots[name]
            group = boot["groups"]["overall"]
            diff = {w: metrics["metrics"][target][w]["mean_pinball"] - metrics["metrics"][reference][w]["mean_pinball"]
                    for w in ("request", "session")}
            overall = result["comparisons"][name]["difference"]["request"]
            comparisons[name] = {"difference": diff, "sign_changed": bool(np.sign(diff["request"]) != np.sign(overall)),
                                 "clusters": boot["clusters"], "valid_replicates": group["valid_replicates"],
                                 "all_metrics_ci95": group["differences"][target],
                                 "ci95": {w: group["differences"][target][w]["mean_pinball"] for w in ("request", "session")}}
        result["subsets"][label] = {"rows": len(frame), "excluded_row_fraction": float(1 - len(frame) / len(validation)),
                                    "metrics": metrics, "account_mean_pinball": account_mean_pinball(frame, frames),
                                    "comparisons": comparisons}
    return result


def primary_result(metrics, bootstrap):
    scores = metrics["overall"]["metrics"]
    intervals = bootstrap["B_minus_L"]["groups"]["overall"]["differences"]["B"]
    ci = intervals["request"]["mean_pinball"]
    interpretation = ("validation에서 B가 L보다 개선" if ci[1] < 0 else
                      "차이 미확인" if ci[0] <= 0 <= ci[1] else "validation에서 B가 L보다 악화")
    return {"metric": "mean_pinball", "comparison": "B−L", "primary_weighting": "request",
            "difference": {w: scores["B"][w]["mean_pinball"] - scores["L"][w]["mean_pinball"]
                           for w in ("request", "session")},
            "ci95": {w: intervals[w]["mean_pinball"] for w in ("request", "session")},
            "interpretation": interpretation, "final_conclusion": "reserved for approved final test issue"}


def supplementary_results(metrics, bootstrap):
    results = {}
    scores = metrics["overall"]["metrics"]
    for reference in ("B", "L"):
        name = f"C_minus_{reference}"
        intervals = bootstrap[name]["groups"]["overall"]["differences"]["C"]
        ci = intervals["request"]["mean_pinball"]
        results[name] = {
            "role": "supplementary; primary decision remains B−L", "metric": "mean_pinball",
            "difference": {w: scores["C"][w]["mean_pinball"] - scores[reference][w]["mean_pinball"]
                           for w in ("request", "session")},
            "ci95": {w: intervals[w]["mean_pinball"] for w in ("request", "session")},
            "interpretation": (f"validation에서 C가 {reference}보다 개선" if ci[1] < 0 else
                               "차이 미확인" if ci[0] <= 0 <= ci[1] else
                               f"validation에서 C가 {reference}보다 악화")}
    return results


def regression_previous(result, previous, revision):
    """Exact aggregate regression against #41; no trace or identifiers read."""
    conditions = {}
    for name in ("A", "L", "B"):
        conditions[name] = {
            "selection_equal": result["selection"][name] == previous["selection"][name],
            "validation_metrics_equal": all(
                values["metrics"][name] == previous["metrics"][group]["metrics"][name]
                for group, values in result["metrics"].items())}
    return {"source_revision": revision, "source_path": PREVIOUS_PATH,
            "conditions": conditions, "primary_equal": result["primary"] == previous["primary"],
            "B_minus_L_bootstrap_equal": result["bootstrap"]["B_minus_L"] == previous["bootstrap"]["B_minus_L"]}


def round_summary(selection):
    by_condition_quantile = {
        condition: {label: {"candidates": len(value["candidates"]),
                            "hit_max_rounds": sum(c["hit_max_rounds"] for c in value["candidates"])}
                    for label, value in values.items()}
        for condition, values in selection.items()}
    return {"definition": "evaluated_rounds >= max_rounds; best_iteration may be smaller",
            "by_condition_quantile": by_condition_quantile,
            "overall": {key: sum(value[key] for values in by_condition_quantile.values()
                                  for value in values.values())
                        for key in ("candidates", "hit_max_rounds")}}


def compare_previous(result, previous, revision):
    """Only aggregate values from the immutable #39 artifact, never raw data."""
    selection = {}
    for condition, values in result["selection"].items():
        if condition not in previous["selection"]:
            continue
        selection[condition] = {}
        for label, current in values.items():
            old = previous["selection"][condition][label]
            selection[condition][label] = {
                "previous": {key: old[key] for key in ("settings", "iterations", "holdout_pinball")},
                "current": {key: current[key] for key in ("settings", "iterations", "holdout_pinball")},
                "holdout_pinball_change": current["holdout_pinball"] - old["holdout_pinball"]}
    comparisons = {}
    for name, bootstrap in result["bootstrap"].items():
        if name not in previous["bootstrap"]:
            continue
        reference = bootstrap["reference"]
        comparisons[name] = {}
        for predictor, intervals in bootstrap["groups"]["overall"]["differences"].items():
            if predictor == reference or predictor not in previous["metrics"]["overall"]["metrics"]:
                continue
            comparisons[name][predictor] = {}
            for weighting in ("request", "session"):
                def difference(report):
                    scores = report["metrics"]["overall"]["metrics"]
                    return scores[predictor][weighting]["mean_pinball"] - scores[reference][weighting]["mean_pinball"]

                old, current = difference(previous), difference(result)
                comparisons[name][predictor][weighting] = {
                    "previous_difference": old, "current_difference": current,
                    "change": current - old,
                    "previous_ci95": previous["bootstrap"][name]["groups"]["overall"]["differences"][predictor][weighting]["mean_pinball"],
                    "current_ci95": intervals[weighting]["mean_pinball"]}
    candidates = [c for values in previous["selection"].values() for value in values.values()
                  for c in value["candidates"]]
    return {"source_revision": revision, "source_path": PREVIOUS_PATH,
            "max_rounds": previous["model_settings"]["max_rounds"],
            "candidates": len(candidates),
            # #39 did not record evaluated_rounds. This count is only a lower
            # bound on candidates that exhausted the old iteration budget.
            "best_iteration_at_max": sum(c["iterations"] == previous["model_settings"]["max_rounds"]
                                         for c in candidates),
            "selection": selection, "overall_mean_pinball": comparisons}


def _markdown(result):
    primary = result["primary"]
    lines = ["# A/L/B/C LightGBM validation 비교", "",
             f"**{primary['interpretation']}**. 최종 결론은 승인 후 test 평가에서 정한다.", "",
             f"- 주 지표: 요청 가중 mean pinball B−L = {primary['difference']['request']:.6f}, "
             f"95% CI {primary['ci95']['request']}.",
             f"- 보조 세션 가중 B−L = {primary['difference']['session']:.6f}, "
             f"95% CI {primary['ci95']['session']}.",
             f"- 데이터 revision: `{result['revision']}`; 코드 HEAD: `{result['code_revision']}`.",
             f"- 명령: `{result['command']}`",
             f"- 환경: `{json.dumps(result['environment'], ensure_ascii=False)}`",
             f"- 소요: {result['elapsed_seconds']:.3f}초; 단계별 초: `{json.dumps(result['timing_seconds'])}`",
             f"- 세트별 행·세션: `{json.dumps(result['sets'])}`",
             f"- train 경계 제외 정답: {result['excluded_train_targets']}; fit 경계 제외: {result['excluded_fit_targets']}",
             f"- 고정 격자: `{json.dumps(result['model_settings']['grid'])}`",
             f"- 고정 학습 설정: `{json.dumps(result['model_settings']['parameters'])}`; "
             f"최대 {result['model_settings']['max_rounds']}회, "
             f"patience={result['model_settings']['early_stopping_rounds']}.",
             "- 1~9주 fit_target으로 학습, 10주 holdout_target의 원척도 해당 q pinball로 조기 종료·격자 선택. "
             "동점은 격자 순서. 선택 설정·반복 수로 1~10주 train_target 전체에 재학습. validation 선택 없음.",
             "- 범주 사전은 선택 단계 fit_target, 최종 단계 train_target에서 고정. 미관측·결측 범주는 결측, "
             "수치 결측은 LightGBM에 전달. 입력은 FEATURE_SETS의 A/L/B/C 열만 사용.",
             "- log1p 예측은 expm1로 복원한 후 조기 종료·선택·평가. 출력 분위수는 행별 정렬.",
             f"- validation 정렬 전 교차 행 비율: `{json.dumps(result['crossing_rates'])}`",
             f"- bootstrap: (week, session_id) 세션 군집, B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED}, "
             "짝지은 95% 백분위 CI. 같은 추출을 비교·구간·가중에 공유하며 재학습하지 않는다.",
             "- 한계: 무료 gateway의 특정 계정·기간·모델 구성에 한정한다. 일 블록은 21개이며 CI는 학습·선택 불확실성을 포함하지 않는다. test(14~16주)는 읽지 않았다.",
             "- 요청별 값·식별자는 저장하지 않는다. 전체 설정·소스 해시·지표·CI·gain: [models.json](models.json)",
             "", "## holdout 선택", "", "| 조건 | 분위수 | 설정 | 반복 수 | 원척도 holdout pinball |",
             "|---|---|---|---:|---:|"]
    supplemental = ["", "## C 보조 결과", "", "주 가설 판정은 B−L만 사용한다.", "",
                    "| 비교 | 가중 | mean pinball 차이 | 95% CI | 요청 가중 해석 |",
                    "|---|---|---:|---|---|"]
    for name, values in result["supplementary"].items():
        for weight in ("request", "session"):
            supplemental.append(f"| {name} (보조) | {weight} | {values['difference'][weight]:.6f} | "
                                f"{values['ci95'][weight]} | {values['interpretation'] if weight == 'request' else '보조 가중'} |")
    supplemental += ["", "C 특징의 전체·주별 결측률과 부모·자식 이력 비율·종료 결측 제외 수는 JSON의 "
                     "`cross_session` 및 [features.json](../features/features.json)에 기록한다. "
                     "10분 창은 공개 데이터 시작 이전을 포함하면 결측이다."]
    regression = result.get("regression_41")
    if regression is not None:
        supplemental += ["", "## #41 A/L/B 회귀 확인", "",
                         f"기준 코드 `{regression['source_revision']}`의 집계 산출물과 비교했다. "
                         f"조건별 동일 여부: `{json.dumps(regression['conditions'])}`. "
                         f"B−L CI 동일: `{regression['B_minus_L_bootstrap_equal']}`, "
                         f"주 판정 전체 동일: `{regression['primary_equal']}`."]
    robust = result["robustness"]
    supplemental += ["", "## 군집·계정 민감도", "", "요청 시작 UTC 날짜의 겹치지 않는 일 블록; 재학습 없음. 계정 가중은 각 계정의 행 수 역수다.", "",
                     "| 비교 | 가중 | 차이 | 세션 CI (군집/유효) | 계정 CI (군집/유효) | 일 CI (군집/유효) | 판정 |",
                     "|---|---|---:|---|---|---|---|"]
    for name, entry in robust["comparisons"].items():
        for weight in ("request", "session"):
            cells = [f"{v['ci95'][weight]} ({v['count']}/{v['valid_replicates']})" for v in entry["clusters"].values()]
            supplemental.append(f"| {name} | {weight} | {entry['difference'][weight]:.6f} | " + " | ".join(cells) + f" | {entry['interpretation']} |")
    supplemental += ["", f"전체 계정 가중 mean pinball: `{json.dumps(robust['account_mean_pinball'])}`", "",
                     "| 하위 집단 | 행 | 제외 행 비율 | 비교 | 요청·세션 가중 차이 | 계정 CI 요청·세션 (군집/유효) | 요청 차이 부호 변화 |",
                     "|---|---:|---:|---|---|---|---|"]
    for label, subset in robust["subsets"].items():
        for name, entry in subset["comparisons"].items():
            supplemental.append(f"| {label} | {subset['rows']} | {subset['excluded_row_fraction']:.6f} | {name} | {entry['difference']} | {entry['ci95']} ({entry['clusters']}/{entry['valid_replicates']}) | {entry['sign_changed']} |")
    for label, subset in robust["subsets"].items():
        if subset["rows"]:
            supplemental.append(f"\n{label} 계정 가중 mean pinball: `{json.dumps(subset['account_mean_pinball'])}`\n")
    if "regression_43" in result:
        supplemental += ["", f"#43 회귀 비교: `{json.dumps(result['regression_43'])}`"]
    # Place supplementary findings before the holdout selection table.
    table_start = lines.index("## holdout 선택") - 1
    lines[table_start:table_start] = supplemental + [""]
    for condition, selection in result["selection"].items():
        for label, value in selection.items():
            lines.append(f"| {condition} | {label} | `{json.dumps(value['settings'])}` | "
                         f"{value['iterations']} | {value['holdout_pinball']:.6f} |")
    summary = result["early_stopping"]
    overall = summary["overall"]
    lines += ["", "## 조기 종료 진단", "",
              "`best_iteration`은 holdout 최적 반복 수, `evaluated_rounds`는 실제 평가한 반복 수다. "
              "`hit_max_rounds`는 실제 평가 회차가 상한에 도달했는지 나타낸다. "
              "최적 회차가 상한보다 작아도 patience를 채우기 전에 상한에 도달할 수 있다.", "",
              f"전체 후보 {overall['candidates']}개 중 {overall['hit_max_rounds']}개가 "
              f"{result['model_settings']['max_rounds']}회 상한에 도달했다.", "",
              "| 조건 | 분위수 | 후보 수 | 상한 도달 후보 수 |", "|---|---|---:|---:|"]
    for condition, values in summary["by_condition_quantile"].items():
        for label, value in values.items():
            lines.append(f"| {condition} | {label} | {value['candidates']} | {value['hit_max_rounds']} |")
    if overall["hit_max_rounds"]:
        lines += ["", f"한계: {result['model_settings']['max_rounds']}회에서도 상한에 도달한 후보가 있어 "
                  "조기 종료·충분한 학습을 확인하지 못했다. "
                  "이 비교에서 상한·격자·learning_rate·patience를 추가 변경하지 않았다."]
    else:
        lines += ["", "모든 후보가 상한 전에 종료했다. 후보별 최적·실제 평가 회차는 JSON에 기록했다."]
    comparison = result.get("comparison_300_rounds")
    if comparison is not None:
        lines += ["", "## #39의 300회 결과 대비 변화", "",
                  f"기준은 #39 커밋 `{comparison['source_revision']}`의 `{comparison['source_path']}` 집계값이다. "
                  f"기준 후보 {comparison['candidates']}개 중 최적 반복 수가 300인 후보는 "
                  f"{comparison['best_iteration_at_max']}개다. 기준에는 실제 평가 회차가 없어 "
                  "현재의 상한 도달 집계와 직접 동일시할 수 없다. 변화는 현재 − 300회 결과다.", "",
                  "| 조건 | 분위수 | 이전→현재 반복 수 | 이전 holdout pinball | 현재 holdout pinball | 변화 | 설정 변경 |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for condition, values in comparison["selection"].items():
            for label, value in values.items():
                old, current = value["previous"], value["current"]
                changed = "있음" if old["settings"] != current["settings"] else "없음"
                lines.append(f"| {condition} | {label} | {old['iterations']}→{current['iterations']} | "
                             f"{old['holdout_pinball']:.6f} | {current['holdout_pinball']:.6f} | "
                             f"{value['holdout_pinball_change']:.6f} | {changed} |")
        lines += ["", "validation 전체의 mean pinball 차이와 95% CI. B−L 요청 가중만 주 판정이며 나머지는 보조다.", "",
                  "| 비교 | 예측기 | 가중 | 이전 차이 | 이전 95% CI | 현재 차이 | 현재 95% CI | 차이의 변화 |",
                  "|---|---|---|---:|---|---:|---|---:|"]
        for name, predictors in comparison["overall_mean_pinball"].items():
            for predictor, weightings in predictors.items():
                for weighting, value in weightings.items():
                    old_ci, ci = value["previous_ci95"], value["current_ci95"]
                    lines.append(f"| {name} | {predictor} | {weighting} | {value['previous_difference']:.6f} | "
                                 f"[{old_ci[0]:.6f}, {old_ci[1]:.6f}] | {value['current_difference']:.6f} | "
                                 f"[{ci[0]:.6f}, {ci[1]:.6f}] | {value['change']:.6f} |")
    lines += ["", "## gain 중요도 상위 항목", "", "| 조건 | 분위수 | 특징 | gain | 조건·분위수 내 비중 |",
              "|---|---|---|---:|---:|"]
    for condition, importance in result["importance_gain"].items():
        for label, values in importance.items():
            for entry in values["top"]:
                lines.append(f"| {condition} | {label} | {entry['feature']} | {entry['gain']:.6f} | {entry['fraction']:.6f} |")
    lines += ["", "## validation 지표", "", "요청 가중=행마다 1, 세션 가중=구간 안에서 세션마다 합계 1. "
              "pinball·MAE·폭은 토큰 단위이며 coverage·교차 비율은 0~1이다.", "",
              "| 구간 | 행 | 세션 | 예측기 | 가중 | " + " | ".join(METRICS) + " |",
              "|---|---:|---:|---|---|" + "---:|" * len(METRICS)]
    for group, values in result["metrics"].items():
        for name, weightings in values["metrics"].items():
            for weight, scores in weightings.items():
                lines.append(f"| {group.replace('|', '/')} | {values['rows']} | {values['sessions']} | {name} | {weight} | "
                             + " | ".join(f"{scores[k]:.6f}" for k in METRICS) + " |")
    for comparison, bootstrap in result["bootstrap"].items():
        lines += ["", f"## {comparison} 95% CI", "",
                  f"차이 = 예측기 − {bootstrap['reference']}. 음수 pinball·MAE는 오차 감소다. "
                  "coverage·폭 차이는 단독 개선 판정에 쓰지 않는다.", "",
                  "| 구간 | 유효 반복 | 예측기 | 가중 | " + " | ".join(METRICS) + " |",
                  "|---|---:|---|---|" + "---|" * len(METRICS)]
        for group, values in bootstrap["groups"].items():
            for name, weightings in values["differences"].items():
                if name == bootstrap["reference"]:
                    continue
                for weight, scores in weightings.items():
                    def interval(k):
                        lo, hi = scores[k]
                        return "미산출" if lo is None else f"[{lo:.6f}, {hi:.6f}]"
                    lines.append(f"| {group.replace('|', '/')} | {values['valid_replicates']} | {name} | {weight} | "
                                 + " | ".join(interval(k) for k in METRICS) + " |")
    return "\n".join(lines) + "\n"


def run(root=None, output=OUTPUT, *, previous_revision=PREVIOUS_REVISION,
        regression_revision=REGRESSION_REVISION):
    started = perf_counter()
    previous = (json.loads(subprocess.check_output(
        ["git", "show", f"{previous_revision}:{PREVIOUS_PATH}"], text=True,
        cwd=Path(__file__).resolve().parent)) if previous_revision is not None else None)
    regression = (json.loads(subprocess.check_output(
        ["git", "show", f"{regression_revision}:{PREVIOUS_PATH}"], text=True,
        cwd=Path(__file__).resolve().parent)) if regression_revision is not None else None)
    paths = weeks(root)
    if len(paths) != 13 or any(p.name[:10] != (FIRST_WEEK + timedelta(weeks=i)).isoformat()
                             for i, p in enumerate(paths)):
        raise ValueError("Models require consecutive train/validation weeks 1–13")
    weekly, histories, feature_frames, session_frames = [], [], [], []
    for number, path in enumerate(paths, 1):
        requests = load_table(path, "requests", REQUEST_COLUMNS)
        sessions = load_table(path, "sessions", FEATURE_SESSION_COLUMNS)
        session_frames.append(sessions.assign(week=number))
        calls = load_table(path, "tool_calls", ["session_id", "step", "category", "outcome", "result_tokens"])
        features = build_features(requests, sessions, calls)
        feature_frames.append(features["B"])
        rows = requests[["session_id", "step", "start_ms", "end_ms", "link_reason", "model", "completion_tokens"]].copy()
        rows["week"] = number
        rows["harness"] = features["A"].harness
        histories.append(build_history(rows))
        weekly.append(rows)
        print(f"week {number}: {len(rows):,} rows; features/history built", flush=True)
    rows = pd.concat(weekly, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    session_metadata = pd.concat(session_frames, ignore_index=True)
    account_keys = rows[["week", "session_id"]].merge(
        session_metadata[["week", "session_id", "user_id"]], on=["week", "session_id"],
        how="left", sort=False, validate="many_to_one")
    rows["account"] = account_keys.user_id.to_numpy()
    cross = build_cross_session_features(rows, session_metadata)
    cross_summary = {"diagnostics": cross.attrs["diagnostics"],
                     "missing_rate": {k: float(v) for k, v in cross.isna().mean().items()},
                     "weeks_missing_rate": {
                         str(week): {k: float(v) for k, v in cross.loc[rows.week.eq(week)].isna().mean().items()}
                         for week in sorted(rows.week.unique())}}
    cross.attrs.clear()
    b = pd.concat([pd.concat(feature_frames, ignore_index=True), cross], axis=1)
    features = {name: b.loc[:, list(columns)] for name, columns in FEATURE_SETS.items()}
    del weekly, histories, feature_frames, session_frames, cross, b, requests, calls
    loaded = perf_counter()
    masks = split_masks(rows)
    models = QuantileModels().fit(rows, features)
    trained = perf_counter()
    validation = rows.loc[masks.validation].copy()
    model_predictions, crossing = models.predict({name: frame.loc[masks.validation]
                                                  for name, frame in features.items()})
    baselines = Baselines().fit(rows, history, masks.train_target)
    baseline_predictions, _ = baselines.predict(validation, history.loc[masks.validation])
    predictions = {**baseline_predictions, **model_predictions}
    metrics = evaluate_predictions(validation, predictions)
    evaluated = perf_counter()
    print("validation metrics ready; paired session bootstrap comparisons", flush=True)
    bootstrap = comparisons(validation, predictions)
    robustness = robustness_results(rows, masks, predictions, bootstrap)
    finished = perf_counter()
    result = {
        "revision": REVISION,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_state": "working tree implementation; harness records final commit",
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("loader.py", "features.py", "split.py", "baselines.py", "evaluate.py", "models.py")},
        "command": 'FREEINFERENCE_DATA_DIR="$FREEINFERENCE_DATA_DIR" OPENBLAS_NUM_THREADS=1 '
                   f"{shlex.quote(sys.executable)} -m freeinference.models",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
                        "lightgbm": lgb.__version__, "platform": platform.platform()},
        "split": {"train_weeks": TRAIN_WEEKS, "validation_weeks": VALIDATION_WEEKS,
                  "fit_weeks": FIT_WEEKS, "holdout_week": HOLDOUT_WEEK,
                  "validation_start_ms": VALIDATION_START_MS, "holdout_start_ms": HOLDOUT_START_MS,
                  "session_assignment": "first start_ms week; identity=(public week, session_id)",
                  "target_boundary": "end_ms < next split start; missing end excluded; history unchanged"},
        "sets": {name: _set_counts(rows, masks[name]) for name in masks if name != "session_week"},
        "excluded_train_targets": int((masks.train & ~masks.train_target).sum()),
        "excluded_fit_targets": int((masks.fit & ~masks.fit_target).sum()),
        "feature_sets": FEATURE_SETS, "cross_session": cross_summary,
        "model_settings": {"grid": GRID, "parameters": PARAMETERS, "quantiles": QUANTILES,
                           "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                           "selection": "original-scale holdout q pinball; first grid candidate on tie",
                           "categorical": CATEGORICAL,
                           "vocabulary": "fit_target for selection, train_target for refit; unknown/missing -> NA",
                           "numeric_missing": "native LightGBM missing handling",
                           "crossing": "sort each row after original-scale inverse transform"},
        "selection": models.selection, "early_stopping": round_summary(models.selection),
        "importance_gain": models.importance,
        "crossing_rates": crossing, "metrics": metrics, "bootstrap": bootstrap,
        "primary": primary_result(metrics, bootstrap),
        "supplementary": supplementary_results(metrics, bootstrap), "robustness": robustness,
        "limitations": ["validation only; test unread", "gateway population and period only",
                        "day bootstrap has only 21 blocks; account subgroups are observational",
                        "bootstrap does not include refitting or model-selection uncertainty"],
        "timing_seconds": {"load_features_history": loaded - started, "selection_refit": trained - loaded,
                           "prediction_metrics": evaluated - trained, "bootstrap": finished - evaluated},
        "elapsed_seconds": finished - started}
    if result["early_stopping"]["overall"]["hit_max_rounds"]:
        result["limitations"].append(f"candidates hit {MAX_ROUNDS}-round cap; early stopping not confirmed for all candidates")
    if previous is not None:
        result["comparison_300_rounds"] = compare_previous(result, previous, previous_revision)
    reference_43 = json.loads(subprocess.check_output(
        ["git", "show", f"HEAD:{PREVIOUS_PATH}"], text=True)) if previous_revision is not None else None
    if reference_43 is not None:
        result["regression_43"] = {
            "selection_equal": {name: result["selection"][name] == reference_43["selection"][name] for name in ("A", "L", "B", "C")},
            "metrics_equal": result["metrics"] == reference_43["metrics"],
            "session_bootstrap_equal": result["bootstrap"] == reference_43["bootstrap"],
            "primary_equal": result["primary"] == reference_43["primary"]}
    if regression is not None:
        result["regression_41"] = regression_previous(result, regression, regression_revision)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "models.md").write_text(_markdown(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(data_dir())
