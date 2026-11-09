"""Train-only A/L/B quantile selection, refit and aggregate validation report."""
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
                       evaluate_predictions, paired_bootstrap)
from .features import FEATURE_SETS, REQUEST_COLUMNS, build_features
from .loader import FIRST_WEEK, REVISION, data_dir, load_table, weeks
from .split import (FIT_WEEKS, HOLDOUT_START_MS, HOLDOUT_WEEK, SESSION_COLUMNS,
                    TRAIN_WEEKS, VALIDATION_START_MS, VALIDATION_WEEKS, split_masks)

SEED = 20261109
NUM_THREADS = 4
MAX_ROUNDS = 300
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
        raise ValueError("Exactly A/L/B features are required")
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

                    model = lgb.train(params, training, num_boost_round=MAX_ROUNDS,
                                      valid_sets=[holdout], feval=score,
                                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                                                   first_metric_only=True, verbose=False)])
                    iterations = int(model.best_iteration)
                    loss = pinball(y_holdout, original_scale(model.predict(
                        x_holdout, num_iteration=iterations, num_threads=NUM_THREADS), transform), q)
                    candidates.append({"settings": dict(setting), "iterations": iterations,
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
                ("conditions_minus_fixed", "fixed", ("A", "L", "B")),
                ("conditions_minus_previous_completion", "previous_completion", ("A", "L", "B")))}


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


def _markdown(result):
    primary = result["primary"]
    lines = ["# A/L/B LightGBM validation 비교", "",
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
             f"최대 {MAX_ROUNDS}회, patience={EARLY_STOPPING_ROUNDS}.",
             "- 1~9주 fit_target으로 학습, 10주 holdout_target의 원척도 해당 q pinball로 조기 종료·격자 선택. "
             "동점은 격자 순서. 선택 설정·반복 수로 1~10주 train_target 전체에 재학습. validation 선택 없음.",
             "- 범주 사전은 선택 단계 fit_target, 최종 단계 train_target에서 고정. 미관측·결측 범주는 결측, "
             "수치 결측은 LightGBM에 전달. 입력은 FEATURE_SETS의 A/L/B 열만 사용.",
             "- log1p 예측은 expm1로 복원한 후 조기 종료·선택·평가. 출력 분위수는 행별 정렬.",
             f"- validation 정렬 전 교차 행 비율: `{json.dumps(result['crossing_rates'])}`",
             f"- bootstrap: (week, session_id) 세션 군집, B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED}, "
             "짝지은 95% 백분위 CI. 같은 추출을 비교·구간·가중에 공유하며 재학습하지 않는다.",
             "- 한계: 무료 gateway의 특정 계정·기간·모델 구성에 한정한다. 계정·일 군집, 분포 이동·계정 분리 "
             "민감도는 미검증이며 CI는 학습·선택 불확실성을 포함하지 않는다. test(14~16주)는 읽지 않았다.",
             "- 요청별 값·식별자는 저장하지 않는다. 전체 설정·소스 해시·지표·CI·gain: [models.json](models.json)",
             "", "## holdout 선택", "", "| 조건 | 분위수 | 설정 | 반복 수 | 원척도 holdout pinball |",
             "|---|---|---|---:|---:|"]
    for condition, selection in result["selection"].items():
        for label, value in selection.items():
            lines.append(f"| {condition} | {label} | `{json.dumps(value['settings'])}` | "
                         f"{value['iterations']} | {value['holdout_pinball']:.6f} |")
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


def run(root=None, output=OUTPUT):
    started = perf_counter()
    paths = weeks(root)
    if len(paths) != 13 or any(p.name[:10] != (FIRST_WEEK + timedelta(weeks=i)).isoformat()
                             for i, p in enumerate(paths)):
        raise ValueError("Models require consecutive train/validation weeks 1–13")
    weekly, histories, feature_frames = [], [], []
    for number, path in enumerate(paths, 1):
        requests = load_table(path, "requests", REQUEST_COLUMNS)
        sessions = load_table(path, "sessions", ["session_id", "harness"])
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
    b = pd.concat(feature_frames, ignore_index=True)
    features = {name: b.loc[:, list(columns)] for name, columns in FEATURE_SETS.items()}
    del weekly, histories, feature_frames, b, requests, calls
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
        "feature_sets": FEATURE_SETS,
        "model_settings": {"grid": GRID, "parameters": PARAMETERS, "quantiles": QUANTILES,
                           "max_rounds": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                           "selection": "original-scale holdout q pinball; first grid candidate on tie",
                           "categorical": CATEGORICAL,
                           "vocabulary": "fit_target for selection, train_target for refit; unknown/missing -> NA",
                           "numeric_missing": "native LightGBM missing handling",
                           "crossing": "sort each row after original-scale inverse transform"},
        "selection": models.selection, "importance_gain": models.importance,
        "crossing_rates": crossing, "metrics": metrics, "bootstrap": bootstrap,
        "primary": primary_result(metrics, bootstrap),
        "limitations": ["validation only; test unread", "gateway population and period only",
                        "no account/day clustering or account-disjoint sensitivity",
                        "bootstrap does not include refitting or model-selection uncertainty"],
        "timing_seconds": {"load_features_history": loaded - started, "selection_refit": trained - loaded,
                           "prediction_metrics": evaluated - trained, "bootstrap": finished - evaluated},
        "elapsed_seconds": finished - started}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "models.md").write_text(_markdown(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(data_dir())
