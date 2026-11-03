#!/usr/bin/env python3
"""Issue 15: frozen issue-13 models, one authorized final evaluation.

Freeze first, validate on synthetic fixtures, then evaluate once. The exclusive
log is a permanent attempt marker, including when an evaluation fails.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import time

import numpy as np
from wildchat import revalidate as rv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "wildchat/results"
CONFIG = OUTPUT / "test_final_config.json"
CODE = ("wildchat/finalize.py", "wildchat/revalidate.py", "wildchat/train.py",
        "wildchat/prepare.py", "wildchat/metrics.py")
POLICY = {
    "training": "issue13 full purged train only; validation never added; no retuning or early stopping",
    "exclusions": "test-only whole groups: missing user, user/group/source in original train OR validation, or recorded_at before fixed test boundary; never change training",
    "quantiles": "original P10/P50/P90; clip negatives to zero; never sort",
    "bootstrap": "paired user bootstrap; seed 42; 1000 replicates; same draws across models/cohorts/weights",
    "selection": "issue13 train-internal best baseline per cohort/weight; no test selection",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def versions():
    return {"python": sys.version.split()[0], **{n: importlib.metadata.version(n)
            for n in ("numpy", "lightgbm", "scipy", "pyarrow")}}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def freeze():
    old_path = OUTPUT / "revalidation.json"
    old = json.loads(old_path.read_text())
    for name, digest in old["code_sha256"].items():
        if rv.fingerprint(ROOT / name) != digest:
            raise ValueError(f"issue13 code changed: {name}")
    if versions() != old["versions"]:
        raise ValueError("issue13 runtime versions changed")
    config = {
        "issue": 15, "frozen_at_utc": now(),
        "approval": "repo owner, 2026-11-03, issue15 operator context",
        "issue13_ref": "origin/agent/issue-13-7-train",
        "issue13_commit": "922370cea7abc36e85379db4cee3f99cf3e8e4d6",
        "issue13_result_sha256": rv.fingerprint(old_path),
        "source": old["source"], "versions": versions(), "features": old["features"],
        "seed": old["settings"]["seed"], "threads": old["settings"]["threads"],
        "bootstrap_replicates": old["settings"]["bootstrap_replicates"],
        "common_parameters": old["training"]["common_parameters"],
        "models": {n: {k: old["training"]["models"][n][k] for k in
                    ("selected_parameters", "selected_quantile_fits")} for n in ("A", "B")},
        "baseline_fit": old["baseline_fit"],
        "best_baselines": old["best_baselines_selected_inner"],
        "policy": POLICY,
        "code_sha256": {name: rv.fingerprint(ROOT / name) for name in CODE},
    }
    write_json(CONFIG, config)
    with CONFIG.with_suffix(".sha256").open("x") as handle:
        handle.write(rv.fingerprint(CONFIG) + "  test_final_config.json\n")
    print(f"Frozen before test access: {rv.fingerprint(CONFIG)}", flush=True)


def checked_config():
    digest = rv.fingerprint(CONFIG)
    if CONFIG.with_suffix(".sha256").read_text().split()[0] != digest:
        raise ValueError("frozen config hash mismatch")
    config = json.loads(CONFIG.read_text())
    if config["policy"] != POLICY or versions() != config["versions"]:
        raise ValueError("frozen policy/runtime mismatch")
    for name, expected in config["code_sha256"].items():
        if rv.fingerprint(ROOT / name) != expected:
            raise ValueError(f"frozen code hash mismatch: {name}")
    if rv.fingerprint(OUTPUT / "revalidation.json") != config["issue13_result_sha256"]:
        raise ValueError("issue13 result changed")
    return config, digest


def reserve(output, digest, seed):
    # Fixed CLI output path means changing --output cannot evade the guard.
    if any((output / name).exists() for name in ("test_final.log", "test_final.json", "test_final.md")):
        raise FileExistsError("final evaluation already attempted; rerun prohibited")
    handle = (output / "test_final.log").open("x", encoding="utf-8")
    event(handle, "reserved_before_test_access", config_sha256=digest, seed=seed)
    return handle


def event(handle, status, **values):
    handle.write(json.dumps({"timestamp_utc": now(), "status": status, **values}, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def fit_models(train, config):
    import lightgbm as lgb
    models = {}
    for name, selected in config["models"].items():
        columns = [config["features"]["B"].index(n) for n in config["features"][name]]
        dataset = lgb.Dataset(train.x[:, columns], label=train.y,
                              feature_name=config["features"][name], free_raw_data=False)
        models[name] = [lgb.train({**config["common_parameters"], **selected["selected_parameters"],
                                   "alpha": fit["quantile"]}, dataset,
                                  num_boost_round=fit["best_iteration"])
                        for fit in selected["selected_quantile_fits"]]
    return models


def predict(rows, models, config):
    predictions = rv.predict_baselines(rows, config["baseline_fit"])
    diagnostics = {}
    for name, fitted in models.items():
        columns = [config["features"]["B"].index(n) for n in config["features"][name]]
        raw = np.column_stack([m.predict(rows.x[:, columns], num_threads=config["threads"]) for m in fitted])
        predictions[name] = np.maximum(raw, 0)
        diagnostics[name] = {"raw_crossing_rows": int(np.any(raw[:, :-1] > raw[:, 1:], axis=1).sum()),
                             "negative_prediction_rows": int(np.any(raw < 0, axis=1).sum())}
    return predictions, diagnostics


def load_final(path, expected, names, cutoff):
    """One streaming read, including SHA256; callable with synthetic fixture paths."""
    values = {k: [] for k in ("x", "y", "groups", "sources", "turns", "stamps", "previous", "cumulative")}
    past, identities = {}, set()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            row = json.loads(raw)
            meta = row["metadata"]
            source, turn = meta["source_row_id"], meta["turn_index"]
            stamp, target = rv.timestamp_us(meta["recorded_at"]), row["target"]["output_tokens"]
            identity = (source, meta["user_message_index"])
            if (meta["split"] != "test" or stamp is None or stamp < cutoff or identity in identities or
                any(isinstance(meta[k], bool) or not isinstance(meta[k], int) or meta[k] < 0
                    for k in ("group_id", "source_row_id", "turn_index", "user_message_index")) or
                turn != row["history_features"]["previous_user_turns"] or
                isinstance(target, bool) or not isinstance(target, (int, float)) or
                not np.isfinite(target) or target < 0):
                raise ValueError("invalid final identity, time, turn or target")
            identities.add(identity)
            total, count, last = past.get(source, (0., 0, -1))
            if turn <= last:
                raise ValueError("source turns must be strictly increasing")
            values["cumulative"].append(total / count if count else np.nan)
            past[source] = (total + target, count + 1, turn)
            values["x"].append(rv.feature_vector(row, names))
            for key, value in (("y", target), ("groups", meta["group_id"]), ("sources", source),
                               ("turns", turn), ("stamps", stamp)):
                values[key].append(value)
            lag = row["history_features"]["lag_1_output_tokens"]
            if lag is not None and (not np.isfinite(lag) or lag < 0):
                raise ValueError("invalid preceding response length")
            values["previous"].append(np.nan if lag is None else lag)
    rows = rv.Rows(**{k: np.asarray(v, dtype=np.int64 if k in ("groups", "sources", "turns", "stamps")
                                   else np.float64) for k, v in values.items()},
                   users=np.full(len(values["y"]), "", dtype=object))
    if (not len(rows.y) or len(rows.y) != expected["samples"] or
        len(np.unique(rows.groups)) != expected["groups"] or
        len(np.unique(rows.sources)) != expected["conversations"] or
        int(rows.stamps.min()) != rv.timestamp_us(expected["recorded_at_min"]) or
        int(rows.stamps.max()) != rv.timestamp_us(expected["recorded_at_max"])):
        raise ValueError("final data differs from frozen manifest")
    return rows, digest.hexdigest()


def filter_final(rows, known, cutoff):
    users = set().union(*(set(p.users) for p in known.values())) - {""}
    groups = set().union(*(set(p.groups) for p in known.values()))
    sources = set().union(*(set(p.sources) for p in known.values()))
    reasons = {
        "missing_user": set(rows.groups[rows.users == ""]),
        "known_user": set(rows.groups[np.isin(rows.users, list(users))]),
        "known_group_or_source": set(rows.groups[np.isin(rows.groups, list(groups)) |
                                                   np.isin(rows.sources, list(sources))]),
        "before_boundary": set(rows.groups[rows.stamps < cutoff]),
    }
    excluded = set().union(*reasons.values())
    kept = rows.take(~np.isin(rows.groups, list(excluded)))
    checks = {"known_user_intersection": len(set(kept.users) & users),
              "known_group_intersection": len(set(kept.groups) & groups),
              "known_source_intersection": len(set(kept.sources) & sources),
              "missing_users": int((kept.users == "").sum()),
              "strict_chronology": bool(len(kept.y) and max(p.stamps.max() for p in known.values()) < cutoff <= kept.stamps.min())}
    if not len(kept.y) or any(checks[k] for k in checks if k != "strict_chronology") or not checks["strict_chronology"]:
        raise ValueError("final user-disjoint chronological split failed")
    return kept, {"before": rv.describe(rows), "after": rv.describe(kept),
                  "removed_rows": len(rows.y) - len(kept.y), "removed_groups": len(excluded),
                  "removed_group_reasons_nonexclusive": {k: len(v) for k, v in reasons.items()},
                  "checks": checks, "training_changed": False}


def report(result):
    lines = ["# #15 최종 테스트 1회 평가", "",
             "2026-11-03 저장소 소유자 승인. #13의 정제 train만 재학습하고 validation은 추가하지 않았다. "
             "학습 표본·모델·분위수별 반복 수를 #13과 그대로 유지하여 평가 기간의 차이를 비교한다. 재튜닝·조기 종료·분위수 정렬은 없다.", "",
             f'고정 SHA256: `{result["config_sha256"]}`. 시작: {result["started_at_utc"]}. '
             f'시드 {result["seed"]}, 사용자 bootstrap {result["bootstrap"]["replicates"]}회.', "",
             f'평가 표본: {result["split"]["before"]["samples"]:,} → {result["split"]["after"]["samples"]:,}행, '
             f'{result["split"]["after"]["users"]:,}사용자. 원래 train 또는 validation에 나타난 사용자·그룹·원본 행 및 ID 결측 그룹을 test에서만 통째로 제외했다. '
             '학습 자료는 바꾸지 않았다. 사용자·그룹·원본 행 교집합 0과 시간순서를 assertion으로 확인했다.', "",
             "행 가중은 행 평균, 사용자 가중은 각 턴 구분 안의 사용자별 평균을 동일 가중한다. "
             "fixed는 train 분위수, a/previous_ratio는 직전 응답×train 추정 분위수 비율, "
             "b/cumulative_mean은 같은 대화에서 현재 턴 이전 유효 응답의 누적 평균과 train 잔차 구간이다. "
             "이력이 없으면 fixed다. #13 코드를 그대로 사용하며 모델 입력은 현재 길이(A), 길이+과거 이력(B)이다.", "",
             "## 평균 pinball 및 사용자 bootstrap CI", "",
             "| 턴 | 가중 | fixed | a | b | A | B | B−A [95% CI] |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for cohort, weighted in result["test_metrics"].items():
        for weight, metrics in weighted.items():
            delta = result["bootstrap_results"][cohort][weight]["comparisons"]["B-A"]
            lo, hi = delta["ci95"]["mean_pinball_loss"]
            vals = " | ".join(f'{metrics[m]["mean_pinball_loss"]:.6f}' for m in rv.MODELS)
            lines.append(f'| {cohort} | {weight} | {vals} | {delta["point"]["mean_pinball_loss"]:.6f} [{lo:.6f}, {hi:.6f}] |')
    lines += ["", "## 검증셋과의 차이", "",
              "| 턴 | 가중 | validation A | test A | validation B | test B | validation B−A | test B−A |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cohort, weighted in result["test_metrics"].items():
        for weight, metrics in weighted.items():
            val = result["validation_reference"][cohort][weight]
            a, b = (metrics[n]["mean_pinball_loss"] for n in ("A", "B"))
            va, vb = (val[n]["mean_pinball_loss"] for n in ("A", "B"))
            lines.append(f'| {cohort} | {weight} | {va:.6f} | {a:.6f} | {vb:.6f} | {b:.6f} | {vb-va:.6f} | {b-a:.6f} |')
    lines += ["", "## 전체 지표", "",
              "| 턴 | 가중 | 모델 | 평균 pinball | P10 | P50 | P90 | P50 MAE | coverage | 부호 있는 폭 | 교차율 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cohort, weighted in result["test_metrics"].items():
        for weight, models in weighted.items():
            for model, metrics in models.items():
                lines.append(f'| {cohort} | {weight} | {model} | ' + " | ".join(f'{metrics[k]:.6f}' for k in rv.METRICS) + " |")
    lines += ["", "## 재현 기록과 한계", "",
              "[고정 설정](test_final_config.json), [해시](test_final_config.sha256), [실행 로그](test_final.log), "
              "[전체 결과](test_final.json)에 설정·모델 해시·시각·시드·분할·원시 진단·모든 기준선 대비 CI를 기록했다. "
              "최선 기준선은 #13 내부 홀드아웃에서 이미 선택한 것을 유지한다. bootstrap 중 재학습·재선택하지 않는다.", "",
              "CI는 사용자 G명에서 G명을 복원 추출하는 짝지은 percentile(선형 보간)이다. 모든 모델·턴·가중 방식에 같은 추출을 쓴다. "
              "모델 학습·선택 불확실성, 시간 상관, hashed_ip 프록시 오차는 포함하지 않는다. "
              "첫 샤드의 다른 기간이며 test는 기존 train/validation 사용자도 제외했으므로 모집단 차이가 있다. "
              "validation 대비 차이는 시간 변화와 사용자 구성 변화를 합친 관측이며 원인별 효과를 분리하지 않는다. "
              "기록시각은 실제 요청 도착시각이 아니며 가시 텍스트 재구성 토큰을 평가한다.", "",
              "```sh", "/workspace/venvs/wildchat/bin/python wildchat/finalize.py freeze",
              "/workspace/venvs/wildchat/bin/python wildchat/finalize.py evaluate", "```", "",
              "위 명령은 실행 이력이다. log/result가 존재하면 재평가는 거부한다. 실패한 시도도 로그를 보존하며 재시도하지 않는다.", ""]
    return "\n".join(lines)


def evaluate():
    config, digest = checked_config()
    with reserve(OUTPUT, digest, config["seed"]) as log:
        started = now()
        tick = time.monotonic()
        try:
            prepared = Path(config["source"]["prepared"])
            manifest_path = prepared / "manifest.json"
            if rv.fingerprint(manifest_path) != config["source"]["manifest_sha256"]:
                raise ValueError("preparation manifest changed")
            manifest = json.loads(manifest_path.read_text())
            known, hashes = rv.load_allowed(prepared, manifest, config["features"]["B"])
            if hashes != config["source"]["prepared_sha256"]:
                raise ValueError("train/validation changed")
            source = Path(config["source"]["metadata_source"])
            rv.attach_users(known, source, manifest["input_files"][0])
            train, validation, audit = rv.purge_and_check(known["train"], known["validation"],
                rv.timestamp_us(manifest["boundaries"]["train_validation_cutoff"]))
            old = json.loads((OUTPUT / "revalidation.json").read_text())
            if audit != old["splits"]["outer"] or rv.fit_baselines(train) != config["baseline_fit"]:
                raise ValueError("issue13 training population/baselines changed")
            models = fit_models(train, config)
            # Reproduce the existing validation report before any final-data access.
            val_predictions, _ = predict(validation, models, config)
            val_metrics = rv.evaluate_all(validation, val_predictions)
            for c in val_metrics:
                for w in val_metrics[c]:
                    for m in rv.MODELS:
                        for k in rv.METRICS:
                            if abs(val_metrics[c][w][m][k] - old["validation_metrics"][c][w][m][k]) > 1e-10:
                                raise ValueError("issue13 validation reproduction failed")
            model_hashes = {n: [hashlib.sha256(m.model_to_string().encode()).hexdigest() for m in ms]
                            for n, ms in models.items()}
            event(log, "models_frozen_before_test_access", model_sha256=model_hashes,
                  training=rv.describe(train), validation_reproduced=True)
            cutoff = rv.timestamp_us(manifest["boundaries"]["validation_test_cutoff"])
            event(log, "single_test_read_started", config_sha256=digest, seed=config["seed"])
            rows, test_hash = load_final(prepared / "test.jsonl", manifest["partitions"]["test"], config["features"]["B"], cutoff)
            rv.attach_users({"final": rows}, source, manifest["input_files"][0])
            final, split = filter_final(rows, known, cutoff)
            predictions, diagnostics = predict(final, models, config)
            metrics = rv.evaluate_all(final, predictions)
            info, intervals = rv.bootstrap(final, predictions, config["best_baselines"],
                                           config["seed"], config["bootstrap_replicates"])
            result = {"issue": 15, "started_at_utc": started, "completed_at_utc": now(),
                      "elapsed_seconds": time.monotonic() - tick, "config_sha256": digest,
                      "seed": config["seed"], "versions": versions(), "code_sha256": config["code_sha256"],
                      "model_sha256": model_hashes, "test_sha256": test_hash, "test_read_count": 1,
                      "policy": config["policy"], "training": rv.describe(train), "split": split,
                      "validation_reproduced_before_test": True, "validation_reference": old["validation_metrics"],
                      "test_metrics": metrics, "raw_test_diagnostics": diagnostics,
                      "bootstrap": info, "bootstrap_results": intervals,
                      "best_baselines_selected_inner": config["best_baselines"]}
            write_json(OUTPUT / "test_final.json", result)
            with (OUTPUT / "test_final.md").open("x", encoding="utf-8") as handle:
                handle.write(report(result))
            event(log, "completed", result_sha256=rv.fingerprint(OUTPUT / "test_final.json"),
                  test_sha256=test_hash, split=split, elapsed_seconds=result["elapsed_seconds"])
            print("Single final evaluation completed; reruns prohibited", flush=True)
            return result
        except BaseException as error:
            event(log, "failed_no_retry", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else evaluate()
