#!/usr/bin/env python3
"""Alfredo P0: metadata representation, causal lag ablation and toy bin view.

Only new outputs are accepted. Prediction input JSONL opens are restricted to
train/validation; source Parquet reads project metadata, never conversation text.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import shlex
import time

import numpy as np

from wildchat import revalidate as rv
from wildchat.alfredo_h1_features import H1_NAMES, h1_features
from wildchat.alfredo_p0_support import allowed_jsonl_reads, load_embeddings
from wildchat.category_pca import PCA
from wildchat.prepare import timestamp_us
from wildchat.train import SEARCH_SPACE, feature_names

ROOT = Path(__file__).resolve().parents[1]
PR24 = "d1a45fa7493a09660b1156239b49acefa729af2d"
MODELS = ("B", "B+lag1-only", "B+H1", "C_emb_PCA", "C_emb_PCA+H1")
PAIRS = (("B+lag1-only", "B"), ("B+H1", "B"), ("B+H1", "B+lag1-only"),
         ("C_emb_PCA", "B"), ("C_emb_PCA+H1", "C_emb_PCA"))
METRICS = ("mean_pinball_loss", "mae_p50")
KST = timezone(timedelta(hours=9))


def save_json(path, value):
    with path.open("x", encoding="utf-8") as h:
        h.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_text(path, value):
    with path.open("x", encoding="utf-8") as h:
        h.write(value)


def split_hash(rows):
    """Hash ordered numeric identities, not identifiers emitted in the report."""
    values = np.column_stack([rows.sources, rows.turns, rows.groups, rows.stamps]).astype("<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def lag1_features(sources, turns, outputs):
    """Previous eligible output only; snapshot before current output is stored."""
    past, last_turn, result = {}, {}, []
    for source, turn, output in zip(sources, turns, outputs, strict=True):
        if turn < 0 or turn <= last_turn.get(source, -1) or not np.isfinite(output) or output < 0:
            raise ValueError("invalid chronological eligible response")
        result.append(past.get(source, np.nan))
        past[source], last_turn[source] = float(output), int(turn)
    return np.asarray(result, dtype=float)[:, None]


def designs(rows, nb, pca=None):
    # Storage layout: B | H1(9) | lag1(1) | embedding(384).
    b, h, lag = rows.x[:, :nb], rows.x[:, nb:nb + 9], rows.x[:, nb + 9:nb + 10]
    result = {"B": b, "B+lag1-only": np.column_stack([b, lag]), "B+H1": np.column_stack([b, h])}
    if pca is not None:
        c = np.column_stack([b, pca.transform(rows.x[:, nb + 10:])])
        result.update({"C_emb_PCA": c, "C_emb_PCA+H1": np.column_stack([c, h])})
    return result


def score_bootstrap(rows, predictions, seed=42, replicates=1000):
    """Shared user draws across metrics/models/cohorts; absent cohort users excluded.

    Resample all validation users with replacement. For row weighting retain all
    rows of sampled users; for user weighting average each user's cohort first.
    Empty-cohort draws are omitted. No predictions/IDs leave this function.
    """
    users, codes = np.unique(rows.users, return_inverse=True)
    masks = rv.cohorts(rows)
    sums = np.zeros((len(users), 3, len(MODELS), 2))
    counts = np.zeros((len(users), 3), dtype=np.int64)
    for ci, mask in enumerate(masks.values()):
        np.add.at(counts[:, ci], codes[mask], 1)
        for mi, model in enumerate(MODELS):
            loss = rv.contributions(rows.y[mask], predictions[model][mask])[:, [0, 4]]
            np.add.at(sums[:, ci, mi], codes[mask], loss)
    means = np.divide(sums, counts[:, :, None, None], out=np.zeros_like(sums),
                      where=counts[:, :, None, None] > 0)
    points, draws = {}, {c: {w: [] for w in ("row", "user")} for c in masks}
    for ci, cohort in enumerate(masks):
        points[cohort] = {}
        for w in ("row", "user"):
            denominator = counts[:, ci].sum() if w == "row" else (counts[:, ci] > 0).sum()
            values = (sums[:, ci].sum(0) if w == "row" else means[:, ci].sum(0)) / max(1, denominator)
            points[cohort][w] = {m: {"n_rows": int(counts[:, ci].sum()),
                "n_users": int((counts[:, ci] > 0).sum()),
                **dict(zip(METRICS, map(float, values[mi])))} for mi, m in enumerate(MODELS)}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        chosen = rng.integers(0, len(users), size=len(users))
        for ci, cohort in enumerate(masks):
            for w in ("row", "user"):
                denominator = counts[chosen, ci].sum() if w == "row" else (counts[chosen, ci] > 0).sum()
                if denominator:
                    values = (sums[chosen, ci].sum(0) if w == "row" else means[chosen, ci].sum(0)) / denominator
                    draws[cohort][w].append(values)
    intervals = {}
    for c in masks:
        intervals[c] = {}
        for w in ("row", "user"):
            values = np.asarray(draws[c][w])
            entry = {"models": {}, "differences": {}}
            for mi, m in enumerate(MODELS):
                entry["models"][m] = {metric: {"point": points[c][w][m][metric],
                    "ci95": np.quantile(values[:, mi, j], [.025, .975], method="linear").tolist(),
                    "effective_replicates": len(values)} for j, metric in enumerate(METRICS)}
            for a, b in PAIRS:
                ai, bi = MODELS.index(a), MODELS.index(b)
                entry["differences"][a + "-" + b] = {metric: {
                    "point": points[c][w][a][metric] - points[c][w][b][metric],
                    "ci95": np.quantile(values[:, ai, j] - values[:, bi, j], [.025, .975], method="linear").tolist(),
                    "effective_replicates": len(values)} for j, metric in enumerate(METRICS)}
            intervals[c][w] = entry
    return points, intervals


def check_reference(reference, args):
    """Fail closed before refitting if PR24 provenance/settings differ."""
    for path, expected in reference["code_sha256"].items():
        if path.endswith("temporal.py"):
            continue  # useful functions vendored in support; original blob identified in provenance
        local = ROOT / ("wildchat/alfredo_h1_features.py" if path.endswith("temporal_features.py") else path)
        if rv.fingerprint(local) != expected:
            raise ValueError("PR24 dependency mismatch: " + path)
    versions = {"python": sys.version.split()[0], **{n: importlib.metadata.version(n)
                for n in ("numpy", "lightgbm", "scipy", "pyarrow")}}
    if versions != reference["versions"]:
        raise ValueError("PR24 numerical environment differs")
    for k in ("seed", "inner_fraction", "threads", "max_rounds", "early_stopping_rounds", "bootstrap_replicates"):
        if getattr(args, k) != reference["settings"][k]:
            raise ValueError("PR24 configuration mismatch: " + k)
    if list(SEARCH_SPACE) != reference["training"]["search_space"] or rv.common_parameters(args.seed, args.threads) != reference["training"]["common_parameters"]:
        raise ValueError("PR24 training configuration differs")
    return versions


def ablation(args):
    start = time.monotonic()
    reference = json.loads(args.reference.read_text())
    versions = check_reference(reference, args)
    manifest_path = args.prepared / "manifest.json"
    if rv.fingerprint(manifest_path) != reference["source"]["manifest_sha256"]:
        raise ValueError("PR24 manifest mismatch")
    manifest = json.loads(manifest_path.read_text())
    names = feature_names(manifest["lags"])["B"]
    nb = len(names)
    fnames = {"B": names, "B+lag1-only": names + ["previous_eligible_output_tokens"],
              "B+H1": names + list(H1_NAMES), "C_emb_PCA": names + [f"pc{i+1}" for i in range(32)],
              "C_emb_PCA+H1": names + [f"pc{i+1}" for i in range(32)] + list(H1_NAMES)}
    for m in reference["models"]:
        if fnames[m] != reference["features"][m]:
            raise ValueError("PR24 feature mismatch")
    with allowed_jsonl_reads(args.prepared) as opened:
        parts, hashes = rv.load_allowed(args.prepared, manifest, names)
        if hashes != reference["source"]["prepared_sha256"]:
            raise ValueError("PR24 prepared train/validation hashes differ")
        rv.attach_users(parts, args.source, manifest["input_files"][0])
        emb, cache = load_embeddings(args.prepared, args.embeddings, reference["embedding_cache"]["sha256"])
        for split, rows in parts.items():
            h = np.asarray(h1_features(rows.sources, rows.turns, rows.x[:, 0], rows.y))
            lag = lag1_features(rows.sources, rows.turns, rows.y)
            rows.x = np.column_stack([rows.x, h, lag, emb[split]])
        train, val, outer = rv.purge_and_check(parts["train"], parts["validation"],
                                              timestamp_us(manifest["boundaries"]["train_validation_cutoff"]))
        fit, holdout, inner = rv.inner_split(train, args.inner_fraction)
        if {"outer": outer, "inner": inner} != reference["splits"]:
            raise ValueError("PR24 split audit mismatch")
        print("PR24 code/config/cache/splits match; selecting only lag1 condition", flush=True)
        selected = dict(reference["training"]["models"])
        f, h = fit.take(slice(None)), holdout.take(slice(None))
        f.x, h.x = designs(fit, nb)["B+lag1-only"], designs(holdout, nb)["B+lag1-only"]
        selected["B+lag1-only"], _ = rv.select_model(f, h, list(range(nb + 1)), fnames["B+lag1-only"], args)
        print("Refitting frozen PR24 selections to recover missing per-row losses; PCA on train only", flush=True)
        pca = PCA().fit(train.x[:, nb + 10:], 32)
        if not np.allclose(pca.explained, reference["pca"]["full_train_explained_variance"], atol=1e-12, rtol=0):
            raise ValueError("PR24 PCA mismatch")
        td, vd = designs(train, nb, pca), designs(val, nb, pca)
        predictions, diagnostics = {}, {}
        for m in MODELS:
            tr, va = train.take(slice(None)), val.take(slice(None))
            tr.x, va.x = td[m], vd[m]
            predictions[m], diagnostics[m] = rv.refit_predict(tr, va, list(range(len(fnames[m]))), fnames[m], selected[m], args)
            print("Refit complete: " + m, flush=True)
        points, intervals = score_bootstrap(val, predictions, args.seed, args.bootstrap_replicates)
        max_metric_diff = max(abs(points[c][w][m]["mean_pinball_loss"] - reference["validation_metrics"][c][w][m]["mean_pinball_loss"])
                              for c in points for w in points[c] for m in reference["models"])
        max_ci_diff = max(abs(np.asarray(intervals[c][w]["differences"][label]["mean_pinball_loss"]["ci95"]) - np.asarray(entry["ci95"])).max()
                          for c in reference["bootstrap_ci95"] for w in reference["bootstrap_ci95"][c]
                          for label, entry in reference["bootstrap_ci95"][c][w].items())
        if max(max_metric_diff, max_ci_diff) > 1e-10:
            raise ValueError("PR24 validation metric/bootstrap reproduction failed")
        def missing_report(rows):
            lag = rows.x[:, nb + 9]
            old = rows.x[:, names.index("lag_1_output_tokens")]
            both = np.isfinite(lag) & np.isfinite(old)
            return {"rows": len(lag), "first_turn_rows": int((rows.turns == 0).sum()),
                    "first_turn_lag_present": int(np.isfinite(lag[rows.turns == 0]).sum()),
                    "lag_missing": int(np.isnan(lag).sum()), "B_lag1_missing": int(np.isnan(old).sum()),
                    "both_present": int(both.sum()), "equal_when_both_present": int((lag[both] == old[both]).sum()),
                    "lag_present_B_lag_missing": int((np.isfinite(lag) & np.isnan(old)).sum())}
        missing = {"train": missing_report(train), "validation": missing_report(val)}
        split_digests = {"train": split_hash(train), "validation": split_hash(val),
                         "inner_fit": split_hash(fit), "inner_holdout": split_hash(holdout)}
    result = {"scope": "Alfredo P0; request-level visible output tokens; validation only",
        "recorded_at_kst": datetime.now(KST).isoformat(), "command": shlex.join(sys.argv),
        "provenance": {"base_commit": "495e7e5ff7103531193c0771bb7ff71215beba89", "pr24_commit": PR24,
            "reference_path": str(args.reference), "reference_sha256": rv.fingerprint(args.reference),
            "reference_source_path": "wildchat/results/temporal_h1.json", "prepared_sha256": hashes,
            "manifest_sha256": rv.fingerprint(manifest_path), "source_sha256_from_preparation_not_rehashed": manifest["input_files"][0],
            "split_hash_algorithm": "SHA256 C-order little-endian int64 [source_row_id,turn_index,group_id,timestamp_us] in retained row order",
            "split_hashes": split_digests, "embedding_cache": cache},
        "reference_checks": {"dependency_hashes_settings_versions_features_splits_match": True,
            "max_reproduced_pinball_absolute_difference": max_metric_diff,
            "max_reproduced_paired_pinball_ci_absolute_difference": float(max_ci_diff),
            "four_conditions_retuned": False,
            "reason_for_refit": "PR24 saved aggregate pinball only, no predictions/MAE; fixed selections refit for missing losses and paired CI"},
        "data_access": {"jsonl_allowlist_enforced": True, "opened_jsonl_files": sorted(opened),
            "source_columns": ["hashed_ip", "timestamp"], "raw_text_read": False, "heldout_test_read": False},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": versions, "models": list(MODELS), "features": fnames,
        "lag1_definition": "B plus only previous eligible output tokens, same source conversation; skip invalid pairs; no ratio/EWMA/count extra; NaN without history",
        "h1_definition": reference["h1_definition"], "missingness_and_redundancy": missing,
        "splits": {"outer": outer, "inner": inner}, "training": {"models": selected,
            "search_space": list(SEARCH_SPACE), "common_parameters": rv.common_parameters(args.seed, args.threads),
            "selection": "lag1 only: train-internal row mean pinball, 3 candidates; other four reuse PR24 selections",
            "quantile_rule": "negative clip only; no sorting", "pca_k": 32},
        "bootstrap": {"seed": args.seed, "replicates": args.bootstrap_replicates, "rng": "numpy.random.PCG64",
            "unit": "user-cluster", "paired": True, "shared_draws": "all models/metrics/cohorts/weights",
            "method": "percentile 2.5/97.5 linear; fixed model; no refit/reselection"},
        "validation_metrics": points, "bootstrap_ci95": intervals, "raw_validation_diagnostics": diagnostics,
        "code_sha256": {str(p.relative_to(ROOT)): rv.fingerprint(p) for p in [Path(__file__),
                            ROOT / "wildchat/alfredo_h1_features.py", ROOT / "wildchat/alfredo_p0_support.py"]},
        "elapsed_seconds": time.monotonic() - start}
    save_json(args.output / "lag1_ablation.json", result)
    save_text(args.output / "lag1_ablation.md", render_ablation(result))
    return result


def cell(d):
    lo, hi = d["ci95"]
    return f'{d["point"]:.6f} [{lo:.6f}, {hi:.6f}]'


def render_ablation(r):
    lines = ["# Lag-1 ablation: validation 전용", "",
        "현재 user 프롬프트 제출 직후, 인접 assistant 가시 텍스트의 토큰 수를 예측한다. 10분 수요 예측 결과가 아니다.",
        "B 49.93·C 46.06은 목표값으로 사용하지 않았다. PR24에서 실제 재현된 C는 아래 표를 따른다.", "",
        "## 동일 표본과 재사용", "",
        f'PR24 `{r["provenance"]["pr24_commit"]}`의 설정·수치 원본은 [동결 reference](../alfredo_p0_reference.json)이다. 기존 네 조건은 다시 튜닝하지 않았다.',
        "의존 코드 해시·패키지·입력 train/val SHA256·manifest·특징 순서·outer/inner split 전체 audit를 확인하고 설정을 재사용했다.",
        "기존 결과에는 집계 pinball만 있어, MAE와 paired CI에 필요한 손실을 얻기 위해 고정 설정으로 재학습했다. PCA 32는 전체 정제 train에만 적합했다.",
        f'네 조건 pinball 재현 최대 절대차 {r["reference_checks"]["max_reproduced_pinball_absolute_difference"]:.3g}, 기존 paired CI 최대 절대차 {r["reference_checks"]["max_reproduced_paired_pinball_ci_absolute_difference"]:.3g} (허용오차 1e-10).', "",
        "| 구간 | 요청 쌍 | IP 사용자 프록시 | 첫 턴 |", "|---|---:|---:|---:|"]
    for scope, audit in r["splits"].items():
        for side, desc in audit["after"].items():
            lines.append(f'| {scope}/{side} | {desc["samples"]} | {desc["users"]} | {desc["first_turn_samples"]} |')
    lines += ["", "outer/inner 모두 사용자·중복 그룹·source 교집합 0, ID 결측 0, 기록시각 순서 검사 통과.",
        "정확한 split hash와 산출 방식은 JSON `provenance.split_hashes`에 있다.", "",
        "## 특징과 관측 가능 시점", "",
        "- B: 현재 입력 길이 3개 + 이전 user 턴 수, 이전 1~3 user 슬롯 입력/출력 길이 및 결측 지표(총 31개).",
        "- B+lag1-only: B + 같은 source 대화의 직전 **유효 쌍** 출력 토큰 한 개(32개). H1과 같은 유효 쌍 상태에서 출력 한 값만 남긴 비교다. B의 `lag_1_output_tokens`는 직전 user 슬롯이므로 무효 쌍을 건너뛰는 이 값과 정의가 다를 수 있다. B를 새로 정의하지 않았다.",
        "- B+H1: B + 출력 EWMA(α=.3), 누적 평균·표준편차(ddof=0), 최근 두 유효 출력 차이·원래 턴 간격당 기울기, 직전 출력/입력 비율·비율 EWMA, 현재/직전 입력 배율, 관측 수(40개).",
        "- C_emb_PCA(k32): B + 기존 MiniLM 현재 프롬프트 임베딩의 train PCA 32성분(63개). +H1은 72개. 임베딩은 재생성하지 않았다.",
        "- lag1과 H1은 현재 응답을 넣기 **전** snapshot. 사용자/중복 그룹 간 상태 공유 없음. 이전 응답은 현재 프롬프트 전에 완전히 관측됐다는 가정이다. 현재 입력은 사용 가능하나 현재/미래 출력·행 timestamp·ID는 특징에 없다.",
        "- 첫 턴 lag1은 NaN, H1은 count=0 외에 NaN. 결측을 0으로 대체하지 않는다. 첫 유효 쌍이 turn>0이어도 과거 유효 관측이 없으면 같은 처리다. 첫 턴 점수 차이는 공유 모델의 학습/선택 변화이며 첫 턴에 이력이 있다는 뜻이 아니다.", "",
        "| 표본 | lag1 결측 | B lag1 결측 | 둘 다 관측 | 관측값 같음 | 새 lag만 관측 | 첫 턴 lag 관측 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, m in r["missingness_and_redundancy"].items():
        lines.append(f'| {split} | {m["lag_missing"]} | {m["B_lag1_missing"]} | {m["both_present"]} | {m["equal_when_both_present"]} | {m["lag_present_B_lag_missing"]} | {m["first_turn_lag_present"]} |')
    lines += ["", "B가 이미 출력 lag를 포함하므로 이 비교는 ‘직전 응답 없는 B에 처음 lag를 추가’하는 실험이 아니다. 중복 열과 무효 쌍 처리, 조건별 내부 선택의 영향이 함께 있다.", "",
        "## 선택 설정", "", "| 조건 | num_leaves | min_data_in_leaf | P10/P50/P90 반복 |", "|---|---:|---:|---|"]
    for m in MODELS:
        s = r["training"]["models"][m]
        p = s["selected_parameters"]
        rounds = "/".join(str(f["best_iteration"]) for f in s["selected_quantile_fits"])
        lines.append(f'| {m} | {p["num_leaves"]} | {p["min_data_in_leaf"]} | {rounds} |')
    lines += ["", "lag1에만 같은 내부 80/20 시간 홀드아웃·사용자 purge·3후보·최대500회·40회 조기종료를 적용했다. 선택 기준은 내부 행 가중 평균 pinball이다. validation으로 선택하지 않았다. 음수 예측만 0으로 자르고 분위수 정렬은 하지 않았다.", "",
        "## 검증 지표와 사용자 cluster bootstrap 95% CI", "",
        "MAE는 P50 점 예측의 절대 오차 평균(토큰), pinball은 P10/P50/P90 손실의 평균이다. 모두 낮을수록 좋다.",
        "행 가중은 요청 동일 가중, 사용자 가중은 각 턴 구분에서 사용자별 요청 평균 후 사용자 동일 가중이다. first_turn=turn_index==0, followup_turn=>0, all=합계.",
        "PCG64 seed42·1,000회, 전체 validation 사용자 1,736명을 같은 수로 복원 추출하고 모든 모델·턴·가중·지표에 같은 추출을 썼다. CI는 2.5/97.5 percentile 선형 보간이다.", "",
        "| 턴 | 가중 | 조건 | 평균 pinball [95% CI] | MAE [95% CI] |", "|---|---|---|---|---|"]
    for c, ws in r["bootstrap_ci95"].items():
        for w, entries in ws.items():
            for m in MODELS:
                e = entries["models"][m]
                lines.append(f'| {c} | {w} | {m} | {cell(e["mean_pinball_loss"])} | {cell(e["mae_p50"])} |')
    lines += ["", "## 짝지은 차이 (앞−뒤, 음수=개선)", "",
        "| 턴 | 가중 | 비교 | 평균 pinball 차이 [95% CI] | MAE 차이 [95% CI] |", "|---|---|---|---|---|"]
    for c, ws in r["bootstrap_ci95"].items():
        for w, entries in ws.items():
            for label, e in entries["differences"].items():
                lines.append(f'| {c} | {w} | {label} | {cell(e["mean_pinball_loss"])} | {cell(e["mae_p50"])} |')
    e = r["bootstrap_ci95"]["all"]["row"]["differences"]["B+H1-B+lag1-only"]
    lines += ["", f'전체 행 가중 H1−lag1: pinball {cell(e["mean_pinball_loss"])}, MAE {cell(e["mae_p50"])}.',
        "CI가 0을 포함하는 비교는 H1의 추가 우월성을 주장할 수 없다. 이는 동등성 검정이 아니므로 동등함을 입증한 것도 아니다. 후속/사용자 가중 결과도 함께 읽어야 한다.", "",
        "## 누출 검사·한계·재현", "",
        "snapshot 이전/이후 순서, 미래 응답 변경 불변성, source 격리, 무효 쌍/첫 턴 결측, PCA train 적합, JSONL allowlist는 합성 unit test로 확인한다. 임베딩 키는 train/val에 정확히 대응하며 test 키가 없다.",
        "validation은 PR19/21/24 등에서 반복 관측됐다. 다중비교 보정 없이 제시한 탐색적 CI이며 고정 학습 모델·평가 사용자에 조건부다. 학습/선택 불확실성·시간 상관·IP 식별 오차는 포함하지 않는다. test 접근/평가·최신 모델 또는 agent로의 전이 주장은 없다.",
        "원문·ID·모델·예측 행은 저장하지 않는다. [raw JSON](lag1_ablation.json)에 선택 trial·버전·코드/입력/캐시 hash·scope·split hash·모든 CI가 있다. [NEW runner](../alfredo_p0.py), [H1 원본 복사](../alfredo_h1_features.py), [캐시/접근 검사](../alfredo_p0_support.py)를 사용한다.", "",
        "```sh", "export WILDCHAT_DATA_DIR=/workspace/data/wildchat-1m",
        "OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /workspace/venvs/wildchat/bin/python -m wildchat.alfredo_p0 --task ablation --prepared \"$WILDCHAT_DATA_DIR/derived/issue5-shard0\" --source \"$WILDCHAT_DATA_DIR/data/train-00000-of-00014.parquet\" --embeddings \"$WILDCHAT_DATA_DIR/derived/issue19-categories/embeddings.npz\" --output /tmp/alfredo-p0-reproduction",
        "```", "", "출력은 미사용 경로로 지정한다. 기존 파일은 덮어쓰지 않는다.", ""]
    return "\n".join(lines)


def metadata_summary(files):
    """Only four top-level metadata fields, including all shards for description.

    Does not request message content, per-request targets or heldout split files.
    hashed_ip values are used only for aggregate set cardinality in memory.
    """
    import pyarrow.parquet as pq
    total, users, models, turns, hours = 0, set(), Counter(), Counter(), Counter()
    missing = Counter()
    earliest = latest = None
    inventory = []
    for path in files:
        pqfile = pq.ParquetFile(path)
        inventory.append({"file": path.name, "rows": pqfile.metadata.num_rows, "bytes": path.stat().st_size})
        for batch in pqfile.iter_batches(batch_size=4096, columns=["timestamp", "hashed_ip", "model", "turn"]):
            for row in batch.to_pylist():
                total += 1
                for k, v in row.items():
                    missing[k + "_null"] += int(v is None)
                    missing[k + "_empty_string"] += int(isinstance(v, str) and not v.strip())
                if isinstance(row["hashed_ip"], str) and row["hashed_ip"].strip():
                    users.add(row["hashed_ip"])
                models[str(row["model"])] += 1
                turns[str(row["turn"])] += 1
                stamp = row["timestamp"]
                if stamp is not None:
                    if stamp.tzinfo is None:
                        raise ValueError("naive source timestamp; UTC interpretation unverified")
                    earliest = stamp if earliest is None else min(earliest, stamp)
                    latest = stamp if latest is None else max(latest, stamp)
                    hours[stamp.astimezone(KST).hour] += 1
    return {"conversation_rows": total, "unique_nonempty_hashed_ip": len(users), "source_files": inventory,
            "recorded_at_min_utc": earliest.astimezone(timezone.utc).isoformat(),
            "recorded_at_max_utc": latest.astimezone(timezone.utc).isoformat(),
            "recorded_at_min_kst": earliest.astimezone(KST).isoformat(),
            "recorded_at_max_kst": latest.astimezone(KST).isoformat(),
            "models_by_conversation_rows": dict(sorted(models.items())),
            "turns_per_conversation_distribution": dict(sorted(turns.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)),
            "sum_declared_turns_not_valid_pairs": sum(int(t) * n for t, n in turns.items() if t.isdigit()),
            "recorded_hour_kst_by_conversation_rows": {str(h): hours[h] for h in range(24)},
            "missingness": dict(missing), "projected_columns": ["timestamp", "hashed_ip", "model", "turn"]}


def representation(args):
    print("Aggregating four metadata columns only; no conversation payloads or test split", flush=True)
    files = sorted(args.source.parent.glob("train-*-of-00014.parquet"))
    if len(files) != 14 or args.source not in files:
        raise ValueError("expected complete local 14-shard metadata inventory")
    full, shard = metadata_summary(files), metadata_summary([args.source])
    result = {"scope": "descriptive metadata only; full 14 shards and first-shard subset kept distinct",
        "recorded_at_kst": datetime.now(KST).isoformat(), "command": shlex.join(sys.argv),
        "full_dataset_metadata": full, "first_shard_metadata": shard,
        "external_facts_status": "not externally verified in P0; local audit records source revision/license/timestamp semantics",
        "heldout_access": "no heldout split file, text, per-request targets or performance; approved full-dataset top-level metadata aggregate only",
        "code_sha256": rv.fingerprint(Path(__file__)),
        "local_audit_sha256": rv.fingerprint(ROOT / "wildchat/results/data_audit.md")}
    save_json(args.output / "representation.json", result)
    lines = ["# WildChat 대표성 노트", "",
        "이 연구는 WildChat-1M의 첫 샤드에서 만든 고정 요청 표본의 가시 응답 길이 예측이다. 전체 WildChat이나 최신 모델·agent의 동작을 대표한다고 검증하지 않았다.", "",
        "## 출처와 읽은 범위", "",
        "실제 로컬 source: `$WILDCHAT_DATA_DIR/data/train-00000-of-00014.parquet`부터 `train-00013-of-00014.parquet`까지 14개. 전체 서술 집계에는 최상위 timestamp·hashed_ip·model·turn 네 열만 읽었다. heldout test 파일·대화 내용·요청별 정답·성과에는 접근하지 않았다.",
        "출처는 로컬 감사에 기록된 AllenAI WildChat-1M, 리비전 `7d6490e462285cf85d91eabea0f9a954fbddcd1f`, ODC-BY 1.0이다. **외부 facts는 이번 P0에서 미검증**이다. 파일의 생산자·리비전·라이선스를 바이트로 증명한 것은 아니다.",
        "근거: `wildchat/results/data_audit.md:5`(출처), `:6`(리비전), `:7`(기존 첫 샤드 범위). 실행 원본·파일별 rows/bytes는 [representation.json](representation.json). 집계 코드 `wildchat/alfredo_p0.py`의 `metadata_summary`/`representation` 참조.", "",
        "## 모집단·대화·요청·subset 구분", "",
        "| 범위 | 대화 기록 rows | 비어 있지 않은 hashed_ip 고유 수 | turn 필드 합(유효 요청 쌍 아님) |", "|---|---:|---:|---:|"]
    for name, r in (("전체 14개 metadata", full), ("첫 샤드 metadata", shard)):
        lines.append(f'| {name} | {r["conversation_rows"]:,} | {r["unique_nonempty_hashed_ip"]:,} | {r["sum_declared_turns_not_valid_pairs"]:,} |')
    lines += ["", "- 원본 row는 메시지 목록을 담은 **대화 기록**이다. `conversation_hash`는 고유 실제 세션을 보장하지 않는다. 첫 샤드 감사에서 같은 가시 대화 반복 추가 사본 494개, 동일 첫 user prefix 추가 사본 7,734개가 확인됐다(감사 `:58`~`:62`). 전체 14개의 중복 구조는 이번에 재검증하지 않았다.",
        "- 모델링의 **요청 한 행**은 비어 있지 않은 인접 user→assistant 쌍이다. 첫 샤드 기존 감사: user 후보 172,418개 중 유효 쌍 170,920개, 공백 user 제외 1,498개(감사 `:66`~`:77`). 이 수는 이번 원문 재접근 없이 기존 집계에서 인용했다.",
        "- 기존 준비 sample은 중복 그룹·기록시각 경계로 나뉜다. 이번 예측은 그중 정제 train 81,307쌍/8,702 IP 프록시, validation 12,920쌍/1,736 IP 프록시만 사용한다. 시간 경계 양쪽 공유 사용자와 ID 결측 그룹을 양쪽에서 제거한다. test subset을 조사하거나 재평가하지 않았다.",
        "- 첫 샤드는 파일 순서의 고정 부분 자료이며 전체 14개에서 무작위 추출한 표본이 아니다. 정제 train/val은 원본보다 추가 선택된 모집단이다. pair 중복은 유지하되 연결된 그룹이 분할을 넘지 않도록 한다.",
        "근거: `docs/prd.md:27`~`:52`, `wildchat/revalidate.py:56`~`:143`(허용 입력/ID 연결), `:153`~`:199`(purge), [lag1 JSON](lag1_ablation.json)의 `splits`·`provenance.split_hashes`.", "",
        "## 관측 기간과 시각 의미", "",
        "| 범위 | 최소 기록시각(KST, UTC+09:00) | 최대 기록시각(KST, UTC+09:00) |", "|---|---|---|"]
    for name, r in (("전체 14개", full), ("첫 샤드", shard)):
        lines.append(f'| {name} | {r["recorded_at_min_kst"]} | {r["recorded_at_max_kst"]} |')
    lines += ["", "Parquet timestamp의 UTC timezone을 확인한 뒤 `astimezone(UTC+09:00)`으로 9시간 더해 표시했다. UTC 원본 최소/최대는 JSON에 함께 저장했다. KST 시각에 다시 UTC 표지를 붙이지 않는다.",
        "로컬 감사가 인용한 문서상 행 timestamp는 마지막 턴 기록시각이고 assistant timestamp는 백엔드의 응답 수신시각이다(외부 의미 미재검증). 첫 샤드 user timestamp는 모두 결측(172,418개)이며, 실제 요청 제출시각을 알 수 없다. 아래 hour 분포는 **대화 마지막 기록시각** 분포로, 요청 도착량·활성 사용자 시간대·출력 생성량 분포가 아니다. 지역별 local time도 아니다.",
        "근거: `wildchat/results/data_audit.md:81`~`:84`, `wildchat/prepare.py:245`~`:292`(시각을 예측 특징으로 대체하지 않음).", "",
        "## 모델 구성 (대화 row 가중)", "", "| model 필드 | 전체 14개 rows (%) | 첫 샤드 rows (%) |", "|---|---:|---:|"]
    for m in sorted(set(full["models_by_conversation_rows"]) | set(shard["models_by_conversation_rows"])):
        a, b = full["models_by_conversation_rows"].get(m, 0), shard["models_by_conversation_rows"].get(m, 0)
        lines.append(f'| {m} | {a:,} ({100*a/full["conversation_rows"]:.2f}%) | {b:,} ({100*b/shard["conversation_rows"]:.2f}%) |')
    lines += ["", "이는 데이터의 model 문자열 구성이다. 현재 서비스 모델의 버전·라우팅·agent 사용 비중·내부 tool 호출 여부를 알려주지 않는다. 요청별 모델 비율로 바꾸지 않는다.", "",
        "## 대화 turn 분포", "", "| 기록된 turn 수 | 전체 14개 대화 rows | 첫 샤드 대화 rows |", "|---|---:|---:|"]
    for t in sorted(set(full["turns_per_conversation_distribution"]) | set(shard["turns_per_conversation_distribution"]), key=int):
        lines.append(f'| {t} | {full["turns_per_conversation_distribution"].get(t,0):,} | {shard["turns_per_conversation_distribution"].get(t,0):,} |')
    lines += ["", "turn은 기록의 상호작용 횟수이며 유효 요청 수와 다르다. 요청 turn_index==0과 >0의 정제 표본 수는 [ablation](lag1_ablation.md)에 별도 보고한다. 전체 turn 숫자와 모델링 요청 분모를 섞지 않는다.", "",
        "## 시간대 분포 (KST, 대화 row 마지막 기록시각)", "", "| KST hour | 전체 14개 rows | 첫 샤드 rows |", "|---|---:|---:|"]
    for h in range(24):
        lines.append(f'| {h:02d} | {full["recorded_hour_kst_by_conversation_rows"][str(h)]:,} | {shard["recorded_hour_kst_by_conversation_rows"][str(h)]:,} |')
    lines += ["", "## 결측과 식별 한계", "", "| metadata 필드 | 전체 null / 빈 문자열 | 첫 샤드 null / 빈 문자열 |", "|---|---:|---:|"]
    for k in full["projected_columns"]:
        lines.append(f'| {k} | {full["missingness"][k+"_null"]:,} / {full["missingness"][k+"_empty_string"]:,} | {shard["missingness"][k+"_null"]:,} / {shard["missingness"][k+"_empty_string"]:,} |')
    lines += ["", "hashed_ip는 실제 개인 ID가 아닌 프록시다. 공유 IP·IP 변경·수집/동의/마스킹에 따른 선택 편향이 남고, 고유 IP 수를 실제 사용자 수로 단정할 수 없다. ID는 집계 후 내보내지 않았다. metadata 결측이 없어도 수집된 대화의 완전성을 뜻하지 않는다.",
        "토큰은 가시 텍스트를 o200k_base로 독립 계수한 값이며 전체 API 문맥·청구 토큰·reasoning/tool/agent 호출 총량이 아니다. 이번 분석은 request 시작·완료시각 또는 실행 중 생성량을 복원하지 않는다. 성능·운영 효과·최신 모델/agent 전이를 이 대표성 집계로 주장하지 않는다.", "",
        "## 재현", "", "```sh", "export WILDCHAT_DATA_DIR=/workspace/data/wildchat-1m",
        "PYTHONDONTWRITEBYTECODE=1 /workspace/venvs/wildchat/bin/python -m wildchat.alfredo_p0 --task representation --source \"$WILDCHAT_DATA_DIR/data/train-00000-of-00014.parquet\" --output /tmp/alfredo-representation",
        "```", "", "새 출력 경로만 허용한다. 전체 source 파일 bytes를 해시하려고 대화 payload를 추가로 읽지 않았으며 footer rows·파일 크기·투영 열·집계 시각·코드 hash를 JSON에 남겼다.", ""]
    save_text(args.output / "representation.md", "\n".join(lines))


def toy_bins(requests):
    bins = {}
    for minute, input_tokens, output_tokens in requests:
        key = int(minute // 10) * 10
        values = bins.setdefault(key, [0, 0])
        values[0] += input_tokens
        values[1] += output_tokens
    return bins


def diagram(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    requests = [(1, 100, 220), (7, 50, 80), (12, 120, 300), (18, 80, 100), (23, 90, 180)]
    sums = toy_bins(requests)
    fig, ax = plt.subplots(figsize=(13.5, 8), dpi=150)
    fig.patch.set_facecolor("#f5f7fb")
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(.5, 7.5, "From requests to 10-minute demand bins", fontsize=22, weight="bold", color="#17283e")
    ax.text(.5, 7.05, "TOY EXAMPLE  |  all times KST (UTC+09:00)  |  no model trained or evaluated", fontsize=11, color="#51617a")
    def box(x, y, w, h, text, color="#e3ebf8", fontsize=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#b3c2d8"))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, color="#17283e", linespacing=1.6)
    box(.6, 3.9, 3.0, 2.55, "Request start | input | final output\n09:01 | 100 | 220\n09:07 | 50 | 80\n09:12 | 120 | 300\n09:18 | 80 | 100\n09:23 | 90 | 180", fontsize=11)
    box(4.4, 3.9, 3.9, 2.55, "Arrival-based labels (known later)\n[09:00,09:10): input 150 / output 300\n[09:10,09:20): input 200 / output 400\n[09:20,09:30): input 90 / output 180", fontsize=11)
    ax.annotate("", xy=(4.2, 5.2), xytext=(3.7, 5.2), arrowprops={"arrowstyle":"->", "lw":2})
    box(9.0, 3.9, 3.9, 2.55, "At t = 09:20\nInputs: eligible closed-bin history\nTarget: demand in [09:20,09:30)\nForecast output: P10 / P50 / P90\n(separate input/output totals)", color="#dcefe8", fontsize=11)
    ax.annotate("", xy=(8.8, 5.2), xytext=(8.4, 5.2), arrowprops={"arrowstyle":"->", "lw":2})
    box(.6, 1.85, 12.3, 1.25, "At request start, input length is observable; complete output is NOT.\nA closed arrival bin can still contain unfinished responses. Output-history features need completion gating or delay.", color="#fff0d7", fontsize=13)
    ax.text(.6, .9, "Labels sum final output for requests STARTING in the bin; they are not tokens generated during the bin.", fontsize=12, color="#34475f")
    ax.text(.6, .4, "WildChat user start times are missing; recorded row time cannot replace them. This operational view is conceptual.", fontsize=11, color="#34475f")
    fig.savefig(args.output / "ten_minute_bins.png", bbox_inches="tight")
    plt.close(fig)
    save_json(args.output / "ten_minute_bins.json", {"scope": "toy, no training or evaluation", "timezone": "KST UTC+09:00",
        "requests_minute_from_09_00_input_output": requests, "arrival_bin_start_minute_sums_input_output": sums,
        "output_is_final_not_available_at_start": True, "source": "manually specified hypothetical values", "code_sha256": rv.fingerprint(Path(__file__))})
    save_text(args.output / "ten_minute_bins.md", """# 요청을 10분 bin으로 보는 운영 관점

![가상 요청을 10분 bin으로 집계하는 개념도](ten_minute_bins.png)

**그림의 모든 시각·토큰은 가상 예시(toy)**이며 KST(UTC+09:00)이다. 실제 WildChat의 시간별 수요·예측 결과가 아니다. NEW model을 만들거나 평가하지 않았다.

개별 요청의 실제 시작시각 s와 입력 I, 최종 출력 O가 있다면, 고정 10분 bin `[t,t+10분)`에 시작한 요청의 I 합·O 합을 각각 수요 정답으로 정의할 수 있다. 경계 09:10 요청은 다음 bin에 포함한다. 09:01/09:07 요청의 입력 100+50=150, 최종 출력 220+80=300처럼 집계한다. 출력 전체를 시작 bin에 귀속하며 bin 안에서 실제 생성한 토큰량과 다르다.

예측 시점 09:20에는 **직전 closed bins에서 그때까지 관측 가능한 정보**를 입력으로, 다음 `[09:20,09:30)`의 입력합·출력합을 각각 예측하는 관점이다. 이는 현재 도착한 한 요청의 응답 길이를 맞히는 B/H1과 별개다. 그림의 마지막 bin 합 90/180은 나중에 확인하는 가상 정답이며 사전 예측값이 아니다.

**요청 시작 시 complete output은 모른다.** bin의 도착 구간이 닫혀도 그 안 요청의 응답이 끝나지 않을 수 있다. 과거 출력합을 특징으로 쓰려면 완료시각으로 관측 여부를 확인하고 지연된 완결 bin만 쓰거나 미완료 상태를 따로 관리해야 한다. 오프라인 최종 출력합을 즉시 입력하면 미래 누출이다. 입력합 역시 실제 도착 로그와 전체 관측 확인이 필요하다.

WildChat 첫 샤드의 user 시작시각은 모두 결측이고, 행 timestamp는 마지막 기록시각이다. 응답 수신시각이 곧 신뢰할 수 있는 시작/생성/완료 타임라인을 제공하지 않는다. 현재 prepared 표본에는 요청별 검증된 완료시각도 없다. 따라서 이 자료로 실제 10분 수요·미완료 작업량·GPU 점유를 복원하거나 성능을 주장하지 않는다. 근거: `wildchat/results/data_audit.md:81`~`:84`, `wildchat/prepare.py:230`~`:292`. 도착 기준 정답과 생성량의 구분은 `docs/prd.md:101`~`:110`의 과거 별도 설계와 연결되지만 그 모델을 구현한 것은 아니다.

재현: `MPLCONFIGDIR=/tmp/alfredo-mpl PYTHONDONTWRITEBYTECODE=1 /workspace/venvs/wildchat/bin/python -m wildchat.alfredo_p0 --task diagram --output /tmp/alfredo-bins`. [raw toy JSON](ten_minute_bins.json)에 모든 가상 입력과 집계가 있다. matplotlib만 사용하며 이미지 생성 도구나 추가 의존성은 쓰지 않는다.
""")


def explainer(args):
    save_text(args.output / "forecast_output_explainer.md", """# 다음 10분 수요, 숫자 하나보다 범위로 읽기

**가상 예시**: 09:20 KST에 다음 10분(`[09:20,09:30)`) 도착 요청의 **최종 출력 토큰 합**을 예측한다고 하자.

| 출력 | 가상 값 | 쉬운 해석(모델이 잘 보정됐다면) |
|---|---:|---|
| P10 | 8,000 토큰 | 비슷한 상황의 약 10%는 이 값 이하일 것으로 예상 |
| P50 | 12,000 토큰 | 절반쯤은 이 값 이하, 절반쯤은 이상인 중앙 예측 |
| P90 | 20,000 토큰 | 비슷한 상황의 약 90%는 이 값 이하일 것으로 예상 |

P10~P90은 약 80%를 담도록 기대하는 범위다. **P90은 보장상한이 아니다.** 잘 보정돼도 약 10%는 넘을 수 있고, 수요 변화나 모델 오차로 초과가 더 많을 수 있다. 큰 수요에 대비할 참고값이지 GPU 용량이나 서비스 품질을 보장하는 값은 아니다.

현재 B/H1 실험의 P50은 **이미 도착한 한 요청**의 가시 응답 길이 점 예측이다. 위 예시는 **아직 도착하지 않은 다음 10분 전체 요청**의 누적 수요 분포다. 개별 요청 분위수를 단순히 더하면 bin의 분위수가 되지 않는다. 입력 수요와 출력 수요도 따로 예측·검증해야 한다.

이 10분 probabilistic forecasting 모델과 coverage 검증은 **미구현·미검증**이다. 위 숫자는 실제 실행 결과가 아니다. request 모델의 P10/P50/P90 결과를 10분 coverage나 최신 모델/agent의 성능으로 해석하지 않는다. coverage는 이후 별도 자료·범위가 정해졌을 때 확인할 문제다.

관련: [10분 bin 개념도](ten_minute_bins.md), [현재 request 검증](lag1_ablation.md). 재현: `/workspace/venvs/wildchat/bin/python -m wildchat.alfredo_p0 --task explainer --output /tmp/alfredo-explainer`.
""")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("ablation", "representation", "diagram", "explainer"), required=True)
    p.add_argument("--prepared", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--embeddings", type=Path)
    p.add_argument("--reference", type=Path, default=ROOT / "wildchat/alfredo_p0_reference.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner-fraction", type=float, default=.8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=40)
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = p.parse_args()
    required = {"ablation": ("prepared", "source", "embeddings"), "representation": ("source",)}
    if any(getattr(args, k) is None for k in required.get(args.task, ())):
        p.error("missing required task input")
    outputs = {"ablation": ("lag1_ablation.json", "lag1_ablation.md"),
               "representation": ("representation.json", "representation.md"),
               "diagram": ("ten_minute_bins.png", "ten_minute_bins.json", "ten_minute_bins.md"),
               "explainer": ("forecast_output_explainer.md",)}
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / n).exists() for n in outputs[args.task]):
        raise FileExistsError("refusing to overwrite existing outputs")
    globals()[args.task](args)
    print("P0 task complete: " + args.task, flush=True)


if __name__ == "__main__":
    main()
