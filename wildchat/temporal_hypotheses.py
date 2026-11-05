#!/usr/bin/env python3
"""VAL-only temporal ablations; fixed issue-21 B baseline without embeddings.

Requires existing prepared train/validation and metadata-only raw-shard projection.
Never opens a held-out split, raw message text, or raw response targets.
No raw text, user IDs, predictions, or models are written to the repository.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import time

import numpy as np

from wildchat import revalidate as rv
from wildchat import temporal_causal as causal
from wildchat.train import feature_names

HYPOTHESES = {
    "H1_sequence_lengths": "동일 대화의 이전 유효 응답 EWMA(α=.3), 누적 평균·표준편차, 마지막 두 응답 차이·턴당 기울기, 이전 출력/입력 비율·EWMA·현재 입력 배율·관측 수",
    "H2_reply_cadence": "현재 턴 이전 assistant 완료 시각끼리의 마지막 간격·누적 span·log1p와 결측 표시. 현재 요청까지의 대기시간이 아님",
    "H3_calendar_drift": "이전 assistant 완료 시각의 epoch day·월·UTC hour/weekday sin/cos와 hour×log1p(현재 입력). 첫 턴 결측",
}
FEATURES = {
    "H1_sequence_lengths": ["seq_ewma", "seq_mean", "seq_std", "seq_delta", "seq_slope", "seq_last_ratio", "seq_ratio_ewma", "seq_input_ratio", "seq_count"],
    "H2_reply_cadence": ["reply_gap", "reply_log_gap", "reply_span", "reply_log_span", "reply_gap_missing"],
    "H3_calendar_drift": ["anchor_epoch_day", "anchor_month", "anchor_hour_sin", "anchor_hour_cos", "anchor_weekday_sin", "anchor_weekday_cos", "anchor_hour_input", "anchor_missing"],
    "H4_prompt_sequence": [f"prior_last_pc{i+1}" for i in range(32)] + [f"prior_mean_pc{i+1}" for i in range(32)] + ["prior_prompt_cosine", "prior_prompt_distance"],
}


def preflight(args):
    """Check named inputs only; no enumeration or raw-source fallback."""
    missing = [str(p) for p in (args.prepared / "manifest.json", args.prepared / "train.jsonl",
                               args.prepared / "validation.jsonl", args.source)
               if not p.is_file()]
    packages = [p for p in ("numpy", "lightgbm", "pyarrow") if importlib.util.find_spec(p) is None]
    return {"missing_inputs": missing, "missing_packages": packages,
            "ready": not missing and not packages}


def load_metadata(parts, prepared, sidecar):
    """Sidecar: exact prepared train/val source IDs, no original-shard scanning.

    Each line has source_row_id, hashed_ip, recorded_at, messages (role/timestamp
    only). Provider must create this file in the approved data environment from
    the pre-existing train/validation allowlist; do not regenerate preparation.
    """
    allowed = set().union(*(set(r.sources.tolist()) for r in parts.values()))
    meta = {}
    with sidecar.open() as handle:
        for line in handle:
            item = json.loads(line)
            sid = item["source_row_id"]
            if sid not in allowed or sid in meta:
                raise ValueError("metadata is not an exact train/validation-only sidecar")
            if set(item) != {"source_row_id", "hashed_ip", "recorded_at", "messages"}:
                raise ValueError("unexpected metadata columns")
            for message in item["messages"]:
                if set(message) - {"role", "timestamp"}:
                    raise ValueError("metadata must contain no text or extra columns")
            meta[sid] = item
    if set(meta) != allowed:
        raise ValueError("missing allowed metadata sources")
    return attach_metadata_mapping(parts, prepared, meta)


def attach_source_metadata(parts, prepared, source, expected):
    """#21 metadata projection, Python materialization restricted to allowed IDs.

    Parquet metadata pages may contain other raw records; no content/target
    column is requested. Non-allowed records never enter Python metadata maps.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(source)
    if source.name != expected["name"] or parquet.metadata.num_rows != expected["rows"]:
        raise ValueError("raw metadata source differs from preparation provenance")
    allowed = set().union(*(set(r.sources.tolist()) for r in parts.values()))
    columns = ["hashed_ip", "timestamp", "conversation.list.element.role", "conversation.list.element.timestamp"]
    meta, offset = {}, 0
    audit = {"projected_columns": columns, "selected_sources": len(allowed),
             "user_messages": 0, "user_timestamps_present": 0, "assistant_timestamps_missing": 0,
             "raw_text_or_target_columns": False, "python_metadata_sources_outside_allowlist": 0}
    for batch in parquet.iter_batches(batch_size=1024, columns=columns):
        take = [i for i in range(len(batch)) if offset+i in allowed]
        for i, item in zip(take, batch.take(pa.array(take, type=pa.int64())).to_pylist()):
            messages = item["conversation"] or []
            for message in messages:
                stamp = rv.timestamp_us(message.get("timestamp"))
                if message["role"] == "user":
                    audit["user_messages"] += 1
                    audit["user_timestamps_present"] += int(stamp is not None)
                elif message["role"] == "assistant":
                    audit["assistant_timestamps_missing"] += int(stamp is None)
            meta[offset+i] = {"hashed_ip": item["hashed_ip"], "recorded_at": item["timestamp"], "messages": messages}
        offset += len(batch)
    if set(meta) != allowed:
        raise ValueError("source IDs unavailable in metadata projection")
    return attach_metadata_mapping(parts, prepared, meta), audit


def attach_metadata_mapping(parts, prepared, meta):
    result = {}
    for split, rows in parts.items():
        with (prepared / (split + ".jsonl")).open() as handle:
            indices = np.asarray([json.loads(s)["metadata"]["user_message_index"] for s in handle])
        anchors, users, gaps = [], [], []
        if len(indices) != len(rows.y):
            raise ValueError("metadata alignment mismatch")
        for sid, index, stamp in zip(rows.sources, indices, rows.stamps):
            item = meta[int(sid)]
            if rv.timestamp_us(item["recorded_at"]) != stamp:
                raise ValueError("source timestamp mismatch")
            messages = item["messages"]
            if index >= len(messages) or messages[index]["role"] != "user":
                raise ValueError("user message alignment mismatch")
            # Reuse #21 prefix validation; never inspect current/future replies.
            prefix = messages[:int(index)+1]
            times, _ = causal.temporal_metadata(prefix, None)
            anchor = times[int(index)]
            anchors.append(causal.MISSING_TIME if anchor is None else anchor)
            user = item["hashed_ip"]
            users.append(user if isinstance(user, str) and user.strip() else "")
            past = [rv.timestamp_us(m.get("timestamp")) for m in prefix[:-1] if m["role"] == "assistant"]
            if anchor is None or len(past) < 2:
                gaps.append([np.nan, np.nan, np.nan, np.nan, 1])
            else:
                gap, span = (past[-1]-past[-2])/1e6, (past[-1]-past[0])/1e6
                assert gap >= 0 and span >= 0
                gaps.append([gap, np.log1p(gap), span, np.log1p(span), 0])
        # Keep all side arrays attached through the existing group/user purges.
        result[split] = causal.HistoryRows(**{**{k: getattr(rows, k) for k in rv.Rows.__dataclass_fields__},
                            "users": np.asarray(users, dtype=object), "x": np.column_stack((rows.x, gaps))}, indices=indices,
                            anchors=np.asarray(anchors, dtype=np.int64),
                            completions=np.full(len(rows.y), causal.MISSING_TIME, dtype=np.int64),
                            embeddings=np.empty((len(rows.y), 0)))
    return result


def attach_embeddings(parts, path):
    """Validate keys BEFORE materializing vectors; mixed-split caches are refused."""
    allowed = {(int(s), int(i)) for rows in parts.values() for s, i in zip(rows.sources, rows.indices)}
    with np.load(path, allow_pickle=False) as archive:
        keys = archive["keys"]
        if keys.ndim != 2 or keys.shape[1] != 2 or not np.issubdtype(keys.dtype, np.integer):
            raise ValueError("invalid embedding keys")
        identities = [tuple(map(int, k)) for k in keys]
        if len(set(identities)) != len(identities) or set(identities) != allowed:
            raise ValueError("cache must contain exactly train/validation keys; vectors not loaded")
        vectors = archive["vectors"]
        if vectors.shape != (len(keys), 384) or not np.isfinite(vectors).all():
            raise ValueError("invalid MiniLM vectors")
        lookup = {k: i for i, k in enumerate(identities)}
        for rows in parts.values():
            rows.embeddings = vectors[[lookup[(int(s), int(i))] for s, i in zip(rows.sources, rows.indices)]].astype(np.float32)


def sequence_features(rows, pcs=None):
    """Snapshot before current y/embedding update; state isolated by source."""
    h1 = np.full((len(rows.y), len(FEATURES["H1_sequence_lengths"])), np.nan)
    h4 = np.full((len(rows.y), len(FEATURES["H4_prompt_sequence"])), np.nan)
    state = {}
    for i, sid in enumerate(rows.sources):
        history = state.setdefault(int(sid), [])
        turn = int(rows.turns[i])
        if history and turn <= history[-1][0]:
            raise ValueError("non-increasing conversation turns")
        h1[i, -1] = len(history)
        if history:
            outputs = np.asarray([h[1] for h in history])
            ewma, ratio_ewma = history[0][1], history[0][1] / max(history[0][2], 1)
            for _, y, inp, _ in history[1:]:
                ewma = .3*y + .7*ewma
                ratio_ewma = .3*y/max(inp, 1) + .7*ratio_ewma
            delta = outputs[-1]-outputs[-2] if len(history) > 1 else np.nan
            slope = delta/(history[-1][0]-history[-2][0]) if len(history) > 1 else np.nan
            h1[i] = [ewma, outputs.mean(), outputs.std(), delta, slope,
                     outputs[-1]/max(history[-1][2], 1), ratio_ewma,
                     rows.x[i, 0]/max(history[-1][2], 1), len(history)]
            if pcs is not None:
                last, mean = history[-1][3], np.mean([h[3] for h in history], axis=0)
                cosine = float(pcs[i] @ last / max(np.linalg.norm(pcs[i])*np.linalg.norm(last), 1e-12))
                h4[i] = np.r_[last, mean, cosine, np.linalg.norm(pcs[i]-last)]
        history.append((turn, float(rows.y[i]), float(rows.x[i, 0]), None if pcs is None else pcs[i].copy()))
    return h1, h4


def calendar_features(rows):
    out = np.full((len(rows.y), len(FEATURES["H3_calendar_drift"])), np.nan)
    out[:, -1] = 1
    for i, anchor in enumerate(rows.anchors):
        if anchor == causal.MISSING_TIME:
            continue
        dt = datetime.fromtimestamp(int(anchor)/1e6, tz=timezone.utc)
        hour, day = 2*np.pi*dt.hour/24, 2*np.pi*dt.weekday()/7
        out[i] = [int(anchor)/86400e6, dt.month, np.sin(hour), np.cos(hour),
                  np.sin(day), np.cos(day), dt.hour*np.log1p(rows.x[i, 0]), 0]
    return out


def build_designs(rows, nb):
    base = rows.x[:, :nb]
    h1, _ = sequence_features(rows)
    extras = dict(zip(HYPOTHESES, (h1, rows.x[:, nb:nb+5], calendar_features(rows))))
    return {"baseline": base, **{m: np.column_stack((base, x)) for m, x in extras.items()}}


def render(result, summary=False):
    actual = result["validation_metrics"]["all"]["row"]["baseline"]["mean_pinball_loss"]
    historical = result["historical_baseline_all_row"]
    lines = ["# 시간 구조 가설 VAL 실험" + (" 요약" if summary else ""), "",
             "상태: 실제 VAL 학습·평가 완료. test.jsonl 미접근·test 정답 미사용. Seed 42.", "",
             "기준선: #21 B (현재 길이+대화 내 길이 이력 31개). 동일 사용자 분리·시간 구간, 음수만 0 처리·분위수 무정렬.",
             "하이퍼파라미터 31 leaves / min leaf 300, P10/P50/P90 403/244/279회 고정. 모든 H는 동일 기준선에 하나씩 추가하며 재튜닝하지 않았다.", ""]
    lines += [f"B 기준선 재현: 과거 {historical:.6f}, 이번 {actual:.6f}, 재현 차이 {actual-historical:+.6f}. 모든 가설 Δ는 이번 기준선에 대한 값이다.",
              "C_emb_PCA k=32의 과거 46.062757은 임베딩 cache 부재로 재현하지 않았다. 사용자 지시에 따라 새 임베딩 생성 없이 H4_prompt_sequence도 건너뛰었다.", ""]
    outer = result["split_audits"]["outer"]["after"]
    lines += [f'정제 train {outer["earlier"]["samples"]:,}행/{outer["earlier"]["users"]:,}사용자, VAL {outer["later"]["samples"]:,}행/{outer["later"]["users"]:,}사용자. 사용자·그룹·source 교집합 0.', ""]
    lines += [f"- {h}: {definition}" for h, definition in HYPOTHESES.items()]
    lines += ["", "평균 P10/P50/P90 pinball (낮을수록 좋음). 사용자 가중은 각 구분 안의 사용자별 평균을 동일 가중.", "",
              "| 턴 | 가중 | " + " | ".join(causal.MODELS) + " |", "|---|---|" + "---:|"*len(causal.MODELS)]
    for cohort in ("all", "first_turn", "followup_turn"):
        for weight in ("row", "user"):
            scores = result["validation_metrics"][cohort][weight]
            lines.append(f"| {cohort} | {weight} | " + " | ".join(f'{scores[m]["mean_pinball_loss"]:.6f}' for m in causal.MODELS) + " |")
    lines += ["", "H−동일 기준선 차이 / 사용자 paired bootstrap 95% CI (1,000회·PCG64 seed 42·공유 추출·percentile 선형 보간):", "",
              "| H | 턴 | 가중 | 차이 | 95% CI |", "|---|---|---|---:|---|"]
    for cohort in ("all", "first_turn", "followup_turn"):
        for weight in ("row", "user"):
            for label, d in result["bootstrap_ci95"][cohort][weight].items():
                lo, hi = d["ci95"]
                lines.append(f'| {label} | {cohort} | {weight} | {d["point_difference"]:.6f} | [{lo:.6f}, {hi:.6f}] |')
    lines += ["", f'권장: **{result["recommended"]}**. 전체 행 가중 CI가 0 아래인 후보 중 가장 낮은 점추정을 탐색적 권장으로 삼고, 없으면 기준선을 유지한다.',
              "H1은 전체 행 가중과 후속 턴의 행/사용자 가중에서 작은 개선을 보였다. 전체 사용자 가중과 첫 턴의 H1 CI는 0을 포함한다. H2 전체 차이 CI는 0을 포함하고 H3는 손실이 증가했다.",
              "시간 구조를 추가하려면 H1의 대화 내 수준·추세·출력/입력 비율 요약을 우선 권장한다. 묶음 특징 실험이므로 EWMA 하나의 효과로 분해할 수 없다. C_emb_PCA에 H1을 추가하는 효과는 아직 검증하지 않았다.",
              "반복 사용한 VAL의 탐색 결과이며 다중비교 보정·새로운 일반화 증거는 아니다. 사용자 bootstrap은 고정 모델에 조건부이고 시간 블록·학습 불확실성을 반영하지 않는다.",
              "입력 시각은 이전 assistant 완료의 하한이고 현재 요청 제출 시각이 아니다. 첫 턴에 시각이 없으므로 달력 특징을 제공하지 않는다. 모델 버전·대화 마지막 시각을 입력에 사용하지 않았다.", "",
              "실행 시간(초): " + json.dumps(result["runtime_seconds"]), "",
              "실행 환경: " + json.dumps(result["environment"]), "",
              "파일: `wildchat/temporal_hypotheses.py`, `wildchat/temporal_causal.py`, `wildchat/temporal_baseline.json`, `wildchat/tests/test_temporal_hypotheses.py`, `wildchat/results/temporal_hypotheses.json`, `wildchat/results/temporal_hypotheses.md`, `wildchat/results/temporal_hypotheses_summary.md`."]
    if "verification" in result:
        lines += ["", "검증·문제: " + result["verification"]["summary"], "",
                  "기존 docs/project.md·wildchat/README.md는 사용자의 미커밋 변경 보존을 위해 수정하지 않았다. 이 VAL 전용 후속 실험의 정의·재현·결과는 요청된 두 결과 MD와 JSON에 둔다.", "",
                  "Notion: 기존 WildChat 실험 페이지의 쓰기는 직전 시도에서 승인 정책(approval policy: never)으로 차단됐다. 이번 실제 결과 요약도 미저장이며 완료된 로컬 실험과 구분한다. 미저장 인계안: “2026-11-06 VAL에서 B 재현, H1 대화 내 요약에 소폭 이득, H2 추가 개선 미확인·H3 손실 증가. H4는 cache 부재로 skip. 정본 wildchat/results/temporal_hypotheses.md 및 summary.md, 미커밋 로컬 산출물. 전체 사용자 가중 H1 CI는 0 포함. test 접근·Git 게시 없음.”"]
    if not summary:
        lines += ["", "## 누출 방지·재현", "",
                  "- `rv.load_allowed`의 train.jsonl/validation.jsonl 고정 허용 목록과 `purge_and_check`, `inner_split` 재사용. 사용자·그룹·source 교집합 0을 실행 검사한다.",
                  "- #21 `HistoryRows`, `temporal_metadata`, `evaluate`를 재사용했다. 같은 metadata projection에서 불필요한 header/country/model 컬럼을 제외했다.",
                  "- 이전 유효 턴의 응답/프롬프트만 스냅샷 후 상태에 반영한다. 원본 source별 상태, 현재·미래 응답/다른 대화·다른 분할 미사용. H1 이력은 lag slot이 아닌 이전 유효 표본 전체다.",
                  "- raw shard에서 hashed_ip/recorded timestamp/message role·timestamp만 projection한다. Arrow metadata batch는 압축 단위로 다른 원본 행을 포함할 수 있지만, train/VAL source ID만 take한 뒤 Python 객체로 만든다. raw text/응답 target 컬럼과 test.jsonl은 요청하지 않는다.",
                  "- 모든 가설의 파라미터/반복 수는 기존 B의 train 내부 선택에 고정했다. 새 sweep·VAL 조기 종료·VAL 재튜닝이 없다. H1은 이전 유효 응답 전체, H2는 이미 끝난 두 응답 사이 cadence이며 현재 요청까지의 간격이 아니다.",
                  "- H1 EWMA는 첫 관측값으로 초기화하고 α=.3으로 갱신한다. std는 ddof=0, 차이·기울기는 직전 두 유효 응답 기준(기울기 분모는 원래 turn_index 차이). 출력/입력과 현재/이전 입력 비율의 분모는 max(이전 입력 토큰,1)이다.",
                  "- H2 gap은 마지막 두 과거 assistant 완료 시각 차이, span은 첫 과거 완료부터 마지막 과거 완료까지다. 현재 assistant 시간은 사용하지 않는다. H3는 anchor의 epoch day·월·UTC hour/weekday sin/cos와 hour×log1p 현재 입력이다.",
                  "- 첫 턴에는 과거 특징이 결측이지만 전체 모델을 재학습하므로 첫 턴 예측도 소폭 변할 수 있다. H3의 악화는 이 고정 특징·모델의 결과이며 시간 drift가 없다는 증명은 아니다.",
                  "- 전체 기계 판독 결과는 temporal_hypotheses.json: P10/P50/P90 개별 loss, coverage, crossing, split 검사, 입력/코드 해시, 환경·시간 포함.",
                  "", "문제·제한: 브랜치 생성이 sandbox의 .git 쓰기 제한으로 실패하여 main의 새 파일만 사용했다. 기존 사용자 변경을 수정하지 않았다. Notion/기존 정본 MD는 사용자 미커밋 파일 보존 제약에 따라 별도 인계한다.",
                  "", "```sh", result["command"], "```", "", "```json", json.dumps(result["split_audits"], ensure_ascii=False, indent=2), "```"]
        lines += ["", "## 메타데이터·특징 감사", "", "```json", json.dumps(result["metadata_audit"], ensure_ascii=False, indent=2), "```",
                  "", "```json", json.dumps(result["feature_audit"], ensure_ascii=False, indent=2), "```"]
        lines += ["", "## 표본 수·분위수별 pinball", "",
                  "| 턴 | 가중 | 조건 | 행/사용자 | P10 loss | P50 loss | P90 loss |", "|---|---|---|---|---:|---:|---:|"]
        for cohort in ("all", "first_turn", "followup_turn"):
            for weight in ("row", "user"):
                for model, score in result["validation_metrics"][cohort][weight].items():
                    lines.append(f'| {cohort} | {weight} | {model} | {score["n_rows"]}/{score["n_users"]} | {score["pinball_p10"]:.6f} | {score["pinball_p50"]:.6f} | {score["pinball_p90"]:.6f} |')
        if "verification" in result:
            lines += ["", "## 환경 준비·검증 기록", "", "```json", json.dumps(result["verification"], ensure_ascii=False, indent=2), "```",
                      "", "추가 생성 파일: wildchat/results/temporal_hypotheses.log, temporal_hypotheses_DONE.txt. 기존 파일 삭제 없음. 이번에 만들었던 설치 실패용 빈 venv와 임시 파일만 정리했다. 실제 실행 환경 /private/tmp/venv-temporal은 기존 환경이므로 보존했다."]
    return "\n".join(lines)+"\n"


def run(args):
    started = time.monotonic()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    prior_path = Path(__file__).with_name("temporal_baseline.json")
    prior = json.loads(prior_path.read_text())
    names = feature_names(manifest["lags"])["B"]
    if len(names) != 31:
        raise ValueError("baseline requires exactly 31 B features")
    parts, hashes = rv.load_allowed(args.prepared, manifest, names)
    if hashes != prior["prepared_sha256"]:
        raise ValueError("prepared inputs differ from fixed historical VAL protocol")
    parts, metadata_audit = attach_source_metadata(parts, args.prepared, args.source, manifest["input_files"][0])
    cutoff = rv.timestamp_us(manifest["boundaries"]["train_validation_cutoff"])
    train, validation, outer = rv.purge_and_check(parts["train"], parts["validation"], cutoff)
    _, _, inner = rv.inner_split(train, .8)
    if len(train.y) != 81307 or len(validation.y) != 12920:
        raise ValueError("split does not reproduce the documented population")
    td = build_designs(train, len(names))
    vd = build_designs(validation, len(names))
    feature_audit = {}
    for split, rows, designs in (("train", train, td), ("validation", validation, vd)):
        feature_audit[split] = {m: {"features": int(x.shape[1]),
            "first_turn_rows": int((rows.turns == 0).sum()),
            "anchor_present_rows": int((rows.anchors != causal.MISSING_TIME).sum()),
            "new_numeric_nonmissing_rows": int(np.any(np.isfinite(x[:, len(names):-1]), axis=1).sum())}
            for m, x in designs.items() if m != "baseline"}
    runtime = {"loading_features": time.monotonic()-started}
    predictions, diagnostics = {}, {}
    for model in causal.MODELS:
        t = time.monotonic()
        fn = names + ([] if model == "baseline" else FEATURES[model])
        predictions[model], diagnostics[model] = rv.refit_predict(
            replace(train, x=td[model]), replace(validation, x=vd[model]),
            list(range(len(fn))), fn, prior["selected_B"], args)
        runtime[model] = time.monotonic()-t
        print(json.dumps({"model": model, "seconds": runtime[model], "val_all_row_pinball": float(rv.contributions(validation.y, predictions[model])[:, 0].mean())}), flush=True)
    t = time.monotonic()
    metrics, bootstrap = causal.evaluate(validation, predictions, np.ones(len(validation.y), dtype=bool), 42, 1000)
    runtime["evaluation_bootstrap"] = time.monotonic()-t
    runtime["total"] = time.monotonic()-started
    deltas = bootstrap["all_users"]["all"]["row"]
    improved = [m for m in HYPOTHESES if deltas[m+"-baseline"]["ci95"][1] < 0]
    recommended = min(improved, key=lambda m: metrics["all_users"]["all"]["row"][m]["mean_pinball_loss"]) if improved else "baseline"
    inputs = {"manifest": args.prepared/"manifest.json",
              "baseline": prior_path, "runner": Path(__file__), "causal_helpers": Path(causal.__file__),
              "revalidate": Path(rv.__file__), "train_code": Path(__file__).with_name("train.py"),
              "prepare_code": Path(__file__).with_name("prepare.py")}
    result = {"status": "completed", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": " ".join(__import__("shlex").quote(s) for s in [sys.executable, *sys.argv]),
              "seed": 42, "bootstrap_replicates": 1000, "evaluation_split": "validation only",
              "hypotheses": HYPOTHESES, "feature_names": FEATURES, "prepared_sha256": hashes,
              "input_code_sha256": {k: rv.fingerprint(p) for k, p in inputs.items()},
              "environment": {p: importlib.metadata.version(p) for p in ("numpy", "lightgbm", "pyarrow", "scipy")},
              "baseline_name": "B", "selected_parameters": prior["selected_B"]["selected_parameters"],
              "selected_rounds": [f["best_iteration"] for f in prior["selected_B"]["selected_quantile_fits"]],
              "metadata_audit": metadata_audit, "feature_audit": feature_audit,
              "source_preparation_sha256": manifest["input_files"][0]["sha256"],
              "source_hash_verification": "manifest provenance only; raw shard not rehashed",
              "skipped": {"H4_prompt_sequence": "No embedding cache; no new embedding calls per user instruction"},
              "runtime_seconds": runtime, "split_audits": {"outer": outer, "inner": inner},
              "validation_metrics": metrics["all_users"], "bootstrap_ci95": bootstrap["all_users"],
              "prediction_diagnostics": diagnostics, "recommended": recommended,
              "historical_baseline_all_row": prior["baseline_B_historical"]["all"]["row"]["B"]["mean_pinball_loss"]}
    for name, content in (("temporal_hypotheses.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"),
                          ("temporal_hypotheses.md", render(result)),
                          ("temporal_hypotheses_summary.md", render(result, True))):
        (args.output/name).write_text(content)
    # Completion sentinel is deliberately last, only after successful real VAL.
    (args.output/"temporal_hypotheses_DONE.txt").write_text("DONE "+recommended+"\n")


def main():
    def deny_heldout_open(event, event_args):
        if event == "open" and event_args and isinstance(event_args[0], (str, bytes)):
            name = event_args[0].decode() if isinstance(event_args[0], bytes) else event_args[0]
            if Path(name).name == "test.jsonl":
                raise PermissionError("held-out split access is forbidden")
    sys.addaudithook(deny_heldout_open)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="raw shard, metadata columns only")
    parser.add_argument("--output", type=Path, default=Path("wildchat/results"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.seed = 42
    if args.threads < 1:
        parser.error("threads must be positive")
    check = preflight(args)
    print(json.dumps(check), flush=True)
    if not check["ready"] or args.check:
        return 0 if check["ready"] else 2
    for name in ("temporal_hypotheses.json", "temporal_hypotheses_DONE.txt"):
        if (args.output/name).exists():
            parser.error("completed output already exists; use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
