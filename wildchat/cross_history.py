#!/usr/bin/env python3
"""Issue 21: causal cross-conversation history, train selection, validation report.

No new sampling, prompt text, user IDs in outputs, or held-out split access.
History is rebuilt independently after each outer/inner purge. A conversation
becomes observable only at its final assistant completion. Since first prompts
have no submission timestamp, only an earlier assistant in the SAME conversation
can supply a conservative prediction-time lower bound. No current response time
or conversation-final time is used as a prediction feature.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import shlex
import time

import numpy as np

from wildchat import revalidate as rv
from wildchat.category_pca import PCA
from wildchat.prepare import timestamp_us
from wildchat.train import SEARCH_SPACE, feature_names

MODELS = ("B", "C_emb_PCA", "B_cross_hist", "C_emb_PCA_cross_hist")
COMPARISONS = (("B_cross_hist", "B"), ("C_emb_PCA_cross_hist", "C_emb_PCA"))
KS = (8, 16, 32)
HISTORY_NAMES = ["cross_last_mean", "cross_last_median", "cross_ewma_mean",
                 "cross_log1p_ewma_mean", "cross_conversation_count", "cross_gap_seconds",
                 "cross_history_missing", "cross_anchor_missing", "cross_anchor_hour_utc",
                 "cross_anchor_weekday_utc"]
EWMA_ALPHA = .3
MISSING_TIME = np.iinfo(np.int64).min


@dataclass
class HistoryRows(rv.Rows):
    indices: np.ndarray
    anchors: np.ndarray
    completions: np.ndarray
    embeddings: np.ndarray


def temporal_metadata(messages, recorded):
    """Prefix-only anchors; future quality checks affect publication only."""
    stamps, anchors, previous, valid_prefix = [], {}, None, True
    for j, message in enumerate(messages):
        stamp = timestamp_us(message.get("timestamp"))
        if message.get("role") == "user":
            anchors[j] = previous if valid_prefix else None
        elif message.get("role") == "assistant":
            valid_prefix = valid_prefix and stamp is not None and (previous is None or previous <= stamp)
            stamps.append(stamp)
            previous = stamp
    complete = (bool(stamps) and valid_prefix and recorded is not None and max(stamps) == recorded)
    return anchors, recorded if complete else None


def attach_metadata(parts, prepared, source, expected_source):
    """Project only metadata leaves, then materialize only allowed source IDs."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if source.name != expected_source["name"]:
        raise ValueError("source filename differs from preparation provenance")
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != expected_source["rows"]:
        raise ValueError("source row count differs from preparation provenance")
    # Retain the exact prepared row/message alignment, never infer it from turns.
    indices = {}
    for split, rows in parts.items():
        with (prepared / (split + ".jsonl")).open("rb") as handle:
            indices[split] = np.asarray([json.loads(line)["metadata"]["user_message_index"]
                                         for line in handle], dtype=np.int64)
        if len(indices[split]) != len(rows.y):
            raise ValueError("prepared metadata alignment mismatch")
    allowed = set().union(*(set(rows.sources.tolist()) for rows in parts.values()))
    projected = ["hashed_ip", "timestamp", "header", "country", "state", "model",
                 "conversation.list.element.role", "conversation.list.element.timestamp"]
    mapping, offset = {}, 0
    audit = {"projected_columns": projected, "selected_source_rows": len(allowed),
             "metadata_missing": {k: 0 for k in ("hashed_ip", "header", "country", "state", "model")},
             "user_messages": 0, "user_timestamps_present": 0,
             "assistant_messages": 0, "assistant_timestamps_missing": 0,
             "nonmonotonic_conversations": 0}
    for batch in parquet.iter_batches(batch_size=1024, columns=projected):
        selected_ids = [i for i in range(len(batch)) if offset + i in allowed]
        selected = batch.take(pa.array(selected_ids, type=pa.int64())).to_pylist()
        for i, meta in zip(selected_ids, selected):
            for key in audit["metadata_missing"]:
                audit["metadata_missing"][key] += int(not meta[key])
            user = meta["hashed_ip"]
            user = user if isinstance(user, str) and user.strip() else ""
            stamps = []
            messages = meta["conversation"] or []
            for message in messages:
                stamp = timestamp_us(message.get("timestamp"))
                role = message.get("role")
                if role == "user":
                    audit["user_messages"] += 1
                    audit["user_timestamps_present"] += int(stamp is not None)
                elif role == "assistant":
                    audit["assistant_messages"] += 1
                    audit["assistant_timestamps_missing"] += int(stamp is None)
                    stamps.append(stamp)
            good = bool(stamps) and all(t is not None for t in stamps)
            monotonic = good and all(a <= b for a, b in zip(stamps, stamps[1:]))
            audit["nonmonotonic_conversations"] += int(good and not monotonic)
            recorded = timestamp_us(meta["timestamp"])
            # Later timestamps may gate publication of a completed conversation,
            # NEVER the anchor of an earlier prediction in that conversation.
            anchors, completion = temporal_metadata(messages, recorded)
            mapping[offset + i] = (user, recorded, anchors, completion)
        offset += len(batch)
    if set(mapping) != allowed:
        raise ValueError("source IDs unavailable in declared source")
    result = {}
    for split, rows in parts.items():
        users, anchors, completions = [], [], []
        for source_id, stamp, index in zip(rows.sources, rows.stamps, indices[split]):
            user, recorded, times, completion = mapping[int(source_id)]
            if recorded != int(stamp) or (times and int(index) not in times):
                raise ValueError("source metadata differs from prepared identity")
            users.append(user)
            anchor = times.get(int(index))
            anchors.append(MISSING_TIME if anchor is None else anchor)
            completions.append(MISSING_TIME if completion is None else completion)
        result[split] = HistoryRows(**{**vars(rows), "users": np.asarray(users, dtype=object)},
                                    indices=indices[split], anchors=np.asarray(anchors, dtype=np.int64),
                                    completions=np.asarray(completions, dtype=np.int64),
                                    embeddings=np.empty((len(rows.y), 0)))
    print(json.dumps({"metadata_audit": audit}), flush=True)
    return result, audit


def attach_embeddings(parts, path):
    with np.load(path, allow_pickle=False) as archive:
        keys, vectors = archive["keys"], archive["vectors"]
        if (keys.ndim != 2 or keys.shape[1] != 2 or vectors.ndim != 2 or
                len(keys) != len(vectors) or vectors.shape[1] != 384 or not np.isfinite(vectors).all()):
            raise ValueError("invalid cached MiniLM embeddings")
        index = {tuple(map(int, key)): i for i, key in enumerate(keys)}
        if len(index) != len(keys):
            raise ValueError("duplicate embedding identities")
        for rows in parts.values():
            try:
                positions = [index[(int(s), int(i))] for s, i in zip(rows.sources, rows.indices)]
            except KeyError:
                raise ValueError("missing allowed embedding identity") from None
            rows.embeddings = vectors[positions].astype(np.float32)


def cross_features(rows, split):
    """Full completed other-conversation summaries, strictly before each anchor.

    One event per (user, original duplicate group), represented by its earliest
    completed source. All targets used in a summary belong to that source and
    THIS partition. The current source AND its duplicate group are excluded.
    Rows with unknown anchors have no historical or calendar information.
    """
    by_source = defaultdict(list)
    for i, source in enumerate(rows.sources):
        by_source[int(source)].append(i)
    unique = {}
    for source, ix in by_source.items():
        users, groups = set(rows.users[ix]), set(rows.groups[ix])
        completions = set(rows.completions[ix])
        if len(users) != 1 or len(groups) != 1 or len(completions) != 1 or "" in users:
            raise ValueError("inconsistent source identity in history partition")
        stamp = int(rows.completions[ix[0]])
        if stamp == MISSING_TIME:
            continue
        user, group = rows.users[ix[0]], int(rows.groups[ix[0]])
        event = (stamp, source, group, float(rows.y[ix].mean()), float(np.median(rows.y[ix])), split)
        key = (user, group)
        if key not in unique or event[:2] < unique[key][:2]:
            unique[key] = event
    events = defaultdict(list)
    for (user, _), event in unique.items():
        events[user].append(event)
    for values in events.values():
        values.sort()
    x = np.full((len(rows.y), len(HISTORY_NAMES)), np.nan)
    x[:, 4], x[:, 6], x[:, 7] = 0, 1, 1
    checked, available, minimum_gap = 0, np.zeros(len(rows.y), dtype=bool), None
    for i, (user, source, group, anchor) in enumerate(zip(rows.users, rows.sources, rows.groups, rows.anchors)):
        if anchor == MISSING_TIME:
            continue
        x[i, 7] = 0
        dt = datetime.fromtimestamp(int(anchor) / 1e6, tz=timezone.utc)
        x[i, 8:] = dt.hour, dt.weekday()
        past = []
        for event in events[user]:
            stamp, other_source, other_group, mean, median, origin = event
            if stamp >= anchor:
                break
            if other_source == source or other_group == group:
                continue
            # Executable provenance assertions; no identities are emitted.
            if (origin != split or other_source not in by_source or
                    rows.users[by_source[other_source][0]] != user or
                    other_source == source or other_group == group or not stamp < anchor):
                raise AssertionError("cross-conversation leakage detected")
            past.append(event)
            checked += 1
        if not past:
            continue
        ewma = past[0][3]
        for event in past[1:]:
            ewma = EWMA_ALPHA * event[3] + (1 - EWMA_ALPHA) * ewma
        gap = (int(anchor) - past[-1][0]) / 1e6
        x[i, :7] = past[-1][3], past[-1][4], ewma, np.log1p(ewma), len(past), gap, 0
        available[i] = True
        minimum_gap = gap if minimum_gap is None else min(gap, minimum_gap)
    audit = {"partition": split, "rows": len(rows.y), "history_rows": int(available.sum()),
             "history_users": len(set(rows.users[available])),
             "unknown_anchor_rows": int((rows.anchors == MISSING_TIME).sum()),
             "eligible_completed_conversation_groups": len(unique), "checked_history_links": checked,
             "minimum_history_gap_seconds": minimum_gap, "violations": 0,
             "checks": {"same_partition_only": True, "same_user_only": True,
                        "strictly_prior_completion": True, "current_source_and_group_excluded": True}}
    print(json.dumps({"history_leakage_audit": audit}), flush=True)
    return x, available, audit


def select_condition(fit, holdout, fit_hist, hold_hist, pcas, model, base_names, args):
    use_pca, use_history = "PCA" in model, "cross_hist" in model
    trials, best, best_score = {}, None, float("inf")
    for k in (KS if use_pca else (0,)):
        f, h = fit.x, holdout.x
        names = list(base_names)
        if use_pca:
            f, h = np.column_stack((f, pcas[0][:, :k])), np.column_stack((h, pcas[1][:, :k]))
            names += [f"pc{i + 1}" for i in range(k)]
        if use_history:
            f, h = np.column_stack((f, fit_hist)), np.column_stack((h, hold_hist))
            names += HISTORY_NAMES
        print(f"Selecting {model}, PCA k={k}, train-internal only", flush=True)
        selected, _ = rv.select_model(replace(fit, x=f), replace(holdout, x=h), list(range(len(names))), names, args)
        score = selected["trials"][selected["selected_trial"]]["inner_row_metrics"]["mean_pinball_loss"]
        trials[str(k)] = selected
        if score < best_score:
            best, best_score = k, score
    return {"chosen_k": best, "inner_mean_pinball": best_score, "k_trials": trials,
            "selected": trials[str(best)]}


def evaluate(rows, predictions, available, seed, replicates):
    """All metrics and paired user bootstrap, same draws for every subgroup."""
    users, codes = np.unique(rows.users, return_inverse=True)
    g = len(users)
    history_users = set(rows.users[available])
    history_mask = np.isin(rows.users, list(history_users))
    populations = {"all_users": np.ones(len(rows.y), dtype=bool),
                   "users_with_history": history_mask, "users_without_history": ~history_mask}
    masks = {(p, c): pm & cm for p, pm in populations.items() for c, cm in rv.cohorts(rows).items()}
    values = np.stack([rv.contributions(rows.y, predictions[m]) for m in MODELS], axis=1)
    metrics, aggregates = {}, {}
    for (population, cohort), mask in masks.items():
        counts = np.bincount(codes[mask], minlength=g)
        sums = np.zeros((g, len(MODELS), len(rv.METRICS)))
        np.add.at(sums, codes[mask], values[mask])
        means = np.divide(sums, counts[:, None, None], out=np.zeros_like(sums), where=counts[:, None, None] > 0)
        aggregates[(population, cohort)] = counts, sums[:, :, 0], means[:, :, 0]
        entry = metrics.setdefault(population, {}).setdefault(cohort, {})
        for weight, scores, denominator in (("row", sums, counts.sum()), ("user", means, (counts > 0).sum())):
            entry[weight] = {m: {"n_rows": int(mask.sum()), "n_users": int((counts > 0).sum()),
                                 **dict(zip(rv.METRICS, map(float, scores[:, j].sum(axis=0) / denominator)))}
                             if denominator else None for j, m in enumerate(MODELS)}
    draws = {(p, c, w, a, b): [] for p, c in masks for w in ("row", "user") for a, b in COMPARISONS}
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(replicates):
        weights = np.bincount(rng.integers(0, g, size=g), minlength=g)
        for (p, c), (counts, sums, means) in aggregates.items():
            for w, data, den in (("row", sums, weights @ counts), ("user", means, weights @ (counts > 0))):
                if not den:
                    continue
                for a, b in COMPARISONS:
                    delta = weights @ (data[:, MODELS.index(a)] - data[:, MODELS.index(b)]) / den
                    draws[(p, c, w, a, b)].append(float(delta))
    boot = {}
    for (p, c, w, a, b), sample in draws.items():
        ms = metrics[p][c][w]
        boot.setdefault(p, {}).setdefault(c, {}).setdefault(w, {})[f"{a}-{b}"] = {
            "point_difference": ms[a]["mean_pinball_loss"] - ms[b]["mean_pinball_loss"] if ms[a] else None,
            "effective_replicates": len(sample), "empty_replicates": replicates - len(sample),
            "ci95": np.quantile(sample, [.025, .975], method="linear").tolist() if sample else None}
    return metrics, boot


def render(result):
    lines = ["# #21 대화 간 이력 재검증 (validation 전용)", "",
             "test 자료 접근·최종 평가 재실행 없이 기존 표본과 사용자 분리를 유지했다.",
             "네 조건 모두 같은 3개 설정·조기 종료 규칙으로 train 내부에서 선택했다. PCA k∈{8,16,32}도 내부 선택이다.",
             "전체 train에서 선택 반복 수로 재학습했다. validation은 보고 전용이며 분위수를 정렬하지 않았다.", "",
             "## 주지표 결론", ""]
    for a, b in COMPARISONS:
        d = result["bootstrap_ci95"]["all_users"]["all"]["row"][f"{a}-{b}"]
        lo, hi = d["ci95"]
        claim = "95% CI가 0 아래로 개선을 지지한다" if hi < 0 else (
            "95% CI가 0 위로 악화를 지지한다" if lo > 0 else "95% CI가 0을 포함해 개선을 확인하지 못했다")
        lines.append(f'- {a} − {b}: {d["point_difference"]:.6f}, 95% CI [{lo:.6f}, {hi:.6f}]. {claim}.')
    lines += ["",
             "## 인과적 시각과 사용자 정의", "",
             "- 사용자 키는 기존 분리와 동일한 top-level hashed_ip. header는 결합하지 않는다(공유 IP·IP 변경 한계).",
             "- UTC assistant timestamp는 응답 완료 시각이다. 현재 응답 시각은 입력으로 쓰지 않는다.",
             "- 각 행의 기준 시각은 같은 대화의 앞선 assistant 완료 시각이다. 요청 시각의 보수적 하한이다.",
             "- user timestamp가 검증된 제출 시각이 아니므로 사용하지 않는다. 첫 턴·시각 비정상 행은 기준 시각/이력이 결측이다.",
             "- 같은 split에서 기준 시각보다 엄격히 먼저 완료된 다른 원본 대화 그룹만 쓴다. 진행 중 대화·같은 시각·현재 중복 그룹은 제외한다.",
             "- 중복 그룹은 사용자별 가장 먼저 완료된 원본 행 하나로 집계한다. 평균/중앙값은 고정 표본의 유효 응답들만 포함한다.",
             "- EWMA α=0.3, 직전 대화 평균/중앙값, EWMA/log1p EWMA, 완료 대화 수, 초 간격, UTC 시간대/요일, 결측 지표를 추가했다.",
             "- header(accept-language/user-agent), country/state(IP 지역), model(기록 모델)은 결측 집계만 감사하고 모델 입력에는 넣지 않는다.",
             "- 이력 사용자 = validation 내 적어도 한 행에 인과적 이력이 있는 사용자. 이 사용자의 첫 턴도 해당 사용자 표에 포함한다.",
             "- validation 이력은 순차 관측을 가정한다. 앞선 validation 정답은 완료 후 이력으로만 사용하며 학습·선택에는 사용하지 않는다.", "",
             "## 내부 선택", "", "| 조건 | PCA k | 내부 평균 pinball | P10/P50/P90 반복 수 |", "|---|---:|---:|---|"]
    for m, s in result["selection"].items():
        iterations = "/".join(str(f["best_iteration"]) for f in s["selected"]["selected_quantile_fits"])
        lines.append(f'| {m} | {s["chosen_k"]} | {s["inner_mean_pinball"]:.4f} | {iterations} |')
    lines += ["", "## validation 평균 pinball", "", "| 사용자 구분 | 턴 | 가중 | " + " | ".join(MODELS) + " |",
              "|---|---|---|" + "---:|" * len(MODELS)]
    for pop, cohorts in result["validation_metrics"].items():
        for cohort, weights in cohorts.items():
            for w, ms in weights.items():
                lines.append(f"| {pop} | {cohort} | {w} | " + " | ".join(
                    f'{ms[m]["mean_pinball_loss"]:.4f}' if ms[m] else "NA" for m in MODELS) + " |")
    lines += ["", "## 사용자 bootstrap 95% CI (차이 < 0이면 개선)", "",
              "| 사용자 구분 | 턴 | 가중 | 차이 | 점추정 | 95% CI |", "|---|---|---|---|---:|---|"]
    for pop, cohorts in result["bootstrap_ci95"].items():
        for cohort, weights in cohorts.items():
            for w, comparisons in weights.items():
                for label, d in comparisons.items():
                    bounds = d["ci95"]
                    value = f'{d["point_difference"]:.4f}' if bounds else "NA"
                    ci = f"[{bounds[0]:.4f}, {bounds[1]:.4f}]" if bounds else "NA"
                    lines.append(f"| {pop} | {cohort} | {w} | {label} | {value} | {ci} |")
    lines += ["", "## 전체 지표 (P10/P50/P90 고유 순서 유지)", "",
              "| 사용자 구분 | 턴 | 가중 | 조건 | 행/사용자 | 평균 pinball | P10 loss | P50 loss | P90 loss | P50 MAE | coverage | 부호 있는 폭 | 교차율 |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pop, cohorts in result["validation_metrics"].items():
        for cohort, weights in cohorts.items():
            for w, ms in weights.items():
                for m, metrics in ms.items():
                    cells = (f'{metrics["n_rows"]}/{metrics["n_users"]} | ' + " | ".join(
                        f"{metrics[k]:.6f}" for k in rv.METRICS)) if metrics else "NA | " + " | ".join(["NA"] * len(rv.METRICS))
                    lines.append(f"| {pop} | {cohort} | {w} | {m} | {cells} |")
    lines += ["", "## 실행·누수 검사", "", "```json", json.dumps(result["history_audits"], ensure_ascii=False, indent=2), "```", "",
              "```sh", result["command"], "```", "",
              "- JSON에 메타데이터 감사, split 교집합, 모든 후보·선택 설정, 입력/코드 해시, 환경을 기록한다.",
              "- 기존 46.3373은 B의 고정 설정으로 PCA만 선택한 #19 결과다. 이번 네 조건은 동일한 내부 재튜닝 절차이므로 기준 점수가 달라질 수 있다.",
              "- 첫 샤드 고정 표본이며 반복 사용된 validation이다. CI는 고정 모델에 조건부이며 학습·선택 불확실성과 IP 식별 오차를 반영하지 않는다.",
              "- 첫 턴의 과거 대화 활용 효과는 제출시각 부재로 검증할 수 없다. 실제 API 사용량·GPU 효과로 일반화하지 않는다.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    manifest_path = args.prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = feature_names(manifest["lags"])["B"]
    parts, hashes = rv.load_allowed(args.prepared, manifest, names)
    parts, metadata = attach_metadata(parts, args.prepared, args.source, manifest["input_files"][0])
    attach_embeddings(parts, args.embeddings)
    cutoff = timestamp_us(manifest["boundaries"]["train_validation_cutoff"])
    train, validation, outer = rv.purge_and_check(parts["train"], parts["validation"], cutoff)
    fit, holdout, inner = rv.inner_split(train, args.inner_fraction)
    histories, available, audits = {}, {}, {}
    for split, rows in (("inner_fit", fit), ("inner_holdout", holdout), ("train", train), ("validation", validation)):
        histories[split], available[split], audits[split] = cross_features(rows, split)
    print("Fitting PCA on inner-fit rows only", flush=True)
    pca = PCA().fit(fit.embeddings, max(KS))
    inner_pcs = pca.transform(fit.embeddings), pca.transform(holdout.embeddings)
    selected = {m: select_condition(fit, holdout, histories["inner_fit"], histories["inner_holdout"],
                                    inner_pcs, m, names, args) for m in MODELS}
    # No validation targets or feature matrices are passed to selection above.
    full_pca = PCA().fit(train.embeddings, max(KS))
    pcs = full_pca.transform(train.embeddings), full_pca.transform(validation.embeddings)
    predictions, diagnostics = {}, {}
    for m in MODELS:
        tx, vx, fn = train.x, validation.x, list(names)
        k = selected[m]["chosen_k"]
        if k:
            tx, vx = np.column_stack((tx, pcs[0][:, :k])), np.column_stack((vx, pcs[1][:, :k]))
            fn += [f"pc{i + 1}" for i in range(k)]
        if "cross_hist" in m:
            tx, vx = np.column_stack((tx, histories["train"])), np.column_stack((vx, histories["validation"]))
            fn += HISTORY_NAMES
        print(f"Refitting {m} on full train with fixed selected rounds", flush=True)
        predictions[m], diagnostics[m] = rv.refit_predict(replace(train, x=tx), replace(validation, x=vx),
                                                         list(range(len(fn))), fn, selected[m]["selected"], args)
    metrics, bootstrap = evaluate(validation, predictions, available["validation"], args.seed, args.bootstrap_replicates)
    result = {"issue": 21, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_split": "validation only", "command": shlex.join([sys.executable, *sys.argv]),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "prepared_sha256": hashes, "manifest_sha256": rv.fingerprint(manifest_path),
              "source_preparation_sha256": manifest["input_files"][0].get("sha256"),
              "embeddings_sha256": rv.fingerprint(args.embeddings),
              "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): rv.fingerprint(p)
                              for p in (Path(__file__).resolve(), Path(rv.__file__), Path(__file__).with_name("category_pca.py"),
                                        Path(__file__).with_name("train.py"), Path(__file__).with_name("prepare.py"))},
              "environment": {p: importlib.metadata.version(p) for p in ("numpy", "lightgbm", "pyarrow", "tiktoken")},
              "features": {"B": names, "cross_history": HISTORY_NAMES, "ewma_alpha": EWMA_ALPHA},
              "search_space": SEARCH_SPACE, "metadata_audit": metadata,
              "split_audits": {"outer": outer, "inner": inner}, "history_audits": audits,
              "selection": selected, "pca_explained_variance_full_train": full_pca.explained,
              "prediction_diagnostics": diagnostics, "validation_metrics": metrics,
              "bootstrap": {"unit": "user", "seed": args.seed, "replicates": args.bootstrap_replicates,
                            "groups_per_draw": len(set(validation.users)), "shared_draws_all_subgroups": True,
                            "refit_or_reselect": False, "interval": "percentile 2.5/97.5, linear"},
              "bootstrap_ci95": bootstrap, "elapsed_seconds": time.monotonic() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cross_history.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "cross_history.md").write_text(render(result))
    print("Wrote cross_history.json and cross_history.md", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "source", "embeddings", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-fraction", type=float, default=.8)
    parser.add_argument("--max-rounds", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if not (0 < args.inner_fraction < 1) or min(args.max_rounds, args.early_stopping_rounds, args.threads, args.bootstrap_replicates) < 1:
        parser.error("positive run settings and inner fraction in (0,1) required")
    args.output.mkdir(parents=True, exist_ok=True)
    # Preserve a prior completed run; a rerun needs an explicitly different output directory.
    for name in ("cross_history.json", "cross_history.md", "cross_history.log"):
        if (args.output / name).exists():
            parser.error("cross-history output already exists; choose a fresh output directory")
    with (args.output / "cross_history.log").open("x") as log:
        class Tee:
            def write(self, value):
                sys.__stdout__.write(value)
                log.write(value)
                return len(value)

            def flush(self):
                sys.__stdout__.flush()
                log.flush()

        with redirect_stdout(Tee()):
            run(args)


if __name__ == "__main__":
    main()
