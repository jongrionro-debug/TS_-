#!/usr/bin/env python3
"""Issue 13: user-disjoint, chronological, train-tuned length revalidation."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import shlex
import time

import numpy as np

from wildchat.metrics import QUANTILES
from wildchat.prepare import iso_time, timestamp_us
from wildchat.train import SEARCH_SPACE, feature_names, feature_vector

MODELS = ("fixed", "previous_ratio", "cumulative_mean", "A", "B")
BASELINES = MODELS[:3]
METRICS = ("mean_pinball_loss", "pinball_p10", "pinball_p50", "pinball_p90",
           "mae_p50", "coverage", "mean_interval_width", "crossing_rate")


@dataclass
class Rows:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sources: np.ndarray
    turns: np.ndarray
    stamps: np.ndarray
    users: np.ndarray
    previous: np.ndarray
    cumulative: np.ndarray

    def take(self, mask):
        return replace(self, **{key: value[mask] for key, value in vars(self).items()})


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_allowed(prepared, manifest, names):
    """Exact filename allowlist; no directory enumeration or arbitrary split loader."""
    parts, hashes = {}, {}
    for split in ("train", "validation"):
        path = prepared / {"train": "train.jsonl", "validation": "validation.jsonl"}[split]
        values = {key: [] for key in ("x", "y", "groups", "sources", "turns", "stamps", "previous", "cumulative")}
        past, identities = {}, set()
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for raw in handle:
                h.update(raw)
                row = json.loads(raw)
                meta = row["metadata"]
                source, turn = meta["source_row_id"], meta["turn_index"]
                stamp, target = timestamp_us(meta["recorded_at"]), row["target"]["output_tokens"]
                identity = (source, meta["user_message_index"])
                if (meta["split"] != split or stamp is None or identity in identities or
                    any(isinstance(meta[k], bool) or not isinstance(meta[k], int) or meta[k] < 0
                        for k in ("group_id", "source_row_id", "turn_index", "user_message_index")) or
                    turn != row["history_features"]["previous_user_turns"] or
                    isinstance(target, bool) or not isinstance(target, (int, float)) or
                    not np.isfinite(target) or target < 0):
                    raise ValueError("invalid prepared identity, time, turn or target")
                identities.add(identity)
                total, count, last = past.get(source, (0.0, 0, -1))
                if turn <= last:
                    raise ValueError("source turns must be strictly increasing")
                # Only earlier eligible pairs in this source conversation; current y is
                # added AFTER the feature snapshot. Invalid pairs are not averaged.
                values["cumulative"].append(total / count if count else np.nan)
                past[source] = (total + target, count + 1, turn)
                values["x"].append(feature_vector(row, names))
                values["y"].append(target)
                values["groups"].append(meta["group_id"])
                values["sources"].append(source)
                values["turns"].append(turn)
                values["stamps"].append(stamp)
                lag = row["history_features"]["lag_1_output_tokens"]
                if lag is not None and (not np.isfinite(lag) or lag < 0):
                    raise ValueError("invalid preceding response length")
                values["previous"].append(np.nan if lag is None else lag)
        numeric = {key: np.asarray(value, dtype=np.int64 if key in
                   ("groups", "sources", "turns", "stamps") else np.float64)
                   for key, value in values.items()}
        rows = Rows(**numeric, users=np.full(len(values["y"]), "", dtype=object))
        expected = manifest["partitions"][split]
        if (len(rows.y) != expected["samples"] or not len(rows.y) or
            len(np.unique(rows.groups)) != expected["groups"] or
            len(np.unique(rows.sources)) != expected["conversations"] or
            int(rows.stamps.min()) != timestamp_us(expected["recorded_at_min"]) or
            int(rows.stamps.max()) != timestamp_us(expected["recorded_at_max"])):
            raise ValueError("prepared data differs from manifest")
        parts[split], hashes[split] = rows, h.hexdigest()
    return parts, hashes


def attach_users(parts, source, expected_source):
    """Project metadata only; materialize only allowed train/validation source IDs.

    The top-level hashed_ip is the conversation-record user proxy used here.
    No content, message payload, or other split's prepared rows are requested.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    if source.name != expected_source["name"]:
        raise ValueError("source filename differs from preparation provenance")
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != expected_source["rows"]:
        raise ValueError("source row count differs from preparation provenance")
    allowed = set().union(*(set(p.sources.tolist()) for p in parts.values()))
    mapping, offset = {}, 0
    for batch in parquet.iter_batches(batch_size=1024, columns=["hashed_ip", "timestamp"]):
        indices = [i for i in range(len(batch)) if offset + i in allowed]
        selected = batch.take(pa.array(indices, type=pa.int64())).to_pylist()
        for index, meta in zip(indices, selected):
            user = meta["hashed_ip"]
            # Never emit identifiers into result files or logs.
            mapping[offset + index] = (user if isinstance(user, str) and user.strip() else "",
                                       timestamp_us(meta["timestamp"]))
        offset += len(batch)
    if set(mapping) != allowed:
        raise ValueError("source IDs unavailable in declared source")
    for rows in parts.values():
        if any(mapping[int(source_id)][1] != int(stamp)
               for source_id, stamp in zip(rows.sources, rows.stamps)):
            raise ValueError("source row timestamps differ from prepared identities")
        rows.users = np.asarray([mapping[int(s)][0] for s in rows.sources], dtype=object)


def describe(rows):
    return {"samples": len(rows.y), "users": len(set(rows.users) - {""}),
            "groups": len(np.unique(rows.groups)), "conversations": len(np.unique(rows.sources)),
            "first_turn_samples": int((rows.turns == 0).sum()),
            "recorded_at_min": iso_time(int(rows.stamps.min())) if len(rows.y) else None,
            "recorded_at_max": iso_time(int(rows.stamps.max())) if len(rows.y) else None}


def purge_and_check(earlier, later, cutoff):
    """Drop whole original groups for missing IDs, boundary spans or shared users.

    Both sides of every shared user are removed, retaining the fixed time windows.
    Removing whole original groups also preserves the existing duplicate isolation.
    """
    before = {"earlier": describe(earlier), "later": describe(later)}
    shared = (set(earlier.users) & set(later.users)) - {""}
    missing_groups = set(earlier.groups[earlier.users == ""]) | set(later.groups[later.users == ""])
    boundary_groups = set(earlier.groups[earlier.stamps >= cutoff]) | set(later.groups[later.stamps < cutoff])
    # A group seen on both sides crosses the chronological boundary.
    boundary_groups.update(set(earlier.groups) & set(later.groups))
    shared_user_groups = set()
    for rows in (earlier, later):
        shared_user_groups.update(rows.groups[np.isin(rows.users, list(shared))])
    bad_groups = missing_groups | boundary_groups | shared_user_groups
    kept = [p.take(~np.isin(p.groups, list(bad_groups))) for p in (earlier, later)]
    left, right = kept
    if not len(left.y) or not len(right.y):
        raise ValueError("purging left an empty partition")
    checks = {"user_intersection": len(set(left.users) & set(right.users)),
              "group_intersection": len(set(left.groups) & set(right.groups)),
              "source_intersection": len(set(left.sources) & set(right.sources)),
              "strict_chronology": bool(left.stamps.max() < cutoff <= right.stamps.min()),
              "missing_users": int((left.users == "").sum() + (right.users == "").sum())}
    if any(checks[k] for k in ("user_intersection", "group_intersection", "source_intersection", "missing_users")) or not checks["strict_chronology"]:
        raise AssertionError("disjoint chronological user split failed")
    audit = {"cutoff": iso_time(int(cutoff)), "before": before,
             "shared_users_before_purge": len(shared),
             "later_rows_with_shared_user_before_purge": int(np.isin(later.users, list(shared)).sum()),
             "removed_original_groups": len(bad_groups),
             "removed_group_reasons_nonexclusive": {"missing_user": len(missing_groups),
                 "crosses_time_boundary": len(boundary_groups), "shared_user": len(shared_user_groups)},
             "removed_rows": {"earlier": len(earlier.y) - len(left.y), "later": len(later.y) - len(right.y)},
             "after": {"earlier": describe(left), "later": describe(right)}, "checks": checks}
    print(json.dumps({"split_audit": audit}, ensure_ascii=False), flush=True)
    return left, right, audit


def inner_split(rows, fraction):
    # Select one boundary from TRAIN group start times; never inspect val scores.
    starts = {}
    for group, stamp in zip(rows.groups, rows.stamps):
        starts[int(group)] = min(starts.get(int(group), int(stamp)), int(stamp))
    ordered = sorted((stamp, group) for group, stamp in starts.items())
    cutoff = ordered[int(len(ordered) * fraction)][0]
    return purge_and_check(rows.take(rows.stamps < cutoff), rows.take(rows.stamps >= cutoff), cutoff)


def contributions(y, pred):
    """Score P10/P50/P90 as emitted, including crossings and signed width."""
    if pred.shape != (len(y), 3) or not np.isfinite(pred).all() or np.any(pred < 0):
        raise ValueError("finite nonnegative (n,3) predictions required")
    error = y[:, None] - pred
    loss = np.maximum(np.asarray(QUANTILES) * error, (np.asarray(QUANTILES) - 1) * error)
    return np.column_stack((loss.mean(axis=1), loss, np.abs(y - pred[:, 1]),
                            (pred[:, 0] <= y) & (y <= pred[:, 2]), pred[:, 2] - pred[:, 0],
                            np.any(pred[:, :-1] > pred[:, 1:], axis=1)))


def cohorts(rows):
    return {"first_turn": rows.turns == 0, "followup_turn": rows.turns > 0,
            "all": np.ones(len(rows.y), dtype=bool)}


def grouped(rows, predictions):
    users, codes = np.unique(rows.users, return_inverse=True)
    masks = cohorts(rows)
    sums = np.zeros((len(users), 3, len(MODELS), len(METRICS)))
    counts = np.zeros((len(users), 3), dtype=np.int64)
    for c, mask in enumerate(masks.values()):
        np.add.at(counts[:, c], codes[mask], 1)
        for m, model in enumerate(MODELS):
            np.add.at(sums[:, c, m], codes[mask], contributions(rows.y[mask], predictions[model][mask]))
    means = np.divide(sums, counts[:, :, None, None], out=np.zeros_like(sums),
                      where=counts[:, :, None, None] > 0)
    return masks, sums, counts, means


def evaluate_all(rows, predictions):
    masks, sums, counts, means = grouped(rows, predictions)
    result = {}
    for c, (cohort, mask) in enumerate(masks.items()):
        result[cohort] = {}
        for weighting in ("row", "user"):
            total = int(counts[:, c].sum()) if weighting == "row" else int((counts[:, c] > 0).sum())
            vals = (sums[:, c].sum(axis=0) if weighting == "row" else means[:, c].sum(axis=0)) / max(total, 1)
            result[cohort][weighting] = {
                name: {"n_rows": int(mask.sum()), "n_users": int((counts[:, c] > 0).sum()),
                       **dict(zip(METRICS, map(float, vals[m])))} if total else None
                for m, name in enumerate(MODELS)}
    return result


def quantile(values, q):
    return float(np.quantile(values, q, method="inverted_cdf"))


def fit_baselines(rows):
    fixed = [quantile(rows.y, q) for q in QUANTILES]
    valid = np.isfinite(rows.previous) & (rows.previous > 0)
    if not valid.any():
        raise ValueError("no positive preceding response in training")
    ratios, weights = rows.y[valid] / rows.previous[valid], rows.previous[valid]
    order = np.argsort(ratios, kind="stable")
    cumulative_weights = np.cumsum(weights[order])
    # Weighted quantiles minimize sum pinball(y, slope * previous) through origin.
    slopes = [float(ratios[order[np.searchsorted(cumulative_weights, q * cumulative_weights[-1])]])
              for q in QUANTILES]
    valid_mean = np.isfinite(rows.cumulative)
    if not valid_mean.any():
        raise ValueError("no prior eligible conversation response in training")
    residuals = rows.y[valid_mean] - rows.cumulative[valid_mean]
    # Baseline b's median remains exactly the causal conversation mean. The
    # train-only residual tails provide an interval without changing that point.
    offsets = [min(0.0, quantile(residuals, .1)), 0.0, max(0.0, quantile(residuals, .9))]
    return {"fixed": fixed, "previous_ratio": slopes, "cumulative_mean_offsets": offsets,
            "positive_previous_fit_rows": int(valid.sum()), "cumulative_fit_rows": int(valid_mean.sum())}


def predict_baselines(rows, fitted):
    fixed = np.tile(fitted["fixed"], (len(rows.y), 1))
    previous, cumulative = fixed.copy(), fixed.copy()
    valid = np.isfinite(rows.previous) & (rows.previous > 0)
    previous[valid] = rows.previous[valid, None] * fitted["previous_ratio"]
    valid = np.isfinite(rows.cumulative)
    cumulative[valid] = np.maximum(0, rows.cumulative[valid, None] + fitted["cumulative_mean_offsets"])
    return {"fixed": fixed, "previous_ratio": previous, "cumulative_mean": cumulative}


def common_parameters(seed, threads):
    return {"objective": "quantile", "metric": "quantile", "learning_rate": .05,
            "max_bin": 255, "feature_fraction": 1.0, "bagging_fraction": 1.0,
            "bagging_freq": 0, "lambda_l1": 0.0, "lambda_l2": 0.0,
            "seed": seed, "data_random_seed": seed, "feature_fraction_seed": seed,
            "bagging_seed": seed, "deterministic": True, "force_col_wise": True,
            "num_threads": threads, "verbosity": -1, "feature_pre_filter": False}


def select_model(fit, holdout, columns, names, args):
    import lightgbm as lgb
    train_set = lgb.Dataset(fit.x[:, columns], label=fit.y, feature_name=names, free_raw_data=False)
    valid_set = lgb.Dataset(holdout.x[:, columns], label=holdout.y, reference=train_set, free_raw_data=False)
    trials, best_predictions = [], None
    best = None
    for index, config in enumerate(SEARCH_SPACE):
        preds, fits = [], []
        for q in QUANTILES:
            curve = {}
            model = lgb.train({**common_parameters(args.seed, args.threads), **config, "alpha": q},
                              train_set, num_boost_round=args.max_rounds, valid_sets=[valid_set],
                              valid_names=["inner_holdout"], callbacks=[lgb.record_evaluation(curve),
                              lgb.early_stopping(args.early_stopping_rounds, first_metric_only=True, verbose=False)])
            preds.append(model.predict(holdout.x[:, columns], num_iteration=model.best_iteration,
                                       num_threads=args.threads))
            fits.append({"quantile": q, "best_iteration": model.best_iteration,
                         "rounds_evaluated": len(curve["inner_holdout"]["quantile"]),
                         "inner_raw_pinball": model.best_score["inner_holdout"]["quantile"]})
        raw = np.column_stack(preds)
        pred = np.maximum(raw, 0)  # Never sort or permute quantile identities.
        score = contributions(holdout.y, pred).mean(axis=0)
        trial = {"index": index, "parameters": config, "quantile_fits": fits,
                 "inner_row_metrics": dict(zip(METRICS, map(float, score))),
                 "raw_crossing_rows": int(np.any(raw[:, :-1] > raw[:, 1:], axis=1).sum()),
                 "negative_prediction_rows": int(np.any(raw < 0, axis=1).sum())}
        trials.append(trial)
        print(f'inner trial {index}: mean pinball={score[0]:.6f}', flush=True)
        if best is None or score[0] < best["inner_row_metrics"]["mean_pinball_loss"]:
            best, best_predictions = trial, pred
    return {"trials": trials, "selected_trial": best["index"],
            "selected_parameters": best["parameters"], "selected_quantile_fits": best["quantile_fits"]}, best_predictions


def refit_predict(train, report_rows, columns, names, selected, args):
    import lightgbm as lgb
    dataset = lgb.Dataset(train.x[:, columns], label=train.y, feature_name=names, free_raw_data=False)
    predictions = []
    for fit in selected["selected_quantile_fits"]:
        model = lgb.train({**common_parameters(args.seed, args.threads), **selected["selected_parameters"],
                           "alpha": fit["quantile"]}, dataset, num_boost_round=fit["best_iteration"])
        predictions.append(model.predict(report_rows.x[:, columns], num_threads=args.threads))
    raw = np.column_stack(predictions)
    return np.maximum(raw, 0), {"raw_crossing_rows": int(np.any(raw[:, :-1] > raw[:, 1:], axis=1).sum()),
                               "negative_prediction_rows": int(np.any(raw < 0, axis=1).sum())}


def bootstrap(rows, predictions, best_baselines, seed, replicates):
    masks, sums, counts, means = grouped(rows, predictions)
    g = len(sums)
    samples = np.full((replicates, 3, 2, len(MODELS), len(METRICS)), np.nan)
    rng = np.random.Generator(np.random.PCG64(seed))
    for r in range(replicates):
        weights = np.bincount(rng.integers(0, g, size=g), minlength=g)
        for w, values in enumerate((sums, means)):
            denominator = weights @ (counts if w == 0 else (counts > 0))
            numerator = np.einsum("g,gk->k", weights, values.reshape(g, -1)).reshape(values.shape[1:])
            np.divide(numerator, denominator[:, None, None], out=samples[r, :, w],
                      where=denominator[:, None, None] > 0)
    point = evaluate_all(rows, predictions)
    result = {}
    for c, cohort in enumerate(masks):
        result[cohort] = {}
        for w, weighting in enumerate(("row", "user")):
            draws = samples[:, c, w]
            valid = np.isfinite(draws).all(axis=(1, 2))
            comparisons = {}
            for label, comparator in {"B-A": "A", "B-best_baseline": best_baselines[cohort][weighting],
                                      **{f"B-{b}": b for b in BASELINES}}.items():
                delta = draws[valid, MODELS.index("B")] - draws[valid, MODELS.index(comparator)]
                bounds = np.quantile(delta, [.025, .975], axis=0, method="linear") if len(delta) else None
                comparisons[label] = {"comparator": comparator,
                    "point": {metric: point[cohort][weighting]["B"][metric] - point[cohort][weighting][comparator][metric]
                              for metric in METRICS} if valid.any() else None,
                    "ci95": {metric: bounds[:, i].tolist() for i, metric in enumerate(METRICS)} if len(delta) else None}
            result[cohort][weighting] = {"effective_replicates": int(valid.sum()),
                                        "empty_replicates": int((~valid).sum()), "comparisons": comparisons}
    return {"method": "paired user percentile bootstrap; all G users drawn G times with replacement",
            "groups_per_draw": g, "replicates": replicates, "seed": seed, "rng": "numpy.random.PCG64",
            "confidence_level": .95,
            "percentiles": [.025, .975], "quantile_method": "linear", "refit_or_reselect": False,
            "shared_draws": "all five models, cohorts and weighting schemes",
            "user_weighting": "mean within user and cohort, then equal mean of participating users",
            "empty_rule": "omit draws with no cohort rows; report counts; unavailable CI is null"}, result


def render_report(r):
    lines = ["# #13 사용자 분리·train 내부 튜닝 재검증", "",
             "기존 #7 결과를 보존한 별도 실행이다. 검증 기간은 보고에만 사용했다.", "",
             "## 분할과 선택", "",
             "- #5의 train/validation 시간 경계를 유지한다. 경계를 넘는 hashed_ip 사용자는 양쪽에서 제외하며, 해당 원래 대화 그룹 전체도 제외한다. ID 결측 그룹도 제외한다.",
             "- 사용자는 원본 대화 행의 최상위 hashed_ip 프록시다. 동일 IP의 여러 실제 사용자와 IP가 바뀐 동일인을 구분할 수 없다.",
             f'- 남은 train 그룹 시작시각의 {r["settings"]["inner_fraction"]:.0%} 위치로 내부 시간 홀드아웃을 만들고 같은 사용자·그룹 제거를 적용한다. 같은 경계시각은 뒤쪽이다.',
             "- A/B 각각 동일한 3개 설정×3분위수, 내부 홀드아웃 행 가중 평균 pinball로 설정을 고른다. 분위수별 조기 종료 횟수도 내부에서만 선택한다. 이후 전체 정제 train으로 고정 횟수 재학습하며 validation은 학습·선택에 전달하지 않는다.",
             "- 비교 기준선은 각 턴·가중 방식의 내부 홀드아웃 평균 pinball 최선으로 미리 고정한다. val에서 재선택하지 않으며 모든 개별 기준선 대비 CI도 JSON에 제공한다.", "",
             "| 구간 | 표본 | 사용자 | 첫 턴 | 기록시각 최소 | 기록시각 최대 |", "|---|---:|---:|---:|---|---|"]
    for key in ("outer", "inner"):
        audit = r["splits"][key]
        for side in ("earlier", "later"):
            p = audit["after"][side]
            lines.append(f'| {key}/{side} | {p["samples"]:,} | {p["users"]:,} | {p["first_turn_samples"]:,} | {p["recorded_at_min"]} | {p["recorded_at_max"]} |')
    for key in ("outer", "inner"):
        audit = r["splits"][key]
        lines += ["", f'- {key}: 제거 행 {audit["removed_rows"]}, 제거 전 공유 사용자 {audit["shared_users_before_purge"]:,}명, 검증 {json.dumps(audit["checks"], ensure_ascii=False)}.', ""]
    lines += ["## 기준선·지표 정의", "",
              "- fixed: train 정답의 경험적 P10/P50/P90(역경험 CDF). a/previous_ratio: 직전 응답×분위수별 비율. 양수 직전 길이로 가중한 정답/직전 길이의 분위수로 원점을 지나는 pinball 최적 비율을 train에서 추정한다. 직전 값 0·결측이면 fixed를 쓴다.",
              "- b/cumulative_mean: 같은 원본 대화에서 현재 턴 이전의 유효 user→assistant 쌍 응답 길이 누적 평균을 P50으로 쓴다. P10/P90은 train의 누적 평균 잔차 하·상 분위수를 더하고 0으로 자른다(잔차 하한≤0≤상한). 과거 유효 쌍이 없으면 fixed다. 3턴 lag 평균이 아니며 현재 응답은 포함하지 않는다. 유효하지 않은 쌍은 평균에서 제외한다.",
              "- 행 가중은 모든 행의 평균, 사용자 가중은 해당 턴 구분 안에서 사용자별 평균 후 사용자 동일 가중이다. 첫 턴은 turn_index==0이다.",
              "- 분위수 정렬·교차 보정을 하지 않는다. 음수 길이만 0으로 자른 후 원래 분위수 순서의 pinball·coverage·부호 있는 P90−P10 폭과 교차율을 보고한다. coverage는 양 끝 포함이며 P10>P90인 구간은 적중하지 않는다. 원시 교차·음수 행 수도 JSON에 보존한다.", "",
              "## 전체 결과 표", "", "| 턴 | 가중 | 모델 | 평균 pinball | P10 loss | P50 loss | P90 loss | P50 MAE | coverage | 구간 폭 | 교차율 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cohort, weighted in r["validation_metrics"].items():
        for weight, models in weighted.items():
            for model, metrics in models.items():
                vals = " | ".join(f'{metrics[key]:.6f}' for key in METRICS) if metrics else " | ".join(["NA"] * len(METRICS))
                lines.append(f"| {cohort} | {weight} | {model} | {vals} |")
    lines += ["", "## 사용자 bootstrap: 평균 pinball 차이와 95% CI", "",
              f'음수는 B의 손실 감소다. 같은 사용자의 모든 대화·행을 유지한 짝지은 사용자 재표집 {r["bootstrap"]["replicates"]:,}회(seed {r["bootstrap"]["seed"]}), percentile 선형 보간이다. 아래 기준선은 train 내부에서 미리 선택했다.', "",
              "| 턴 | 가중 | 차이 | 비교 대상 | 점추정 | 95% CI | 유효/전체 반복 |", "|---|---|---|---|---:|---|---|"]
    for cohort, weighted in r["bootstrap_results"].items():
        for weight, entry in weighted.items():
            for label in ("B-A", "B-best_baseline"):
                d = entry["comparisons"][label]
                if d["point"] is None:
                    lines.append(f'| {cohort} | {weight} | {label} | {d["comparator"]} | NA | NA | 0 |')
                    continue
                lo, hi = d["ci95"]["mean_pinball_loss"]
                lines.append(f'| {cohort} | {weight} | {label} | {d["comparator"]} | {d["point"]["mean_pinball_loss"]:.6f} | [{lo:.6f}, {hi:.6f}] | {entry["effective_replicates"]}/{r["bootstrap"]["replicates"]} |')
    lines += ["", "## 재현과 한계", "", "```sh", r["command"],
              "/workspace/venvs/wildchat/bin/python -m unittest wildchat.tests.test_revalidate wildchat.tests.test_metrics",
              "python3 scripts/verify.py", "```", "",
              "전체 설정·후보별 내부 점수·학습 횟수·분할 검증·파일 해시·버전·분위수별 지표·모든 차이의 CI는 [revalidation.json](revalidation.json), 분할 로그는 [revalidation.log](revalidation.log)에 있다.", "",
              "- 첫 샤드의 기존 고정 표본만 썼다. 사용자 제거 후 평가 모집단도 달라져 기존 −9.55와의 차이를 특정 보정 요인의 인과 효과로 분해할 수 없다.",
              "- 기록시각은 대화 행 마지막 시각이므로 실제 개별 요청 도착 순서를 보장하지 않는다. 대화 내 이전 응답은 이미 완료되었다고 가정한다.",
              "- 원본 메타데이터 연결은 준비 manifest의 단일 소스 파일명·행 수와 선택한 모든 행의 시각을 검사한다. 원본 전체 바이트 해시는 다시 읽지 않았으며 manifest의 SHA-256은 출처 주장으로만 기록한다.",
              "- CI는 선택·재학습을 마친 예측과 이 평가 표본에 조건부다. 학습·선택 불확실성, 시간 블록 상관, IP 식별 오차를 보정하지 않는다. 가시 텍스트 재구성 토큰이며 실제 API 사용량이나 GPU 효율에 대한 결론이 아니다.",
              "- 기존 #7에서 이미 본 validation의 재검증이므로 완전히 새로운 미관측 기간에 대한 일반화 검증은 아니다.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    if (not 0 < args.inner_fraction < 1 or args.seed < 0 or args.threads < 1 or
        min(args.max_rounds, args.early_stopping_rounds, args.bootstrap_replicates) < 1):
        raise ValueError("invalid experiment settings")
    manifest_path = args.prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["input_files"]) != 1 or manifest["lags"] < 1:
        raise ValueError("one declared source and at least one lag required")
    names = feature_names(manifest["lags"])
    parts, hashes = load_allowed(args.prepared, manifest, names["B"])
    attach_users(parts, args.source, manifest["input_files"][0])
    cutoff = timestamp_us(manifest["boundaries"]["train_validation_cutoff"])
    train, validation, outer_audit = purge_and_check(parts["train"], parts["validation"], cutoff)
    fit, holdout, inner_audit = inner_split(train, args.inner_fraction)
    selected, inner_predictions = {}, predict_baselines(holdout, fit_baselines(fit))
    for name in ("A", "B"):
        columns = [names["B"].index(n) for n in names[name]]
        print(f"Selecting {name} using train-internal holdout only", flush=True)
        selected[name], inner_predictions[name] = select_model(fit, holdout, columns, names[name], args)
    inner_scores = evaluate_all(holdout, inner_predictions)
    best = {cohort: {weight: min(BASELINES, key=lambda b: scores[b]["mean_pinball_loss"])
                     if scores["fixed"] else "fixed" for weight, scores in weighted.items()}
            for cohort, weighted in inner_scores.items()}
    print("All selection frozen; refitting on full purged train", flush=True)
    baseline_fit = fit_baselines(train)
    predictions, raw_diagnostics = predict_baselines(validation, baseline_fit), {}
    for name in ("A", "B"):
        columns = [names["B"].index(n) for n in names[name]]
        predictions[name], raw_diagnostics[name] = refit_predict(train, validation, columns, names[name], selected[name], args)
    metrics = evaluate_all(validation, predictions)
    bootstrap_info, bootstrap_results = bootstrap(validation, predictions, best, args.seed, args.bootstrap_replicates)
    command_args = [sys.executable, "wildchat/revalidate.py"]
    for key, value in vars(args).items():
        command_args.extend(["--" + key.replace("_", "-"), str(value)])
    result = {"issue": 13, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": shlex.join(command_args),
              "versions": {"python": sys.version.split()[0], **{n: importlib.metadata.version(n)
                           for n in ("numpy", "lightgbm", "scipy", "pyarrow")}},
              "source": {"prepared": str(args.prepared), "metadata_source": str(args.source),
                         "preparation_provenance": manifest["input_files"],
                         "metadata_checks": ["filename", "row count", "all selected row timestamps"],
                         "prepared_sha256": hashes, "manifest_sha256": fingerprint(manifest_path)},
              "code_sha256": {str(p.relative_to(Path.cwd())): fingerprint(p) for p in
                              (Path(__file__).resolve(), Path(__file__).with_name("train.py").resolve(),
                               Path(__file__).with_name("prepare.py").resolve(),
                               Path(__file__).with_name("metrics.py").resolve())},
              "settings": vars(args) | {"prepared": str(args.prepared), "source": str(args.source), "output": str(args.output)},
              "features": names, "splits": {"outer": outer_audit, "inner": inner_audit},
              "training": {"common_parameters": common_parameters(args.seed, args.threads), "search_space": SEARCH_SPACE,
                           "selection_and_early_stopping": "train-internal holdout, row mean pinball, first candidate on ties",
                           "refit": "full purged train, selected iteration per quantile; no val callbacks",
                           "quantile_rule": "nonnegative clipping only; NEVER sort quantiles", "models": selected},
              "baseline_fit": baseline_fit, "inner_metrics": inner_scores, "best_baselines_selected_inner": best,
              "raw_validation_diagnostics": raw_diagnostics, "validation_metrics": metrics,
              "bootstrap": bootstrap_info, "bootstrap_results": bootstrap_results,
              "elapsed_seconds": time.monotonic() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "revalidation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "revalidation.md").write_text(render_report(result))
    print(f"Wrote revalidation results in {args.output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="exact original training-shard Parquet for user metadata")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-fraction", type=float, default=.8)
    parser.add_argument("--max-rounds", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, value):
                for stream in self.streams:
                    stream.write(value)
                return len(value)

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        with (args.output / "revalidation.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)):
                run(args)
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(2, f"revalidate: {error}\n")


if __name__ == "__main__":
    main()
