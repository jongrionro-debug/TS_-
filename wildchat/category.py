#!/usr/bin/env python3
"""Issue 19: condition C = per-prompt-category B models, validation only.

Reuses the #13 user-disjoint split, train-internal holdout selection and
unsorted quantile scoring. the test split file is never read (no option exists).
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import shlex
import time

import numpy as np

from wildchat import revalidate as rv
from wildchat.prepare import timestamp_us
from wildchat.train import feature_names

MODELS = ("A", "B", "C", "C_fallback", "C_inner_select")
COMPARISONS = (("C", "B"), ("C_fallback", "B"), ("C_inner_select", "B"), ("C", "A"))


@dataclass
class CRows(rv.Rows):
    categories: np.ndarray = None


def read_categories(prepared, labels_path):
    """Category per prepared row, in load_allowed's file order."""
    labels = {}
    with labels_path.open() as handle:
        for line in handle:
            r = json.loads(line)
            labels[(r["source_row_id"], r["user_message_index"])] = r["category"]
    out = {}
    for split, name in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        cats = []
        with (prepared / name).open("rb") as handle:
            for raw in handle:
                meta = json.loads(raw)["metadata"]
                cats.append(labels[(meta["source_row_id"], meta["user_message_index"])])
        out[split] = np.asarray(cats, dtype=object)
    return out


def with_categories(rows, cats):
    if len(cats) != len(rows.y):
        raise ValueError("category labels misaligned with prepared rows")
    return CRows(**vars(rows), categories=cats)


def score_table(rows, predictions, models):
    """Mean pinball by cohort x weighting x model (+ n_rows, n_users)."""
    result = {}
    users, codes = np.unique(rows.users, return_inverse=True)
    for cohort, mask in rv.cohorts(rows).items():
        result[cohort] = {"row": {}, "user": {}}
        for model in models:
            if not mask.any():
                result[cohort]["row"][model] = result[cohort]["user"][model] = None
                continue
            loss = rv.contributions(rows.y[mask], predictions[model][mask])[:, 0]
            sums = np.bincount(codes[mask], loss, minlength=len(users))
            cnt = np.bincount(codes[mask], minlength=len(users))
            base = {"n_rows": int(mask.sum()), "n_users": int((cnt > 0).sum())}
            result[cohort]["row"][model] = {**base, "mean_pinball_loss": float(loss.mean())}
            result[cohort]["user"][model] = {**base, "mean_pinball_loss":
                                             float((sums[cnt > 0] / cnt[cnt > 0]).mean())}
    return result


def bootstrap_diffs(rows, predictions, seed, replicates):
    """Paired user bootstrap of model differences in mean pinball."""
    users, codes = np.unique(rows.users, return_inverse=True)
    g = len(users)
    masks = rv.cohorts(rows)
    per = {}
    for cohort, mask in masks.items():
        cnt = np.bincount(codes[mask], minlength=g).astype(float)
        sums = {m: np.bincount(codes[mask], rv.contributions(rows.y[mask], predictions[m][mask])[:, 0],
                               minlength=g) for m in MODELS}
        per[cohort] = (cnt, sums)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = {(c, w, a, b): [] for c in masks for w in ("row", "user") for a, b in COMPARISONS}
    for _ in range(replicates):
        weight = np.bincount(rng.integers(0, g, size=g), minlength=g).astype(float)
        for cohort, (cnt, sums) in per.items():
            present = cnt > 0
            if not (weight @ cnt) > 0:
                continue
            for a, b in COMPARISONS:
                row = (weight @ sums[a] - weight @ sums[b]) / (weight @ cnt)
                ua = np.divide(sums[a], cnt, out=np.zeros(g), where=present)
                ub = np.divide(sums[b], cnt, out=np.zeros(g), where=present)
                user = weight @ (ua - ub) / (weight @ present)
                draws[(cohort, "row", a, b)].append(row)
                draws[(cohort, "user", a, b)].append(user)
    out = {}
    for (c, w, a, b), values in draws.items():
        out.setdefault(c, {}).setdefault(w, {})[f"{a}-{b}"] = {
            "effective_replicates": len(values),
            "ci95": np.quantile(values, [.025, .975], method="linear").tolist() if values else None}
    return out


def run(args):
    started = time.monotonic()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    names = feature_names(manifest["lags"])
    parts, hashes = rv.load_allowed(args.prepared, manifest, names["B"])
    rv.attach_users(parts, args.source, manifest["input_files"][0])
    cats = read_categories(args.prepared, args.labels)
    parts = {k: with_categories(v, cats[k]) for k, v in parts.items()}
    cutoff = timestamp_us(manifest["boundaries"]["train_validation_cutoff"])
    train, validation, outer = rv.purge_and_check(parts["train"], parts["validation"], cutoff)
    fit, holdout, inner = rv.inner_split(train, args.inner_fraction)
    cols = {m: [names["B"].index(n) for n in names[m]] for m in ("A", "B")}
    selected, inner_pred = {}, {}
    for m in ("A", "B"):
        print(f"Selecting global {m} on train-internal holdout", flush=True)
        selected[m], inner_pred[m] = rv.select_model(fit, holdout, cols[m], names[m], args)
    predictions = {m: rv.refit_predict(train, validation, cols[m], names[m], selected[m], args)[0]
                   for m in ("A", "B")}
    categories = sorted(set(train.categories) | set(validation.categories))
    predictions["C"] = predictions["B"].copy()
    predictions["C_fallback"] = predictions["B"].copy()
    predictions["C_inner_select"] = predictions["B"].copy()
    per_category = {}
    for cat in categories:
        tr, f, h = (train.categories == cat), (fit.categories == cat), (holdout.categories == cat)
        va = validation.categories == cat
        info = {"train_rows": int(tr.sum()), "fit_rows": int(f.sum()), "inner_holdout_rows": int(h.sum()),
                "validation_rows": int(va.sum()),
                "validation_users": len(set(validation.users[va])),
                "validation_first_turn_rows": int((va & (validation.turns == 0)).sum())}
        trainable = f.sum() >= args.min_fit_rows and h.sum() >= args.min_holdout_rows
        info["own_model"] = bool(trainable)
        info["fallback_to_global_B"] = bool(tr.sum() < args.fallback_train_rows) or not trainable
        if trainable:
            print(f"Selecting C[{cat}] fit={f.sum()} holdout={h.sum()}", flush=True)
            sel, hpred = rv.select_model(fit.take(f), holdout.take(h), cols["B"], names["B"], args)
            pred, _ = rv.refit_predict(train.take(tr), validation.take(va), cols["B"], names["B"], sel, args)
            predictions["C"][va] = pred
            if not info["fallback_to_global_B"]:
                predictions["C_fallback"][va] = pred
            own = rv.contributions(holdout.y[h], hpred)[:, 0].mean()
            glob = rv.contributions(holdout.y[h], inner_pred["B"][h])[:, 0].mean()
            info["inner_holdout_pinball"] = {"category_model": float(own), "global_B": float(glob)}
            info["inner_select_uses_category_model"] = bool(own < glob)
            if own < glob:
                predictions["C_inner_select"][va] = pred
            info["selected"] = {"parameters": sel["selected_parameters"],
                                "iterations": [q["best_iteration"] for q in sel["selected_quantile_fits"]]}
        else:
            info["inner_select_uses_category_model"] = False
        if va.any():
            info["validation"] = score_table(validation.take(va), {m: p[va] for m, p in predictions.items()}, MODELS)
        per_category[cat] = info
    metrics = score_table(validation, predictions, MODELS)
    boot = bootstrap_diffs(validation, predictions, args.seed, args.bootstrap_replicates)
    command = shlex.join([sys.executable, "wildchat/category.py"] +
                         [x for k, v in vars(args).items() for x in ("--" + k.replace("_", "-"), str(v))])
    result = {"issue": 19, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "evaluation_split": "validation only; test split never read",
              "versions": {"python": sys.version.split()[0],
                           **{n: importlib.metadata.version(n) for n in ("numpy", "lightgbm", "pyarrow")}},
              "prepared_sha256": hashes, "labels_sha256": rv.fingerprint(args.labels),
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "split_sizes": {"train": len(train.y), "validation": len(validation.y),
                              "fit": len(fit.y), "inner_holdout": len(holdout.y)},
              "global_selection": {m: {"parameters": s["selected_parameters"],
                                       "iterations": [q["best_iteration"] for q in s["selected_quantile_fits"]]}
                                   for m, s in selected.items()},
              "validation_metrics": metrics, "bootstrap_ci95": boot, "per_category": per_category,
              "elapsed_seconds": time.monotonic() - started}
    (args.output / "category_c.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "category_c.md").write_text(render(result))
    print("Wrote category_c results", flush=True)
    return result


def render(r):
    s = r["settings"]
    lines = ["# #19 조건 C: 프롬프트 의미 범주별 분위수 모델 (validation 전용)", "",
             "test 분할 파일은 읽지 않았다. #13의 사용자 분리·train 내부 홀드아웃 선택 규칙을 그대로 사용했다.", "",
             f'- C: 범주별로 B와 같은 31개 특징의 LightGBM P10/P50/P90을 범주 내부 홀드아웃에서 선택·전체 범주 train으로 재학습한다. 내부 fit<{s["min_fit_rows"]}행 또는 홀드아웃<{s["min_holdout_rows"]}행이면 전역 B.',
             f'- C_fallback: train 행 수 <{s["fallback_train_rows"]}인 범주는 전역 B(임계값은 결과를 보기 전 고정).',
             "- C_inner_select: train 내부 홀드아웃에서 범주 모델이 전역 B보다 나을 때만 범주 모델 사용.", "",
             "## 전체 validation 평균 pinball", "", "| 턴 | 가중 | 행 | 사용자 | " + " | ".join(MODELS) + " |",
             "|---|---|---:|---:|" + "---:|" * len(MODELS)]
    for c, w_ in r["validation_metrics"].items():
        for w, ms in w_.items():
            a = ms["A"]
            lines.append(f'| {c} | {w} | {a["n_rows"]:,} | {a["n_users"]:,} | ' +
                         " | ".join(f'{ms[m]["mean_pinball_loss"]:.4f}' for m in MODELS) + " |")
    lines += ["", "## 차이의 사용자 bootstrap 95% CI (음수 = 앞 모델이 좋음)", "",
              "| 턴 | 가중 | 차이 | 점추정 | 95% CI |", "|---|---|---|---:|---|"]
    for c, w_ in r["bootstrap_ci95"].items():
        for w, ds in w_.items():
            for label, d in ds.items():
                a, b = label.split("-", 1)
                pt = r["validation_metrics"][c][w][a]["mean_pinball_loss"] - r["validation_metrics"][c][w][b]["mean_pinball_loss"]
                lo, hi = d["ci95"]
                lines.append(f"| {c} | {w} | {label} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ["", "## 범주별 (행 가중, all 턴)", "",
              "| 범주 | train 행 | val 행 | val 사용자 | val 첫 턴 | 자체 모델 | fallback | 내부선택 | A | B | C |",
              "|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|"]
    for cat, i in r["per_category"].items():
        v = i.get("validation", {}).get("all", {}).get("row", {})
        vals = " | ".join(f'{v[m]["mean_pinball_loss"]:.4f}' if v else "NA" for m in ("A", "B", "C"))
        lines.append(f'| {cat} | {i["train_rows"]:,} | {i["validation_rows"]:,} | {i["validation_users"]:,} | '
                     f'{i["validation_first_turn_rows"]:,} | {i["own_model"]} | {i["fallback_to_global_B"]} | '
                     f'{i["inner_select_uses_category_model"]} | {vals} |')
    lines += ["", "## 재현과 한계", "", "```sh", r["command"], "```", "",
              "- 범주는 로컬 다국어 임베딩 zero-shot 분류(wildchat/classify.py)이며 사람 라벨로 검증하지 않았다. 오분류가 섞인다.",
              "- validation은 #7·#13에서 이미 여러 번 본 기간이므로 C의 이점은 낙관적일 수 있다. test 재평가는 하지 않았다.",
              "- 첫 샤드 고정 표본(train+validation 전체 행 분류, 표본 추출 없음) 결과다.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner-fraction", type=float, default=.8)
    p.add_argument("--max-rounds", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=40)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    p.add_argument("--min-fit-rows", type=int, default=500)
    p.add_argument("--min-holdout-rows", type=int, default=100)
    p.add_argument("--fallback-train-rows", type=int, default=5000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "category_c.log").open("w") as log:
        class Tee:
            def write(self, v):
                sys.__stdout__.write(v); log.write(v); return len(v)
            def flush(self):
                sys.__stdout__.flush(); log.flush()
        with redirect_stdout(Tee()):
            run(args)


if __name__ == "__main__":
    main()
