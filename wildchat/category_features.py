#!/usr/bin/env python3
"""Issue 19 follow-up: category information as features of the global B model.

C_feat_label: B's 31 features + the MiniLM category label (LightGBM categorical).
C_feat_sim:   B's 31 features + the 7 MiniLM category similarity scores.
Both reuse global B's selected hyperparameters and per-quantile iteration
counts from category_c.json (no re-tuning) and refit on the full purged train.
A, B and C (separate per-category models) are recomputed with category.py's
rules so all models share one paired user bootstrap. Validation only; the
test split file is never read.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import time

import numpy as np

from wildchat import category as cg
from wildchat import revalidate as rv
from wildchat.classify import CATEGORIES
from wildchat.prepare import timestamp_us
from wildchat.train import feature_names

MODELS = ("A", "B", "C", "C_feat_label", "C_feat_sim")
COMPARISONS = (("C", "B"), ("C_feat_label", "B"), ("C_feat_sim", "B"))
SIM_NAMES = [f"sim_{c}" for c in CATEGORIES]


def read_sims(prepared, path):
    sims = {}
    with path.open() as h:
        for line in h:
            r = json.loads(line)
            sims[(r["source_row_id"], r["user_message_index"])] = r["sims"]
    out = {}
    for split, name in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        with (prepared / name).open("rb") as h:
            out[split] = np.asarray([sims[(m["source_row_id"], m["user_message_index"])]
                                     for m in (json.loads(raw)["metadata"] for raw in h)], dtype=float)
    return out


def label_codes(categories):
    index = {c: i for i, c in enumerate(CATEGORIES)}
    return np.asarray([index[c] for c in categories], dtype=float)


def fit_fixed(x_train, y, x_val, names, selected, args, categorical=()):
    """Refit each quantile with fixed parameters and iteration counts."""
    import lightgbm as lgb
    data = lgb.Dataset(x_train, label=y, feature_name=names, categorical_feature=list(categorical) or "auto",
                       free_raw_data=False)
    preds = []
    for q, it in zip(rv.QUANTILES, selected["iterations"]):
        model = lgb.train({**rv.common_parameters(args.seed, args.threads), **selected["parameters"], "alpha": q},
                          data, num_boost_round=it)
        preds.append(model.predict(x_val, num_threads=args.threads))
    return np.maximum(np.column_stack(preds), 0)


def run(args):
    started = time.monotonic()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    names = feature_names(manifest["lags"])
    prior = json.loads(args.category_c.read_text())["global_selection"]
    parts, hashes = rv.load_allowed(args.prepared, manifest, names["B"])
    rv.attach_users(parts, args.source, manifest["input_files"][0])
    cats, sims = cg.read_categories(args.prepared, args.labels), read_sims(args.prepared, args.sims)
    for k in parts:
        if len(sims[k]) != len(parts[k].y):
            raise ValueError("similarities misaligned with prepared rows")
        # Append label code and 7 sims as extra columns after B's 31 features.
        parts[k].x = np.column_stack([parts[k].x, label_codes(cats[k]), sims[k]])
        parts[k] = cg.with_categories(parts[k], cats[k])
    cutoff = timestamp_us(manifest["boundaries"]["train_validation_cutoff"])
    train, validation, _ = rv.purge_and_check(parts["train"], parts["validation"], cutoff)
    nb = len(names["B"])
    cols = {"A": [names["B"].index(n) for n in names["A"]], "B": list(range(nb)),
            "C_feat_label": list(range(nb + 1)), "C_feat_sim": list(range(nb)) + list(range(nb + 1, nb + 8))}
    fnames = {"A": names["A"], "B": names["B"], "C_feat_label": names["B"] + ["category"],
              "C_feat_sim": names["B"] + SIM_NAMES}
    preds = {}
    for m in ("A", "B", "C_feat_label", "C_feat_sim"):
        sel = prior["A" if m == "A" else "B"]
        print(f"Fitting {m} with fixed {sel}", flush=True)
        preds[m] = fit_fixed(train.x[:, cols[m]], train.y, validation.x[:, cols[m]], fnames[m], sel, args,
                             categorical=[nb] if m == "C_feat_label" else ())
    # C: same per-category selection as category.py (all 7 categories are trainable there).
    fit, holdout, _ = rv.inner_split(train, args.inner_fraction)
    preds["C"] = preds["B"].copy()
    for cat in CATEGORIES:
        tr, f, h, va = (r.categories == cat for r in (train, fit, holdout, validation))
        if f.sum() < args.min_fit_rows or h.sum() < args.min_holdout_rows:
            continue
        print(f"Selecting C[{cat}]", flush=True)
        sel, _ = rv.select_model(fit.take(f), holdout.take(h), cols["B"], names["B"], args)
        preds["C"][va] = rv.refit_predict(train.take(tr), validation.take(va), cols["B"], names["B"], sel, args)[0]
    metrics = cg.score_table(validation, preds, MODELS)
    boot = cg.bootstrap_diffs(validation, preds, args.seed, args.bootstrap_replicates, MODELS, COMPARISONS)
    result = {"issue": 19, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_split": "validation only; test split never read",
              "prepared_sha256": hashes, "labels_sha256": rv.fingerprint(args.labels),
              "sims_sha256": rv.fingerprint(args.sims), "fixed_hyperparameters": prior,
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "validation_metrics": metrics, "bootstrap_ci95": boot, "elapsed_seconds": time.monotonic() - started}
    (args.output / "category_features.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "category_features.md").write_text(render(result))
    print("Wrote category_features results", flush=True)
    return result


def render(r):
    lines = ["# #19 후속: 범주 정보를 전역 B의 특징으로 추가 (validation 전용)", "",
             "test 분할 파일은 읽지 않았다. 재분류 없이 기존 MiniLM 라벨을 재사용했다(유사도는 같은 모델로 재임베딩, argmax가 저장 라벨과 일치하는지 검사).", "",
             "- C_feat_label: B 31개 특징 + 범주 라벨(LightGBM categorical).",
             "- C_feat_sim: B 31개 특징 + 7개 범주 유사도(수치).",
             "- 두 변형 모두 전역 B의 하이퍼파라미터와 분위수별 반복 수를 그대로 고정해 전체 purged train으로 재학습.",
             "- C: category_c.md와 같은 범주별 별도 모델.", "",
             "## validation 평균 pinball", "", "| 턴 | 가중 | " + " | ".join(MODELS) + " |",
             "|---|---|" + "---:|" * len(MODELS)]
    for c, w_ in r["validation_metrics"].items():
        for w, ms in w_.items():
            lines.append(f"| {c} | {w} | " + " | ".join(f'{ms[m]["mean_pinball_loss"]:.4f}' for m in MODELS) + " |")
    lines += ["", "## 사용자 bootstrap 95% CI (음수 = 앞 모델이 좋음)", "", "| 턴 | 가중 | 차이 | 점추정 | 95% CI |",
              "|---|---|---|---:|---|"]
    for c, w_ in r["bootstrap_ci95"].items():
        for w, ds in w_.items():
            for label, d in ds.items():
                a, b = label.split("-", 1)
                pt = r["validation_metrics"][c][w][a]["mean_pinball_loss"] - r["validation_metrics"][c][w][b]["mean_pinball_loss"]
                lo, hi = d["ci95"]
                lines.append(f"| {c} | {w} | {label} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ["", "- 유사도 파일 sims.jsonl은 저장소 밖(데이터 디렉터리)에 있다. 범주 분류 정확도는 train 200행 수기 라벨 대비 0.32로 낮다. validation은 이미 여러 번 본 기간이다.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("prepared", "source", "labels", "sims", "category-c", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner-fraction", type=float, default=.8)
    p.add_argument("--max-rounds", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=40)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    p.add_argument("--min-fit-rows", type=int, default=500)
    p.add_argument("--min-holdout-rows", type=int, default=100)
    args = p.parse_args()
    with (args.output / "category_features.log").open("w") as log:
        class Tee:
            def write(self, v):
                sys.__stdout__.write(v); log.write(v); return len(v)
            def flush(self):
                sys.__stdout__.flush(); log.flush()
        with redirect_stdout(Tee()):
            run(args)


if __name__ == "__main__":
    main()
