#!/usr/bin/env python3
"""Issue 19 follow-up: MiniLM prompt-embedding PCA components as features of global B.

C_emb_PCA:     B's 31 features + top-k PCA components of the 384-dim embedding.
C_feat_sim_PCA: B + 7 category similarities + top-k PCA components.
PCA is fit on train rows only (inner fit rows for selection, full purged train
for the validation refit). k in {8,16,32} is chosen on the #13 train-internal
holdout, never on validation. Global B's selected hyperparameters and
per-quantile iteration counts are fixed throughout. The test split file is
never read.
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
from wildchat import category_features as cf
from wildchat import revalidate as rv
from wildchat.prepare import timestamp_us
from wildchat.train import feature_names

KS = (8, 16, 32)


def read_embeddings(prepared, path):
    data = np.load(path)
    index = {(int(s), int(m)): i for i, (s, m) in enumerate(data["keys"])}
    vectors = data["vectors"].astype(np.float32)
    out = {}
    for split, name in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        with (prepared / name).open("rb") as h:
            rows = [index[(m["source_row_id"], m["user_message_index"])]
                    for m in (json.loads(raw)["metadata"] for raw in h)]
        out[split] = vectors[rows]
    return out


class PCA:
    """Plain SVD PCA; fit() sees only the rows it is given."""

    def fit(self, x, k):
        self.mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[:k]
        self.explained = (s[:k] ** 2 / (s ** 2).sum()).tolist()
        return self

    def transform(self, x):
        return (x - self.mean) @ self.components.T


def design(rows, emb, base_cols, pca, k):
    return np.column_stack([rows.x[:, base_cols], pca.transform(emb)[:, :k]])


def pinball(y, pred):
    return float(rv.contributions(y, pred)[:, 0].mean())


def run(args):
    started = time.monotonic()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    names = feature_names(manifest["lags"])
    prior = json.loads(args.category_c.read_text())["global_selection"]
    parts, hashes = rv.load_allowed(args.prepared, manifest, names["B"])
    rv.attach_users(parts, args.source, manifest["input_files"][0])
    sims, emb = cf.read_sims(args.prepared, args.sims), read_embeddings(args.prepared, args.embeddings)
    nb = len(names["B"])
    for k in parts:
        if len(emb[k]) != len(parts[k].y) or len(sims[k]) != len(parts[k].y):
            raise ValueError("embeddings/similarities misaligned with prepared rows")
        # Columns: B(31) | sims(7) | embedding(384); purge/inner_split keep them aligned.
        parts[k].x = np.column_stack([parts[k].x, sims[k], emb[k]])
    cutoff = timestamp_us(manifest["boundaries"]["train_validation_cutoff"])
    train, validation, _ = rv.purge_and_check(parts["train"], parts["validation"], cutoff)
    fit, holdout, _ = rv.inner_split(train, args.inner_fraction)
    E = slice(nb + 7, None)
    bases = {"C_emb_PCA": list(range(nb)), "C_feat_sim_PCA": list(range(nb + 7))}
    base_names = {"C_emb_PCA": names["B"], "C_feat_sim_PCA": names["B"] + cf.SIM_NAMES}
    sel = prior["B"]
    max_k = max(KS)
    inner_pca = PCA().fit(fit.x[:, E], max_k)
    full_pca = PCA().fit(train.x[:, E], max_k)
    preds, inner, chosen = {}, {}, {}
    for m in ("A", "B"):
        c = [names["B"].index(n) for n in names[m]]
        preds[m] = cf.fit_fixed(train.x[:, c], train.y, validation.x[:, c], names[m], prior[m], args)
    preds["C_feat_sim"] = cf.fit_fixed(train.x[:, :nb + 7], train.y, validation.x[:, :nb + 7],
                                       base_names["C_feat_sim_PCA"], sel, args)
    for variant, base in bases.items():
        inner[variant] = {}
        for k in KS:
            fn = base_names[variant] + [f"pc{i + 1}" for i in range(k)]
            hp = cf.fit_fixed(design(fit, fit.x[:, E], base, inner_pca, k), fit.y,
                              design(holdout, holdout.x[:, E], base, inner_pca, k), fn, sel, args)
            inner[variant][k] = pinball(holdout.y, hp)
            preds[f"{variant}_k{k}"] = cf.fit_fixed(design(train, train.x[:, E], base, full_pca, k), train.y,
                                                    design(validation, validation.x[:, E], base, full_pca, k),
                                                    fn, sel, args)
            print(f"{variant} k={k}: inner holdout pinball={inner[variant][k]:.4f}", flush=True)
        chosen[variant] = min(KS, key=lambda k: inner[variant][k])
        preds[variant] = preds[f"{variant}_k{chosen[variant]}"]
    models = ("A", "B", "C_feat_sim", "C_emb_PCA", "C_feat_sim_PCA") + tuple(
        f"{v}_k{k}" for v in bases for k in KS)
    comparisons = tuple((m, ref) for m in models[3:] for ref in ("B", "C_feat_sim")) + (("C_feat_sim", "B"),)
    metrics = cg.score_table(validation, preds, models)
    boot = cg.bootstrap_diffs(validation, preds, args.seed, args.bootstrap_replicates, models, comparisons)
    result = {"issue": 19, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_split": "validation only; test split never read",
              "prepared_sha256": hashes, "sims_sha256": rv.fingerprint(args.sims),
              "embeddings_sha256": rv.fingerprint(args.embeddings), "fixed_hyperparameters": prior,
              "pca_explained_variance_full_train": full_pca.explained,
              "inner_holdout_pinball": inner, "chosen_k": chosen, "models": models,
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "validation_metrics": metrics, "bootstrap_ci95": boot, "elapsed_seconds": time.monotonic() - started}
    (args.output / "category_pca.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "category_pca.md").write_text(render(result))
    print("Wrote category_pca results", flush=True)
    return result


def table(r, models):
    lines = ["| 턴 | 가중 | " + " | ".join(models) + " |", "|---|---|" + "---:|" * len(models)]
    for c, w_ in r["validation_metrics"].items():
        for w, ms in w_.items():
            lines.append(f"| {c} | {w} | " + " | ".join(f'{ms[m]["mean_pinball_loss"]:.4f}' for m in models) + " |")
    return lines


def render(r):
    ch = r["chosen_k"]
    ev = sum(r["pca_explained_variance_full_train"][:32])
    lines = ["# #19 후속: MiniLM 임베딩 PCA 성분을 전역 B 특징으로 (validation 전용)", "",
             "test 분할 파일은 읽지 않았다. PCA는 train에서만 적합(선택 단계는 내부 fit 행, validation용은 전체 purged train).",
             "k는 #13 train 내부 홀드아웃 pinball로 골랐고 validation은 선택에 쓰지 않았다. 전역 B 하이퍼파라미터·반복 수 고정.", "",
             f'- 선택된 k: C_emb_PCA={ch["C_emb_PCA"]}, C_feat_sim_PCA={ch["C_feat_sim_PCA"]}. 32성분 설명 분산 합(전체 train) {ev:.3f}.', "",
             "## 내부 홀드아웃 pinball (k 선택)", "", "| 변형 | " + " | ".join(f"k={k}" for k in KS) + " |",
             "|---|" + "---:|" * len(KS)]
    for v, d in r["inner_holdout_pinball"].items():
        lines.append(f"| {v} | " + " | ".join(f"{d[str(k)] if str(k) in d else d[k]:.4f}" for k in KS) + " |")
    lines += ["", "## validation 평균 pinball (선택된 k)", ""] + table(r, r["models"][:5])
    lines += ["", "## 보조: 모든 k의 validation (선택에 사용하지 않음)", ""] + table(r, r["models"][5:])
    lines += ["", "## 사용자 bootstrap 95% CI (음수 = 앞 모델이 좋음)", "", "| 턴 | 가중 | 차이 | 점추정 | 95% CI |",
              "|---|---|---|---:|---|"]
    for c, w_ in r["bootstrap_ci95"].items():
        for w, ds in w_.items():
            for label, d in ds.items():
                a, b = label.split("-", 1)
                pt = r["validation_metrics"][c][w][a]["mean_pinball_loss"] - r["validation_metrics"][c][w][b]["mean_pinball_loss"]
                lo, hi = d["ci95"]
                lines.append(f"| {c} | {w} | {label} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ["", "- 임베딩(embeddings.npz)·유사도(sims.jsonl)는 저장소 밖 데이터 디렉터리에 있다. validation은 이미 여러 번 본 기간이다.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("prepared", "source", "sims", "embeddings", "category-c", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner-fraction", type=float, default=.8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = p.parse_args()
    with (args.output / "category_pca.log").open("w") as log:
        class Tee:
            def write(self, v):
                sys.__stdout__.write(v); log.write(v); return len(v)
            def flush(self):
                sys.__stdout__.flush(); log.flush()
        with redirect_stdout(Tee()):
            run(args)


if __name__ == "__main__":
    main()
