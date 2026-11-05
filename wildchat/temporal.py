#!/usr/bin/env python3
"""Issue 23: B / B+H1 / C_emb_PCA(32) / C_emb_PCA+H1, validation only."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import shlex
import time

import numpy as np

from wildchat import category as cg
from wildchat import revalidate as rv
from wildchat.category_pca import PCA
from wildchat.classify import MODEL, MAX_CHARS, SPLIT_FILES
from wildchat.prepare import timestamp_us
from wildchat.temporal_features import ALPHA, H1_NAMES, h1_features
from wildchat.train import SEARCH_SPACE, feature_names

MODELS = ("B", "B+H1", "C_emb_PCA", "C_emb_PCA+H1")
COMPARISONS = (("B+H1", "B"), ("C_emb_PCA+H1", "C_emb_PCA"))
PCA_K = 32


@contextmanager
def allowed_jsonl_reads(prepared):
    """Audit actual Python file opens; allow only the two declared JSONL inputs."""
    allowed = {(prepared / name).resolve() for name in SPLIT_FILES.values()}
    opened, active = set(), [True]

    def audit(event, args):
        if active[0] and event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
            if path.suffix == ".jsonl":
                if path.resolve() not in allowed:
                    raise PermissionError("JSONL access outside train/validation allowlist")
                opened.add(path.name)

    sys.addaudithook(audit)
    try:
        yield opened
    finally:
        active[0] = False


def load_embeddings(prepared, path, expected_hash):
    """Check prior checksum, format, unique keys and exact allowed-row coverage."""
    digest = rv.fingerprint(path)
    if digest != expected_hash:
        raise ValueError("embedding cache checksum differs from recorded provenance")
    with np.load(path, allow_pickle=False) as data:
        keys, vectors = data["keys"], data["vectors"]
        if (keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype.kind not in "iu" or
            vectors.shape != (len(keys), 384) or vectors.dtype.kind != "f" or
            not np.isfinite(vectors).all() or np.any(keys < 0)):
            raise ValueError("invalid embedding cache format or values")
        index = {tuple(map(int, key)): i for i, key in enumerate(keys)}
        if len(index) != len(keys):
            raise ValueError("duplicate embedding cache identities")
        out, seen = {}, set()
        for split, name in SPLIT_FILES.items():
            indices = []
            with (prepared / name).open("rb") as handle:
                for raw in handle:
                    meta = json.loads(raw)["metadata"]
                    key = (meta["source_row_id"], meta["user_message_index"])
                    if key not in index or key in seen:
                        raise ValueError("embedding cache missing or repeated prepared identity")
                    seen.add(key)
                    indices.append(index[key])
            out[split] = vectors[indices].astype(np.float32)
        if seen != set(index):
            raise ValueError("embedding cache has identities outside train/validation")
    return out, {"sha256": digest, "rows": len(index), "dimensions": 384,
                 "unique_keys": True, "exact_train_validation_coverage": True,
                 "model": MODEL, "prompt_prefix_characters": MAX_CHARS,
                 "regenerated": False, "reuse_authority": "issue 23 operator context"}


def designs(rows, base_names, pca):
    """B | H1 | embedding storage; PCA transform uses the supplied fit only."""
    nb, nh = len(base_names), len(H1_NAMES)
    b, h1 = rows.x[:, :nb], rows.x[:, nb:nb + nh]
    c = np.column_stack([b, pca.transform(rows.x[:, nb + nh:])])
    return {"B": b, "B+H1": np.column_stack([b, h1]),
            "C_emb_PCA": c, "C_emb_PCA+H1": np.column_stack([c, h1])}


def comparisons_with_points(metrics, intervals):
    for cohort, weighted in intervals.items():
        for weighting, comparisons in weighted.items():
            for a, b in COMPARISONS:
                entry = comparisons[f"{a}-{b}"]
                scores = metrics[cohort][weighting]
                entry["point"] = (scores[a]["mean_pinball_loss"] - scores[b]["mean_pinball_loss"]
                                  if scores[a] is not None else None)
    return intervals


def run(args):
    started = time.monotonic()
    if (not 0 < args.inner_fraction < 1 or args.seed < 0 or args.threads < 1 or
        min(args.max_rounds, args.early_stopping_rounds, args.bootstrap_replicates) < 1):
        raise ValueError("invalid experiment settings")
    manifest_path = args.prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["input_files"]) != 1 or manifest["lags"] != 3:
        raise ValueError("requires the declared single shard and B's three lags")
    names = feature_names(manifest["lags"])["B"]
    fnames = {"B": names, "B+H1": names + list(H1_NAMES),
              "C_emb_PCA": names + [f"pc{i + 1}" for i in range(PCA_K)],
              "C_emb_PCA+H1": names + [f"pc{i + 1}" for i in range(PCA_K)] + list(H1_NAMES)}
    reference = json.loads(args.embedding_reference.read_text())
    with allowed_jsonl_reads(args.prepared) as opened:
        parts, hashes = rv.load_allowed(args.prepared, manifest, names)
        rv.attach_users(parts, args.source, manifest["input_files"][0])
        embeddings, cache = load_embeddings(args.prepared, args.embeddings, reference["embeddings_sha256"])
        for split, rows in parts.items():
            h1 = np.asarray(h1_features(rows.sources, rows.turns, rows.x[:, names.index("input_tokens")], rows.y))
            rows.x = np.column_stack([rows.x, h1, embeddings[split]])
        train, validation, outer = rv.purge_and_check(
            parts["train"], parts["validation"], timestamp_us(manifest["boundaries"]["train_validation_cutoff"]))
        fit, holdout, inner = rv.inner_split(train, args.inner_fraction)
        emb_columns = slice(len(names) + len(H1_NAMES), None)
        if min(fit.x[:, emb_columns].shape) < PCA_K:
            raise ValueError("insufficient inner fit rows or embedding dimensions for PCA(32)")
        print("Fitting inner PCA on train-internal fit rows only", flush=True)
        inner_pca = PCA().fit(fit.x[:, emb_columns], PCA_K)
        fit_design, holdout_design = designs(fit, names, inner_pca), designs(holdout, names, inner_pca)
        selected = {}
        for model in MODELS:
            print(f"Selecting {model} on the same train-internal holdout", flush=True)
            f, h = fit.take(slice(None)), holdout.take(slice(None))
            f.x, h.x = fit_design[model], holdout_design[model]
            selected[model], _ = rv.select_model(f, h, list(range(len(fnames[model]))), fnames[model], args)
        print("All four selections frozen; fitting full-train PCA and refitting", flush=True)
        full_pca = PCA().fit(train.x[:, emb_columns], PCA_K)
        train_design, val_design = designs(train, names, full_pca), designs(validation, names, full_pca)
        predictions, diagnostics = {}, {}
        for model in MODELS:
            tr, va = train.take(slice(None)), validation.take(slice(None))
            tr.x, va.x = train_design[model], val_design[model]
            predictions[model], diagnostics[model] = rv.refit_predict(
                tr, va, list(range(len(fnames[model]))), fnames[model], selected[model], args)
            print(f"Refit {model} complete", flush=True)
        metrics = cg.score_table(validation, predictions, MODELS)
        boot = comparisons_with_points(metrics, cg.bootstrap_diffs(
            validation, predictions, args.seed, args.bootstrap_replicates, MODELS, COMPARISONS))
    command = shlex.join([sys.executable, "wildchat/temporal.py"] +
                         [item for k, v in vars(args).items() for item in ("--" + k.replace("_", "-"), str(v))])
    result = {"issue": 23, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "evaluation_split": "validation only", "models": list(MODELS), "features": fnames,
              "h1_definition": {"alpha": ALPHA, "ewma": "first valid value initialization; adjust=False",
                  "std_ddof": 0, "slope": "last two valid outputs difference / original user-turn gap",
                  "previous": "previous eligible pair in the same source conversation, not group or user",
                  "missing": "NaN except count=0; std=0 for one prior observation",
                  "zero_input": "ratios undefined; ratio EWMA skips zero-denominator observations",
                  "update_order": "snapshot before adding current response; current input is available"},
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "versions": {"python": sys.version.split()[0], **{n: importlib.metadata.version(n)
                           for n in ("numpy", "lightgbm", "scipy", "pyarrow")}},
              "source": {"prepared_sha256": hashes, "manifest_sha256": rv.fingerprint(manifest_path),
                         "metadata_checks": ["source filename", "source row count", "allowed source timestamps"],
                         "embedding_reference_sha256": rv.fingerprint(args.embedding_reference)},
              "data_access": {"jsonl_allowlist_enforced": True, "opened_jsonl_files": sorted(opened),
                              "source_columns": ["hashed_ip", "timestamp"], "raw_text_read": False},
              "embedding_cache": cache, "splits": {"outer": outer, "inner": inner},
              "pca": {"k": PCA_K, "selection": "fixed by issue before validation evaluation",
                      "inner_fit_rows": len(fit.y), "full_train_rows": len(train.y),
                      "inner_explained_variance": inner_pca.explained,
                      "full_train_explained_variance": full_pca.explained},
              "training": {"common_parameters": rv.common_parameters(args.seed, args.threads),
                           "search_space": list(SEARCH_SPACE), "models": selected,
                           "selection": "each condition: same train-internal row mean pinball; first candidate on ties",
                           "refit": "full purged train; selected iterations; no validation callbacks",
                           "quantile_rule": "clip negative predictions only; no quantile sorting"},
              "code_sha256": {f"wildchat/{name}": rv.fingerprint(Path(__file__).with_name(name)) for name in
                              ("temporal.py", "temporal_features.py", "category_pca.py", "category.py",
                               "classify.py", "revalidate.py", "train.py", "prepare.py", "metrics.py")},
              "raw_validation_diagnostics": diagnostics, "validation_metrics": metrics,
              "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.seed, "confidence_level": .95,
                            "unit": "user", "paired": True, "rng": "numpy.random.PCG64",
                            "shared_draws": "all four models, cohorts and weighting schemes",
                            "method": "percentile 2.5/97.5; linear interpolation; no refit/reselection"},
              "bootstrap_ci95": boot, "elapsed_seconds": time.monotonic() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "temporal_h1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "temporal_h1.md").write_text(render(result))
    print("Wrote temporal_h1 JSON and Markdown", flush=True)
    return result


def render(r):
    lines = ["# #23 H1: 같은 대화 순차 특징 (validation 전용)", "",
             "B, B+H1, C_emb_PCA(k=32), C_emb_PCA+H1을 같은 정제 표본·분할에서 비교했다.",
             "각 조건에 동일한 3개 후보와 train 내부 홀드아웃 선택·조기 종료 규칙을 적용했다. validation은 보고에만 썼다.",
             "#19의 B 설정을 모든 조건에 고정했던 PCA 실험과 달리 이번에는 네 조건을 각각 내부 튜닝한다. Mac 탐색 코드의 재실행이 아니다.", "",
             "- H1: 이전 유효 출력의 EWMA(α=0.3), 누적 평균·모집단 표준편차, 최근 두 출력 차이·원래 user 턴 간격당 기울기, 직전 유효 출력/입력 비율·비율 EWMA, 현재/직전 유효 입력 배율, 관측 수(9개).",
             "- 같은 source 대화에서 특징을 먼저 기록한 뒤 현재 응답을 상태에 넣는다. 유효하지 않은 쌍은 제외하고 원래 턴 간격은 유지한다. 사용자·중복 그룹 사이에 상태를 공유하지 않는다.",
             "- 이력이 없으면 count=0 외에는 NaN, 관측 1개면 std=0, 2개 미만이면 차이·기울기는 NaN. 입력 0의 비율은 NaN이고 비율 EWMA에서만 건너뛴다. EWMA는 첫 유효 값으로 초기화한다.",
             "- 운영자 지시에 따라 기존 MiniLM 캐시를 재사용했다. 기존 SHA256·384차원 형식·키 유일성·train/validation 전체 키의 정확한 일치를 검사했다. 원문은 읽지 않았다.",
             "- PCA(32)는 선택 시 내부 fit만, 최종 val 예측 시 전체 정제 train만으로 적합했다. 음수 예측만 0으로 자르고 분위수는 정렬하지 않았다.",
             "- JSONL 접근은 train/validation 두 경로로 실행 중 제한했다. test 분할 접근·최종 평가 실행은 하지 않았다.", "",
             "## 표본과 내부 선택", "", "| 구간 | 행 | 사용자 | 첫 턴 |", "|---|---:|---:|---:|"]
    for scope, audit in r["splits"].items():
        for side, desc in audit["after"].items():
            lines.append(f'| {scope}/{side} | {desc["samples"]} | {desc["users"]} | {desc["first_turn_samples"]} |')
    lines.append("")
    for scope, audit in r["splits"].items():
        lines.append(f'- {scope} 분리 검사: {json.dumps(audit["checks"], ensure_ascii=False)}')
    lines += ["", "| 조건 | num_leaves | min_data_in_leaf | P10/P50/P90 반복 |", "|---|---:|---:|---|"]
    for model, selection in r["training"]["models"].items():
        p = selection["selected_parameters"]
        iterations = "/".join(str(q["best_iteration"]) for q in selection["selected_quantile_fits"])
        lines.append(f'| {model} | {p["num_leaves"]} | {p["min_data_in_leaf"]} | {iterations} |')
    lines += ["", "## validation 평균 pinball (P10/P50/P90 평균, 낮을수록 좋음)", "",
              "행 가중은 행 전체 평균, 사용자 가중은 각 턴 구분 안의 사용자별 평균 후 사용자 동일 가중이다.",
              "첫 턴은 turn_index==0, 후속 턴은 >0이다.", "",
              "| 턴 | 가중 | " + " | ".join(MODELS) + " |", "|---|---|" + "---:|" * len(MODELS)]
    for cohort, weighted in r["validation_metrics"].items():
        for weighting, models in weighted.items():
            cells = [f'{models[m]["mean_pinball_loss"]:.6f}' if models[m] else "NA" for m in MODELS]
            lines.append(f"| {cohort} | {weighting} | " + " | ".join(cells) + " |")
    lines += ["", "## H1 추가 차이와 사용자 paired bootstrap 95% CI", "",
              f'사용자 복원추출 {r["bootstrap"]["replicates"]:,}회, seed {r["bootstrap"]["seed"]}. 같은 추출을 네 조건·모든 턴·가중에 공유한다. 음수는 H1 개선이다.', "",
              "| 턴 | 가중 | 차이 | 점추정 | 95% CI | 유효 반복 |", "|---|---|---|---:|---|---:|"]
    for cohort, weighted in r["bootstrap_ci95"].items():
        for weighting, comparisons in weighted.items():
            for label, entry in comparisons.items():
                if entry["ci95"] is None:
                    values = "NA | NA"
                else:
                    lo, hi = entry["ci95"]
                    values = f'{entry["point"]:.6f} | [{lo:.6f}, {hi:.6f}]'
                lines.append(f'| {cohort} | {weighting} | {label} | {values} | {entry["effective_replicates"]} |')
    lines += ["", "## 재현·한계", "", "```sh", "export WILDCHAT_DATA_DIR=/workspace/data/wildchat-1m",
              "OPENBLAS_NUM_THREADS=1 " + r["command"], "```", "",
              "설정·후보별 내부 점수·패키지 버전·입력/코드 해시·키 정합성·분리 검사는 [temporal_h1.json](temporal_h1.json), 실행 로그는 [temporal_h1.log](temporal_h1.log)에 있다.", "",
              "- 첫 샤드의 고정 표본이며 validation은 여러 탐색에서 이미 사용됐다. Mac 탐색의 유의성이나 독립 미관측 성능을 확증하는 평가가 아니다.",
              "- CI는 고정된 학습 결과·평가 표본에 조건부다. 학습·선택 불확실성, 시간 블록 상관, IP 프록시 식별 오차를 포함하지 않는다.",
              "- 첫 턴의 H1은 count=0 외에는 결측이다. 첫 턴 점수 차이는 공유 모델의 학습·선택 변화로 생길 수 있으며 첫 턴의 과거 응답 정보를 뜻하지 않는다.",
              "- 과거 인접 응답은 현재 프롬프트 전에 관측됐다고 가정한다. 행 기록시각은 실제 요청 도착시각이 아니다. 실제 API 토큰·GPU 효과나 전체 WildChat으로 일반화하지 않는다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "source", "embeddings", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--embedding-reference", type=Path, default=Path("wildchat/results/category_pca.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-fraction", type=float, default=.8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
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

    with (args.output / "temporal_h1.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)):
            run(args)


if __name__ == "__main__":
    main()
