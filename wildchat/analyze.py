#!/usr/bin/env python3
"""Summarize training data and evaluate its constant baseline on validation."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path

from metrics import baseline, empirical_quantile, evaluate


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def distribution(values):
    return [len(values), sum(values) / len(values), *[empirical_quantile(values, q) for q in (.1, .5, .9)], max(values)]


def fmt(values):
    return " | ".join(f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    inputs, outputs, conversations = [], [], Counter()
    lag_counts = [[0, 0, 0] for _ in range(3)]
    for row in rows(args.prepared / "train.jsonl"):
        if row["metadata"]["split"] != "train":
            raise ValueError("unexpected training split")
        inputs.append(row["prompt_features"]["input_tokens"])
        outputs.append(row["target"]["output_tokens"])
        conversations[row["metadata"]["source_row_id"]] += 1
        h = row["history_features"]
        for i in range(3):
            for j, suffix in enumerate(("turn", "input", "output")):
                lag_counts[i][j] += not h[f"lag_{i+1}_{suffix}_missing"]
    quantiles = baseline(outputs)
    def validation_targets():
        for row in rows(args.prepared / "validation.jsonl"):
            if row["metadata"]["split"] != "validation":
                raise ValueError("unexpected validation split")
            yield row["target"]["output_tokens"]
    metrics = evaluate(validation_targets(), itertools.repeat(quantiles, manifest["partitions"]["validation"]["samples"]))
    if len(outputs) != manifest["partitions"]["train"]["samples"]:
        raise ValueError("training sample count differs from manifest")
    result = {"prepared": str(args.prepared), "fit_split": "train", "evaluation_split": "validation",
              "quantile_method": "inverse empirical CDF, ceil(n*q)", "crossing_rule": "reject",
              "baseline_output_tokens": dict(zip(("p10", "p50", "p90"), quantiles)), "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "baseline_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mx, my = sum(inputs)/len(inputs), sum(outputs)/len(outputs)
    covariance = sum((x-mx)*(y-my) for x,y in zip(inputs,outputs))
    denom = math.sqrt(sum((x-mx)**2 for x in inputs) * sum((y-my)**2 for y in outputs))
    corr = covariance / denom if denom else float('nan')
    lines = ["# WildChat 학습 표본 분석 및 단순 기준", "",
        f"- 전처리 산출물: `{args.prepared}`. #5의 동일 표본과 `o200k_base` 계수를 사용한다.",
        f'- 출처: WildChat-1M 리비전 `{manifest["revision"]}`의 14개 중 첫 샤드 {manifest["input_files"][0]["rows"]:,}행 전체를 순서대로 처리한 결정적 표본이다. 추가 추출은 없고 전체 데이터의 대표 표본이 아니다.',
        "- 대화 그룹과 기록시각 순 분할 및 경계 걸침 제외는 #5 결과를 그대로 유지한다. 기록시각은 개별 요청 도착시각을 보장하지 않는다. 중복 표본은 유지된다.",
        "- 분할별 표본 수는 manifest에서 읽는다. 데이터에서 계산하는 분포·관계·기준은 train만 사용하고, 기준 지표는 validation만 사용한다. test는 읽거나 평가하지 않는다.", "",
        "| 분할 | 그룹 수 | 대화 기록 행 수 | 유효 쌍 표본 수 |", "|---|---:|---:|---:|"]
    for split in ("train", "validation", "test"):
        p = manifest["partitions"][split]
        lines.append(f'| {split} | {p["groups"]:,} | {p["conversations"]:,} | {p["samples"]:,} |')
    lines += ["", "## 학습 분할 분포", "", "분위수는 정렬한 n개 값 중 ceil(n×q)번째 값(경험적 누적분포의 역함수)이다. 대화 길이는 원본 대화 기록 행별 **분할에 남은 유효 user→assistant 쌍 수**이며 원본 전체 메시지 수가 아니다.", "",
        "| 항목 | 수 | 평균 | P10 | P50 | P90 | 최대 |", "|---|---:|---:|---:|---:|---:|---:|",
        f'| 입력 토큰/쌍 | {fmt(distribution(inputs))} |', f'| 출력 토큰/쌍 | {fmt(distribution(outputs))} |', f'| 유효 쌍/대화 기록 행 | {fmt(distribution(list(conversations.values())))} |', "",
        "| 학습 lag | 이전 턴 존재 | 입력 토큰 사용 가능 | 출력 토큰 사용 가능 |", "|---|---:|---:|---:|"]
    for i, counts in enumerate(lag_counts, 1):
        lines.append(f'| {i} | ' + ' | '.join(f'{c:,} ({100*c/len(outputs):.2f}%)' for c in counts) + ' |')
    lines += ["", f"학습 표본의 현재 입력과 출력 토큰 간 Pearson 상관계수는 {corr:.4f}이다. 이는 기술 통계이며 인과관계나 예측 성능 향상의 증거가 아니다.",
        "이전 user 제출시각의 의미가 검증되지 않았고 첫 샤드의 user 시각은 모두 결측이므로 턴 간격 분석은 생략한다. assistant 완료시각이나 행 기록시각으로 대체하지 않는다.", "",
        "## 단순 기준과 검증", "", f"학습 출력의 P10/P50/P90은 각각 {quantiles[0]:,}, {quantiles[1]:,}, {quantiles[2]:,}토큰이며 모든 검증 표본에 같은 값을 예측한다.",
        "분위수 교차(P10 > P50 또는 P50 > P90)는 정렬이나 보정 없이 오류로 거부한다. 포함률은 양 끝을 포함한다. pinball 평균은 세 분위수 손실의 산술평균이다.",
        f'검증 {metrics["n"]:,}쌍: P50 MAE {metrics["mae_p50"]:.4f}, 평균 절대 log1p 오차 {metrics["mean_absolute_log1p_error_p50"]:.4f}, 평균 pinball {metrics["mean_pinball_loss"]:.4f}, P10~P90 포함률 {metrics["p10_p90_coverage"]:.4f}, 평균 구간 폭 {metrics["mean_interval_width"]:.4f}. 각 분위수 손실과 전체 정밀도는 [baseline_metrics.json](baseline_metrics.json)에 기록한다.', ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
