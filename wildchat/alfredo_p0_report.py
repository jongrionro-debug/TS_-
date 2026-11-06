"""Render P0 interpretation from saved aggregate JSON, without data/model access.

This separate presentation step preserves the executed numerical runner hash.
All destinations must be new files; use a temporary destination to review a
rendered replacement for a P0-created Markdown report.
"""
import argparse
from pathlib import Path
import json

from wildchat import alfredo_p0 as p0


def ablation_report(raw, output):
    r = json.loads(raw.read_text())
    row = r["bootstrap_ci95"]["all"]["row"]["differences"]["B+H1-B+lag1-only"]
    user = r["bootstrap_ci95"]["all"]["user"]["differences"]["B+H1-B+lag1-only"]
    reference = r["validation_metrics"]["all"]["row"]["B+lag1-only"]
    missing = r["missingness_and_redundancy"]["validation"]
    conclusion = ["## 실제 차이의 크기와 해석", "",
        f'전체 행 가중 H1−lag1 pinball은 {p0.cell(row["mean_pinball_loss"])}이며, lag1 대비 감소율은 {-100*row["mean_pinball_loss"]["point"]/reference["mean_pinball_loss"]:.2f}%다.',
        f'P50 MAE 감소도 {-row["mae_p50"]["point"]:.3f}토큰({-100*row["mae_p50"]["point"]/reference["mae_p50"]:.2f}%)로 작다. 두 행 가중 CI는 0 아래지만 큰 개선이라고 주장하지 않는다.',
        f'전체 사용자 가중 H1−lag1 pinball {p0.cell(user["mean_pinball_loss"])}, MAE {p0.cell(user["mae_p50"])}는 모두 CI가 0을 포함한다. 후속 턴의 사용자 가중 CI도 0을 포함한다.',
        "따라서 이 val에서 두 조건의 실용적 차이는 작으며, 모든 사용자에 걸친 H1 우월성을 확인했다고 해석하지 않는다. 동등성 검정이나 독립 확증은 아니다.",
        f'validation에서 둘 다 관측된 {missing["both_present"]:,}행 중 {missing["equal_when_both_present"]:,}행({100*missing["equal_when_both_present"]/missing["both_present"]:.2f}%)의 새 lag1은 B의 기존 출력 lag1과 같다. B에 이미 있는 정보가 대부분이고, 이 비교만으로 H1 효과를 직전 출력 하나의 효과와 완전히 분해할 수 없다.',
        "B의 기존 슬롯에는 유효 user→assistant 쌍이 아니더라도 관측 가능한 출력이 남을 수 있다. H1/새 lag1은 과거 유효 쌍만 누적하므로, 추가값이 항상 기존 B 값과 같지는 않다. 특징 선택·표현 중복·내부 튜닝 차이가 함께 작용한다.", "",
        "이 해석은 [표시 코드](../alfredo_p0_report.py)가 저장된 JSON으로 생성했다. 재학습·test 접근 없이 재현:", "",
        "```sh",
        "PYTHONDONTWRITEBYTECODE=1 /workspace/venvs/wildchat/bin/python -m wildchat.alfredo_p0_report ablation --input wildchat/results/lag1_ablation.json --output /tmp/alfredo-interpretation.md",
        "```", ""]
    text = p0.render_ablation(r)
    text = text.replace("## 누출 검사·한계·재현", "\n".join(conclusion)+"\n## 누출 검사·한계·재현")
    p0.save_text(output, text)


def bin_notes(output):
    """Provide the concept and runnable matplotlib source even if unavailable."""
    from datetime import datetime
    requests = [(1,100,220), (7,50,80), (12,120,300), (18,80,100), (23,90,180)]
    p0.save_json(output.with_suffix(".json"), {
        "scope": "toy aggregation executed; matplotlib rendering failed; no model trained/evaluated",
        "recorded_at_kst": datetime.now(p0.KST).isoformat(), "command": " ".join(p0.sys.argv),
        "timezone": "KST UTC+09:00", "requests_minute_from_09_00_input_output": requests,
        "arrival_bin_start_minute_sums_input_output": p0.toy_bins(requests),
        "output_is_final_not_available_at_start": True, "source": "hypothetical values, not WildChat rows",
        "png_created": False, "failure": "ModuleNotFoundError: matplotlib; isolated /tmp installation failed: PyPI DNS name resolution",
        "code_sha256": {"wildchat/alfredo_p0.py": p0.rv.fingerprint(Path(p0.__file__)),
                        "wildchat/alfredo_p0_report.py": p0.rv.fingerprint(Path(__file__))}})
    p0.save_text(output, """# 요청을 10분 bin으로 보는 운영 관점

모든 시각·토큰은 **가상 예시(toy)**다. KST(UTC+09:00)로 표시한다. NEW model을 만들거나 학습·평가하지 않았다.

| 요청 시작(KST) | 도착 시 입력 토큰 | 나중에 관측한 최종 출력 토큰 | 시작 기준 10분 bin |
|---|---:|---:|---|
| 09:01 | 100 | 220 | [09:00,09:10) |
| 09:07 | 50 | 80 | [09:00,09:10) |
| 09:12 | 120 | 300 | [09:10,09:20) |
| 09:18 | 80 | 100 | [09:10,09:20) |
| 09:23 | 90 | 180 | [09:20,09:30) |

집계하면 입력합/출력합은 각각 150/300, 200/400, 90/180이다. 시작 경계 09:10에 도착한 요청은 다음 bin에 들어간다. 정답 출력합은 **그 bin에 시작한 요청의 최종 출력 전체**이며, bin 안에서 실제 생성한 토큰량이 아니다.

09:20 예측 시점에는 **직전 closed bins에서 당시 관측 가능한 집계**를 입력으로, 다음 [09:20,09:30)의 입력합·출력합을 예측하는 관점이다. 마지막 bin의 90/180은 나중에 확인할 가상 정답이며 사전 예측값이 아니다. 현재 도착한 한 요청의 응답 길이를 맞히는 B/H1과 구분한다.

**요청 시작 시 complete output은 모른다.** 도착 구간이 닫혀도 그 안 응답은 미완료일 수 있다. 과거 출력합을 특징으로 쓰려면 완료시각으로 가용성을 확인하고 지연된 완결 bin만 쓰거나 미완료 상태를 별도로 관리해야 한다. 오프라인 최종 출력합을 즉시 제공하면 미래 누출이다. 입력합도 도착 로그가 완전한지 확인해야 한다.

WildChat 첫 샤드의 user 시작시각은 모두 결측이며 행 timestamp는 마지막 기록시각이다. assistant 수신시각만으로 검증된 시작/생성/완료 타임라인이 생기지 않고, prepared 표본에는 검증된 요청 완료시각도 없다. 따라서 여기서는 실제 10분 수요·출력 생성량·GPU 점유를 복원하거나 성능을 주장하지 않는다. 근거: `wildchat/results/data_audit.md:81`~`:84`, `wildchat/prepare.py:230`~`:292`. 과거 별도 도착 기준 집계 설계는 `docs/prd.md:101`~`:110`을 따른다.

## 그림 생성 상태와 재현

VM의 `/workspace/venvs/wildchat/bin/python`과 `wildchat-c`, 시스템 Python에 matplotlib이 없다. `/tmp` 설치 시도는 PyPI DNS 연결 실패로 끝났다. **요청된 matplotlib PNG는 미완료**다. 아래 코드에는 한 장짜리 diagram 구현이 있으며, 환경 제한을 우회하지 않았다. 이 문서의 표는 임시 개념 설명이고 그림 생성 성공으로 보고하지 않는다.

```sh
MPLCONFIGDIR=/tmp/alfredo-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /workspace/venvs/wildchat/bin/python -m wildchat.alfredo_p0 --task diagram --output /tmp/alfredo-bins
```

matplotlib이 준비된 환경에서 위 CLI는 PNG·Markdown·raw toy JSON을 새 경로에 만든다. 구현은 `wildchat/alfredo_p0.py`의 `toy_bins`/`diagram`이다. 이번 VM에서 실제 실행한 toy 집계는 [ten_minute_bins.json](ten_minute_bins.json)에 따로 저장했다. 집계·이 문서 재현은 `python -m wildchat.alfredo_p0_report bin-notes --output /tmp/alfredo-bin-notes.md`다.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("ablation", "bin-notes"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.task == "ablation":
        ablation_report(args.input, args.output)
    else:
        bin_notes(args.output)


if __name__ == "__main__":
    main()
