# 요청을 10분 bin으로 보는 운영 관점

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
