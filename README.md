# 에이전트 세션 상태로 LLM 출력 토큰 수요 예측

시계열 데이터 분석 및 실습 과목 프로젝트(Group 5). FreeInference agentic trace에서 **LLM 요청이 시작될 때 그 요청의 completion 토큰 P10/P50/P90을 예측**하고, 세션 상태가 현재 요청 정보·직전 출력 lag 이상으로 기여하는지 검증한다. 요청별 예측은 10분 출력 토큰 수요 뷰로 모아 보인다.

현재 상태: 1~13주 데이터 감사·10분 집계·EDA·A/L/B 특징과 고정 분할·기준선 4종 validation 평가·세션 bootstrap CI 구현. 현재 입력 기반 연결과 엄격 완료 시각 조건을 적용한다. 연구 정의·시점 계약·평가는 [docs/project.md](docs/project.md), 결정 이유는 [ADR 0005](docs/adr/0005-freeinference-agentic-trace.md).

```
요청 i (세션 S, step k)      start_ms ──────────── end_ms
  예측 시점 = start_ms        특징: 현재 요청 정보 + end_j < start_i 인 과거 요청·tool 결과
  출력 = completion P10/P50/P90
        │
        ▼ start_ms 기준 10분 구간으로 합산
10분 시계열    [00:00) 입력합·출력합  [00:10) ...  [00:20) ...
```

## 작업 방식

Issue → geass(Codex 구현 → 고정 검증 → Claude 검수) → PR → 사람이 병합. 절차는 [workflow.md](docs/workflow.md), 규칙은 [AGENTS.md](AGENTS.md).

## 데이터 감사 재현

고정 revision의 준비된 `release/`와 `REVISION` 파일을 `FREEINFERENCE_DATA_DIR`에 둔다(기본 `data/freeinference`). `pip install -r requirements.txt`로 의존성을 설치한 뒤 `python -m freeinference.audit`를 실행한다. [감사 결과](freeinference/outputs/audit/audit.md)는 train·validation만 다루며 test 주차와 test 시점 요청은 로더가 기본 거부한다. `block_ids`는 읽지 않는다.

## 10분 집계와 EDA 재현

같은 준비 데이터와 의존성으로 `python -m freeinference.eda`를 실행한다. [EDA 결과](freeinference/outputs/eda/eda.md)에 train·validation별 표, 생성 그림과 실행 revision이 기록된다.

## A·L·B 특징 생성 재현

같은 준비 데이터와 의존성으로 `python -m freeinference.features`를 실행한다. [집계 산출물](freeinference/outputs/features/features.json)에 1~13주의 생성 시간·행 수·특징별 결측률이 기록된다. 특징 정의·결측 규칙·4절 항목 대조표는 [project.md](docs/project.md)에 있으며, 요청별 특징은 `build_features` API로 반환한다. 특징 생성 명령은 C·분할·학습·평가를 포함하지 않는다.

## 기준선 validation 평가 재현

준비된 1~13주 데이터만 사용한다. Ubuntu 분석 환경에서는 다음과 같이 실행한다.

```sh
FREEINFERENCE_DATA_DIR=~/data/freeinference/bb26cf7 OPENBLAS_NUM_THREADS=1 ~/.venvs/ts-freeinference/bin/python -m freeinference.evaluate
```

[기준선 보고서](freeinference/outputs/baselines/baselines.md)와 [집계 JSON](freeinference/outputs/baselines/baselines.json)을 생성한다. 1~10주 train 정답의 종료 경계 마스크로 기준선 4종을 맞추고, 11~13주 validation의 요청·세션 가중 지표와 fixed 대비 세션 bootstrap 95% CI(B=2000, seed=20261109)를 기록한다. 분할·holdout·대체 규칙과 재사용 API는 [project.md 4·5절](docs/project.md)을 따른다. 명령·환경·소요 시간·소스 해시는 JSON에 기록한다. 요청별 값·식별자를 내보내지 않으며 test는 로더가 거부한다. 실제 데이터 평가는 아래 검증 명령에 포함하지 않는다.

## 검증

```sh
python3 scripts/verify.py
python3 -m unittest discover -s tests
```

같은 명령이 [.harness/verify.toml](.harness/verify.toml)과 CI에서 실행된다. 원본 trace는 `data/`(Git 제외)에 두며 커밋하지 않는다.

## 이전 연구

WildChat 현재 프롬프트 응답 길이 예측(사용자 분리 재검증·최종 test 1회 완료)과 BurstGPT EDA는 태그 `wildchat-final`에 보존한다. 미병합 탐색 브랜치(H1·교차 대화·Alfredo P0)는 원격 브랜치로 남아 있다.

## 문서

[문서 안내](docs/README.md) · [현재 프로젝트](docs/project.md) · [작업 절차](docs/workflow.md) · [결정 기록](docs/adr/README.md)
