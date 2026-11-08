# 에이전트 세션 상태로 LLM 출력 토큰 수요 예측

시계열 데이터 분석 및 실습 과목 프로젝트(Group 5). FreeInference agentic trace에서 **LLM 요청이 시작될 때 그 요청의 completion 토큰 P10/P50/P90을 예측**하고, 세션 상태가 현재 요청 정보·직전 출력 lag 이상으로 기여하는지 검증한다. 요청별 예측은 10분 출력 토큰 수요 뷰로 모아 보인다.

현재 상태: 재설계 직후, 데이터 감사 전. 연구 정의·시점 계약·평가는 [docs/project.md](docs/project.md), 결정 이유는 [ADR 0005](docs/adr/0005-freeinference-agentic-trace.md).

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
