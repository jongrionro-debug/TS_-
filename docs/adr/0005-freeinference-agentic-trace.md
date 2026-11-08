# 0005 · 연구 데이터를 FreeInference agentic trace로 바꾸고 요청 단위 출력 토큰을 예측한다

- Status: Proposed
- Date: 2026-11-08
- Related Work: 2026-11-08 사용자 요청(재설계), 제안 발표 피드백(Alfredo, Group 5)
- Approval: 방향(요청 단위 출력 토큰·WildChat 태그 후 main 재구성)은 2026-11-08 사용자 선택. 분할·특징 세부는 데이터 감사 후 승인 필요
- Supersedes: [ADR 0002](0002-wildchat-request-forecast.md)·[ADR 0004](0004-wildchat-category-condition-c.md)의 현재 연구 범위 전체

## 맥락
WildChat 연구는 첫 샤드의 가시 텍스트 재계수 길이를 예측했고 user 시작시각이 없어 10분 수요 시계열을 만들 수 없었다. 발표 피드백은 (1) 데이터가 지원하려는 워크로드(최신 모델·에이전트)를 얼마나 대표하는지, (2) 세션 상태가 강하게 상관된 최근 수요 lag 이상으로 기여하는지와 세션 내·간 분석, 예측 시점 가용성, 작은 개선의 불확실성, (3) 요청→10분 입력·출력 토큰 시계열 도식과 실용 출력의 명확화를 요구했다.

FreeInference trace는 16주간 실제 에이전트 세션의 요청 시작·종료 시각, provider 보고 completion 토큰, 역할별 입력 토큰, tool 호출·결과 크기를 제공한다. 메시지 원문은 없다.

## 결정
- 데이터를 FreeInference `requests`·`tool_calls`로 바꾸고 revision을 고정한다.
- 예측 단위는 요청 시작 시점의 해당 요청 completion 토큰 P10/P50/P90. 10분 집계는 요청 예측을 모은 운영 뷰로 제시한다.
- 주 비교는 B(세션 상태) − L(현재 요청 + 직전 completion lag)로 사전 고정한다.
- 이력은 `end_j < s_i`인 완료 요청만 쓰며, 시간 분할(1~10주 train, 11~13주 validation, 14~16주 test)과 세션 단위 bootstrap CI를 쓴다. 세부는 [project.md](../project.md).
- WildChat 코드·결과는 태그 `wildchat-final`과 기존 원격 브랜치에 보존하고 main에서 제거한다.

## 대안
- 10분 집계 수요를 주 목표로: 피드백 3과 직접 맞지만 세션 상태 기여를 분리하기 어렵고 표본 수가 적다(16주 ≈ 1.6만 구간). 운영 뷰로 유지한다.
- WildChat 유지·확장: 에이전트 워크로드와 시각 정보가 없어 피드백 1·3을 충족하기 어렵다.
- 기존 Issue #26 초안 그대로 진행: 요청 단위 방향은 같지만 10분 뷰·세션 간 분석이 빠져 있어 작은 Issue로 나눠 다시 쓴다.

## 영향
- 실측 시각 기반 10분 시계열과 세션 내·간 분석이 가능하다.
- 세션 재구성이 사후 연결이라 예측 시점 가용성 검증이 필요하다. 주별 요청 수가 크게 늘어 분포 이동이 있다. 오픈 가중치·무료 gateway 사용자로 결론 범위를 한정한다.
- 검증: 감사 Issue에서 스키마·결측·중복·세션 연결 가용성을 확인한 뒤 이 ADR을 Accepted로 바꾸거나 수정한다.
