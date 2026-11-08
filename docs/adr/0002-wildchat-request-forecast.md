# ADR-0002: WildChat 현재 프롬프트 응답 토큰 예측을 별도 실험으로 추가
- 상태: 채택 → Superseded by [ADR 0005](0005-freeinference-agentic-trace.md) (2026-11-08, 현재 연구 범위). 코드·결과는 태그 `wildchat-final`
- 날짜: 2026-11-02
- 관련: Issue #4

## 작성 전 질문
- 무엇이 바뀌는가? 기존 집계 시계열과 다른 데이터·예측 단위가 추가되므로 PRD 일부 수정에 더해 ADR로 범위 변경을 기록한다.

## 맥락
BurstGPT 연구는 다음 10분에 도착하는 요청의 입력·최종 출력 합계를 예측한다. WildChat은 대화 원문과 메시지 구조를 제공하여 현재 프롬프트에 대한 개별 응답 길이를 조사할 수 있지만, 개별 요청 시각과 실제 API 사용량이 동일한 의미로 보장되지는 않는다.

## 결정
BurstGPT의 데이터, 목표, A/B/C 비교 및 평가를 보존하고 WildChat을 독립 트랙으로 추가한다. 인접 user→assistant 쌍의 가시 텍스트를 o200k_base로 각각 계수하여 입력과 응답 목표를 정의한다. 후속 GBDT 실험은 중복·공유 prefix 그룹을 보존하고 기록시각 순으로 분할한다. 메시지 시각의 의미가 검증되지 않으면 턴 간격을 사용하지 않는다. 이번 이슈는 적합성 조사까지이며 학습은 수행하지 않는다.

## 대안
- WildChat으로 BurstGPT를 대체: 기존 연구 질문과 비교가 바뀌므로 채택하지 않는다.
- WildChat을 사용하지 않음: 개별 프롬프트의 응답 길이와 대화 구조를 조사할 기회를 놓친다.

## 결과
두 목표와 토큰 의미를 혼동하지 않고 각각 조사할 수 있다. 현재 결과는 첫 샤드에 한정되며 전체 자료의 중복·그룹·시각 품질 및 분할 가능성을 확인해야 한다. 원문과 원본 데이터를 보고서나 저장소에 싣지 않는다.

## 후속 구현 상태 (2026-11-03 갱신)
- 위 “조사까지”는 #4의 범위다. 이후 #5 전처리·#6 분석·#7 길이 A 대 이력 B 분위수 실험을 완료했다.
- #13에서 사용자 분리·train 내부 튜닝·단순 기준선 2개 추가·분위수 무정렬·사용자 bootstrap으로 평가를 보정했다. 현재 비교는 fixed·직전 응답 비율·대화 누적 평균·A·B다. [재검증 결과](https://github.com/jongrionro-debug/TS_-/blob/wildchat-final/wildchat/results/revalidation.md)를 따른다.
- #15에서 소유자 승인에 따른 최종 test 1회 평가를 완료했다. test 승인 대기나 모델 구현 예정 상태가 아니다. [최종 결과](https://github.com/jongrionro-debug/TS_-/blob/wildchat-final/wildchat/results/test_final.md)·고정 설정·로그를 보존하고 재평가하지 않는다.
- 현재 작업 안내는 WildChat을 우선하되 BurstGPT 설계·EDA와 TraceLab·Exgentic 후보 검토는 이력으로 보존한다. 독립 트랙이라는 원래 결정과 토큰 의미는 바꾸지 않는다.
- 현재 연구 정의·한계는 [PRD](../project.md), 단계별 명령은 [WildChat README](https://github.com/jongrionro-debug/TS_-/blob/wildchat-final/wildchat/README.md)가 정본이다. 이번 갱신은 새 연구 결정이 아닌 이미 수행한 작업의 상태 반영이다.
