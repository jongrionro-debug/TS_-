# 프로젝트: 에이전트 세션 상태로 LLM 요청의 출력 토큰 수요 예측

- 과목: 시계열 데이터 분석 및 실습 (Group 5, "Forecasting LLM Token Demand with Conversation State")
- 최종 갱신일: 2026-11-08
- 현재 상태: **재설계 직후. 데이터 감사 전.** 아래 정의는 [ADR 0005](adr/0005-freeinference-agentic-trace.md)의 제안이며 감사 결과에 따라 첫 학습 전에 확정한다.
- 이전 연구: WildChat 현재 프롬프트 응답 길이 예측(최종 test 완료)은 태그 `wildchat-final`에 보존한다. 이 문서의 수치·정의와 섞지 않는다.

## 1. 연구 질문

> 에이전트 세션에서 LLM 요청이 시작될 때, **세션 상태**(앞선 요청·tool 호출의 진행)가 현재 요청 정보와 **최근 수요(직전 출력 lag)** 이상으로 그 요청의 completion 토큰 분포(P10/P50/P90)를 더 잘 예측하는가?

실용 출력은 요청마다의 completion 토큰 P10/P50/P90이며, 이를 시작시각 기준 10분 구간으로 모은 **출력 토큰 수요 뷰**를 함께 제시한다. 이번 데이터는 실제 요청 시작·종료 시각이 있어 10분 입력·출력 토큰 시계열을 실측으로 만들 수 있다.

## 2. 데이터

- 출처: Hugging Face `harvardMadsys/freeinference_agentic_trace` (CC BY 4.0), 논문 *From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads*. 사용할 revision은 감사 Issue에서 커밋 해시로 고정한다.
- 범위(카드 기준): 2026-05-17~09-05의 16주, 12,002 세션, 267 계정, 14 harness, LLM 요청 1,186,582건, tool 호출 1,213,347건. **메시지 원문 없음.**
- 사용 구성: `requests`(요청 단위: `session_id, step, start_ms, end_ms, ttft_ms, model, provider, stream, step_trigger, waited_for, link_reason, prompt_tokens, completion_tokens, cache_read_tokens, role_tokens, n_messages, tool_definition_names, finish_reason, transition, cause, block_ids`)와 `tool_calls`. `block_ids`는 크기가 커서 읽지 않는다. 원본 `traces/*.jsonl.zst`는 subagent 중첩 구조가 필요할 때만 쓴다.
- 주의(카드): 입력 토큰은 provider tokenizer가 아닌 `o200k_base` 재계수, `completion_tokens`는 provider 보고값. 세션은 tool-call ID·assistant 메시지 반복으로 **사후 재구성**되며 주 경계를 넘지 않는다. 드문 tool 이름은 `other`로 치환. 주별 요청 수가 1만→19만 건으로 크게 증가하는 비정상 구간이다.

### 대표성과 결론 범위 (피드백 1)

무료 gateway를 쓰는 267개 계정의 코딩·어시스턴트 에이전트(오픈 가중치 계열 모델 중심) 16주 기록이다. 상용 API 전체, 다른 모델·harness, 이후 기간으로의 전이를 주장하지 않는다. 감사 단계에서 계정·harness·모델 집중도(상위 계정 점유율 등), 주별 구성 변화, 제외 세션(`manifest.json`)을 보고하고, 결과는 harness·모델별로도 나눠 본다.

## 3. 예측 단위와 시점 계약

요청 *i*의 시작 `s_i`, 종료 `e_i`, 정답 `y_i = completion_tokens`.

- 예측 시점은 `s_i`. 특징은 그 시점에 gateway가 알 수 있는 값만 쓴다.
- 과거 요청 *j*의 결과(completion, finish_reason, tool 결과)는 `e_j < s_i`일 때만 이력에 넣는다. 병렬 subagent·동시 요청 때문에 `step` 순서만으로 판단하지 않는다.
- 현재 요청의 `completion_tokens, end_ms, ttft_ms, finish_reason`, 현재 응답이 만든 tool 호출은 특징에서 제외한다.
- 세션 연결이 사후 재구성이므로, 현재 요청의 `link_reason`처럼 현재 입력에 이미 포함된 정보로 연결된 경우만 세션 이력으로 인정한다. 이 가정은 감사 Issue에서 검증하고 안 되면 B를 그 범위로 줄인다.

## 4. 비교 모델 (LightGBM 분위수 회귀, P10/P50/P90)

| 조건 | 입력 | 질문 |
|---|---|---|
| 기준선 | 고정 분위수, 모델별 분위수, 직전 completion, 세션 누적 평균 | 단순 규칙 대비 |
| A | 현재 요청: `prompt_tokens`, `role_tokens` 분해, `cache_read_tokens`, `n_messages`, tool 정의 수, model·provider·harness·stream·`step_trigger`·`waited_for`·`transition` | 현재 요청만으로 |
| L | A + 같은 세션 직전 완료 요청의 completion 1개 | 최근 수요 lag 통제 |
| B | L + 세션 상태: step·turn 위치, 이전 k개 completion/prompt lag·EWMA·차분, 직전 finish_reason, 직전 tool 호출 수·범주·outcome·result_tokens, 세션 경과·직전 종료 후 간격 | **주 비교: B − L** |
| C (보조) | B + 세션 간 정보: 같은 계정의 이전 세션 요약, 시작 시점 활성 세션 수·직전 10분 gateway 전체 출력 | 세션 간 효과 |

주 가설 검정은 **B − L** 하나로 미리 고정하고, 나머지는 보조 비교로 보고한다(다중 비교 완화).

## 5. 분할과 평가

- 시간 분할(주 단위, 결정적): train = 1~10주(05-17~07-25), validation = 11~13주(07-26~08-15), **test = 14~16주(08-16~09-05)**. 경계를 넘는 세션은 시작 주 기준으로 두고 경계 이후 정답은 학습에서 제외한다. 계정 분리 민감도 분석을 추가로 보고한다.
- 하이퍼파라미터·조기 종료는 train 내부 시간 홀드아웃으로만 고른다. validation은 비교 보고용, test는 최종 평가 Issue에서 승인 후 1회.
- 지표: 평균·분위수별 pinball, P50 MAE, P10–P90 coverage·폭. 요청 가중과 세션(또는 계정) 가중을 함께 보고하고 첫 요청(step 0)·후속 요청, harness·모델별로 나눈다.
- 불확실성(피드백 2): 세션 단위(민감도: 계정 단위) 짝지은 bootstrap으로 B−L 등 차이의 95% CI. 같은 추출을 모든 모델에 공유하고 bootstrap 중 재학습하지 않는다. 시간 의존성 확인용으로 일 단위 블록 bootstrap을 보조로 둔다.

## 6. 탐색 분석 (피드백 2·3)

- 요청→10분 시계열 그림: 개별 요청이 `start_ms` 기준 10분 구간의 입력(prompt) 합·출력(completion) 합으로 모이는 과정을 도식과 실제 시계열로 보인다.
- 세션 내: step 간 completion 자기상관, finish_reason·tool 범주별 다음 completion 분포, 상태 특징과 직전 completion lag의 상관(정보 중복 확인).
- 세션 간: 계정 내 세션 간 상관, 동시 활성 세션 수와 요청 completion의 관계, 주별 분포 이동.

## 7. 제외 범위

다음 요청의 도착 시각·개수 예측, 메시지 의미·임베딩(원문 없음), prefix cache replay, RNN·Transformer, GPU·비용·SLO 효과 주장.

## 8. 진행 상황

작업은 Issue에서 관리한다([workflow.md](workflow.md)). 예정 순서: 데이터 감사·대표성 → 10분 시계열·EDA → 특징 생성기·누출 테스트 → 분할·기준선·평가 틀 → A/L/B/C validation 비교 → 10분 수요 뷰 → 최종 test 1회.
