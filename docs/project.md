# 프로젝트: 에이전트 세션 상태로 LLM 요청의 출력 토큰 수요 예측

- 과목: 시계열 데이터 분석 및 실습 (Group 5, "Forecasting LLM Token Demand with Conversation State")
- 최종 갱신일: 2026-11-09
- 현재 상태: **1~13주 감사·EDA·A/L/B/C 특징·고정 분할·기준선 4종과 LightGBM validation 비교·세션 bootstrap CI 구현·실행 완료. C−B·C−L은 보조 결과로 보고한다.** [모델 결과](../freeinference/outputs/models/models.md)의 주 지표 CI 상한 < 0으로 validation에서 B가 L보다 개선했다. 최종 test는 미실행이다. 현재 입력 기반 연결과 엄격 완료 조건을 적용한다. [기준선 지표·CI](../freeinference/outputs/baselines/baselines.md), [특징 산출물](../freeinference/outputs/features/features.json), 연구 결정은 [ADR 0005](adr/0005-freeinference-agentic-trace.md)를 따른다.
- 이전 연구: WildChat 현재 프롬프트 응답 길이 예측(최종 test 완료)은 태그 `wildchat-final`에 보존한다. 이 문서의 수치·정의와 섞지 않는다.

## 1. 연구 질문

> 에이전트 세션에서 LLM 요청이 시작될 때, **세션 상태**(앞선 요청·tool 호출의 진행)가 현재 요청 정보와 **최근 수요(직전 출력 lag)** 이상으로 그 요청의 completion 토큰 분포(P10/P50/P90)를 더 잘 예측하는가?

실용 출력은 요청마다의 completion 토큰 P10/P50/P90이며, 이를 요청 시작 UTC 10분 구간 [시작, 끝)으로 모은 **출력 토큰 수요 뷰**를 함께 제시한다. 실측은 구간에서 시작한 요청의 completion 합이며 빈 구간은 0이다. 요청 도착에 따라 완성되는 현재 추정(nowcast)이다. 이번 데이터는 실제 요청 시작·종료 시각이 있어 10분 입력·출력 토큰 시계열을 실측으로 만들 수 있다.

## 2. 데이터

- 출처: Hugging Face `harvardMadsys/freeinference_agentic_trace` (CC BY 4.0), 논문 *From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads*. 사용 revision은 `bb26cf72c9c20f2b05f765d88548156a74d3c253`이다.
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
- 세션 연결은 Issue #30의 1~13주 `processed/` 확인 근거를 따른다. 같은 세션에서 step은 연속이며 `link_reason`이 `tool_call_id` 또는 `assistant_message`인 후속 요청은 이전 요청의 tool call ID나 assistant 메시지가 현재 입력에 포함되어 연결됐다. 연결 대상은 같은 세션의 `step-1`이며 반드시 `e_{i-1} < s_i`를 확인한다. 확인된 718,627쌍 중 718,615쌍이 엄격 완료이고, 미완료 12행은 L·B 이력 특징 전체를 결측으로 둔다. 첫 요청·이전 step 없음·시각 없음·미확인 연결도 같은 규칙을 적용한다. 이 근거는 원본 JSONL을 확인했다는 뜻이 아니다.
- L·B의 이력 후보는 같은 공개 주·세션의 이전 step 중 `end_ms < s_i`인 요청만이다. 그중 `(end_ms, step)` 순서가 가장 최근인 요청을 L의 직전 완료 요청으로 쓴다(종료 동률이면 큰 step). `sessions.parent_session_id`로 연결된 자식 세션은 섞지 않는다.
- C는 계정 이력과 gateway 부하만 추가한다. 다른 요청의 시작 사실은 `start_j < s_i`, 완료 결과는 `e_j < s_i`일 때만 쓴다. `e_j = s_i`는 완료 이력에서 제외하고 진행 중(`start_j < s_i ≤ e_j`)에는 포함한다. 종료 결측 요청은 진행 중 집계에서 제외하며, 그때까지 시작된 제외 요청 수의 평균·최대와 원본 결측 수를 주별·전체로 기록한다. 현재 요청 자신과 이후 시작 요청은 포함하지 않는다.
- C 계정 이력은 `sessions.user_id`로 결합하며 현재 `(week, session_id)` 세션을 통째로 제외한다. 같은 계정의 다른 공개 주·train 구간 과거도 포함한다. 부모·자식 세션도 포함하며 같은 공개 주의 직접 `parent_session_id` 연결에서 온 완료 요청 비율을 진단한다. 비율의 분모·분자는 각 예측 행에서 사용한 이력 요청 수의 합(반복 노출 포함)이며 모델 입력에는 넣지 않는다. B의 같은 세션 이력 규칙은 바꾸지 않는다.
- C gateway 창은 `[s_i − 10분, s_i)` 하나로 고정한다. 공개 데이터 시작(2026-05-17 00:00 UTC) 이전에 걸치는 10분 창 특징 3개는 결측이다. 세션의 최종 길이·최종 종료와 EDA의 사후 `active_sessions`는 사용하지 않는다.
- 요청 j의 tool 결과는 다음 step 입력에 들어온다는 Issue #30의 처리 테이블 근거를 따른다. 특징에서는 j가 엄격 완료됐고 j+1의 입력이 이미 도착했으며 연결 근거가 확인될 때만 사용한다. 현재 step의 tool 호출은 제외한다. `latency_ms`는 tool 실행 시간이 아니라 다음 LLM 요청까지 간격이므로 별도 특징으로 넣지 않는다.

## 4. 비교 모델 (LightGBM 분위수 회귀, P10/P50/P90)

| 조건 | 입력 | 질문 |
|---|---|---|
| 기준선 | 고정 분위수, 모델별 분위수, 직전 completion, 세션 누적 평균 | 단순 규칙 대비 |
| A | 현재 요청: `prompt_tokens`, `role_tokens` 분해, `cache_read_tokens`, `n_messages`, tool 정의 수, model·provider·harness·stream·`step_trigger`·`waited_for`·`transition` | 현재 요청만으로 |
| L | A + 같은 세션 직전 완료 요청의 completion 1개 | 최근 수요 lag 통제 |
| B | L + 세션 상태: step·turn 위치, 이전 k개 completion/prompt lag·EWMA·차분, 직전 finish_reason, 직전 tool 호출 수·범주·outcome·result_tokens, 세션 경과·직전 종료 후 간격 | **주 비교: B − L** |
| C (보조) | B + 세션 간 정보: 현재 세션을 제외한 같은 계정 완료 이력, 진행 중 요청 수·직전 10분 시작 세션 수·완료 출력 | 세션 간 효과 |

기준선은 `freeinference.baselines.Baselines`로 구현한다. P10/P50/P90 경험 분위수는 모두 NumPy의 `method="linear"`를 쓰며, 통계는 5절의 `train_target` 마스크가 참인 정답만으로 계산한다.

| 기준선 | 구체 정의 | 대체 규칙 |
|---|---|---|
| 고정 분위수 (`fixed`) | train 전체 completion의 경험 분위수 | 없음 |
| 모델별 (`model`) | `model`별 completion 경험 분위수 | `MIN_GROUP_ROWS=100` 미만 또는 처음 보는·결측 모델은 고정 분위수 |
| 직전 completion (`previous_completion`) | `completion_lag_1 × r_q`, `r_q = Q_q(y / max(lag, 1))` | lag 결측 또는 비율 학습 표본 없음이면 모델별 → 필요시 고정 분위수 |
| 세션 누적 평균 (`session_mean`) | 같은 공개 주·세션의 이전 step 중 `end_ms < start_ms`인 completion 전체의 평균 `m`에 `r_q = Q_q(y / max(m, 1))`를 곱함 | 평균 결측 또는 비율 학습 표본 없음이면 모델별 → 필요시 고정 분위수 |

두 이력 기준선의 `r_q`는 각 이력이 있는 train 정답에서 별도로 계산한다. 평균은 관측된 completion만 평균하며 최근 3개로 제한하지 않는다. `build_history`는 `build_features`와 `history_allowed` 연결 검사를 공유한다. 미확인 연결·직전 step 부재·직전 step 미완료이면 과거 완료 행이 있어도 이력 전체를 결측으로 둔다. 예측 세 분위수는 행별 정렬하며, 기준선별 직접 대체율과 최종 고정 분위수 대체율은 산출물에 기록한다. 현재·미래 정답은 예측에 쓰지 않지만 validation의 과거 엄격 완료 응답은 이후 요청 시점에 관측되므로 이력에 쓴다.

`freeinference.features.build_features(requests, sessions, tool_calls)`는 입력 행 순서·인덱스를 유지한 A/L/B DataFrame을 반환한다. 한 공개 주씩 호출하거나 세 테이블 모두에 `week`를 넣는다. A는 `current_input_features`로 생성하며 harness는 sessions에서 결합한다. `build_cross_session_features(requests, sessions)`는 1~13주 전체를 한 번에 받아 행 순서·인덱스를 유지한 C 추가 특징 9개를 반환한다. C는 기존 B와 이 열들을 결합하며 A/L/B 값은 바꾸지 않는다. 계정별·세션별 정렬·누적합·`searchsorted`와 세션별 시작 구간 합집합을 사용해 행마다 전체를 훑지 않는다.

| 4절 항목 | 구현 특징·규칙 | 누락·제외 이유 |
|---|---|---|
| A 현재 요청 | `prompt_tokens`, `cache_read_tokens`, `n_messages`, `role_tokens_{system,user,assistant,tool,tool_definitions}`, `tool_definition_count`, `model`, `provider`, `harness`, `stream`, `step_trigger`, `waited_for`, `transition` | 없음 |
| L 직전 완료 completion | `completion_lag_1` | 없음 |
| B step·turn 위치 | `step`, `turn_index`, `turn_step`: 현재까지 관측한 user-trigger 입력의 0 기반 turn 번호, 마지막 user-trigger 이후 step 차이 | user-trigger 입력을 아직 관측하지 못했으면 turn 위치 결측 |
| B 이전 k개 completion/prompt | `completion_lag_{1,2,3}`, `prompt_lag_{1,2,3}`; k=3, 최근 완료 순서 | 이력이 부족한 lag는 결측 |
| B EWMA·차분 | `completion_ewma`, `prompt_ewma`: 최근 최대 3개를 오래된 순서로 alpha=0.5, adjust=False 계산(결측 토큰은 건너뜀); `completion_diff`, `prompt_diff`: lag1−lag2 | 두 완료 요청이 없으면 차분 결측 |
| B 직전 finish_reason | `previous_finish_reason` | 없음 |
| B 직전 tool 호출 수·범주 | `previous_tool_call_count`, `previous_tool_category_*`: EDA와 같은 고정 16범주별 호출 수 및 `other/missing` | 없음 |
| B 직전 tool outcome·result_tokens | `previous_tool_outcome_{ok,error,timeout,denied,other,missing}` 호출 수, `previous_tool_result_tokens` 합 | 결과 크기가 하나라도 결측이면 합도 결측; 관측되지 않은 결과는 tool 특징 전체 결측 |
| B 세션 경과·직전 종료 후 간격 | `session_elapsed_ms`: 현재 시작−step 0 시작, `previous_end_gap_ms`: 현재 시작−가장 최근 완료 종료 | step 0 입력이 없으면 경과 결측; tool `latency_ms`는 간격과 중복되어 제외 |
| C 계정 완료 이력 수 | `account_prev_completed_requests`: 다른 세션의 엄격 완료 요청 수; `account_prev_sessions`: 그 세션 수 | 이력 없음·계정 결측이면 수는 0 |
| C 계정 completion 요약 | `account_completion_mean`, `account_completion_last`, `account_last_end_gap_ms`: 전체 이력 평균·최근 완료 값·그 종료부터 간격 | 이력 없으면 결측; 평균에 필요한 토큰이 하나라도 결측이면 평균 결측. 종료 동률은 `(week, session_id, step)` 사전순의 마지막 요청 |
| C gateway 진행 중 | `gateway_inflight_requests`: `start_j < s_i ≤ e_j` 요청 수 | 종료 결측은 제외·진단. 세션 최종 종료 사용 안 함 |
| C gateway 10분 | `gateway_recent_sessions_10m`: 창 안에 시작 요청이 있는 서로 다른 `(week, session_id)` 수; `gateway_completion_10m`, `gateway_completed_requests_10m`: 창 안에 종료된 요청의 completion 합·수 | 시작 이전에 걸친 창은 결측, 완전한 빈 창은 0; 완료 토큰이 하나라도 결측이면 합도 결측 |

A 입력의 결측 목록·역할은 결측으로 유지하고, 명시적 빈 tool 정의 목록은 0이다. 첫 요청이나 연결 검사를 통과하지 못한 행도 A는 유지하며 **L 추가 특징과 B 추가 특징 전체를 결측**으로 둔다(step·turn·경과 포함). 정상 연결에서 이전 요청에 tool 호출이 없으면 호출 수·범주·outcome·result_tokens는 0이다. 완전한 특징 이름 목록과 주별·전체 결측률은 [생성 산출물](../freeinference/outputs/features/features.json)의 `feature_sets`, `missing_count`, `missing_rate`에 있다. 재현은 준비된 `FREEINFERENCE_DATA_DIR`와 분석 의존성이 있는 Python에서 `python -m freeinference.features`를 실행한다. 산출물에는 집계만 기록하며 요청별 특징·정답·식별자는 내보내지 않는다. 실행은 분할·학습·평가를 하지 않는다.

`freeinference.models.QuantileModels`는 `FEATURE_SETS["A"|"L"|"B"|"C"]`와 열·순서가 정확히 같은 입력만 받으며 조건·분위수마다 LightGBM 1개(총 12개)를 학습한다. LightGBM은 `4.7.0`으로 고정한다. 범주형은 `model`, `provider`, `harness`, `stream`, `step_trigger`, `waited_for`, `transition`, `previous_finish_reason`이며 선택 단계에는 `fit_target`, 최종 단계에는 `train_target`에서 본 범주 사전만 고정한다. 미관측·결측 범주는 결측으로 두고 수치 결측은 LightGBM 기본 처리에 맡긴다. 범주 사전이나 특징 열은 validation으로 확장하지 않는다.

고정 격자는 `(num_leaves, min_data_in_leaf) ∈ {(15,100), (31,300)}` × `target_transform ∈ {identity, log1p}`의 4개 후보이며 `learning_rate=0.05`다. A/L/B/C·P10/P50/P90 모두 같은 격자를 사용한다. `objective="quantile"`, `alpha ∈ {0.1,0.5,0.9}`, `seed=20261109`, `deterministic=True`, `force_col_wise=True`, `num_threads=4`를 고정한다. 최대 3000회·patience 30으로 `fit_target`만 학습하며, `holdout_target`의 **원척도 해당 q pinball** 하나로 조기 종료·설정을 선택한다. `log1p` 예측은 조기 종료·격자 비교부터 `expm1`로 복원하며, 동점이면 격자에서 먼저 나온 후보를 택한다. 선택한 설정·반복 수로 `train_target` 전체에 재학습한 뒤 validation을 예측한다. 원척도 복원 후 행별 정렬로 P10 ≤ P50 ≤ P90을 보장하고 정렬 전 교차 행 비율을 기록한다. 중요도는 조건·분위수별 gain 상위 10개 집계만 저장한다.

주 가설 검정은 **B − L** 하나로 미리 고정하고, C−B·C−L·L−A와 각 조건의 fixed·previous_completion 대비 차이는 보조 비교로 보고한다(다중 비교 완화). 실행은 `python -m freeinference.models`; [집계 JSON](../freeinference/outputs/models/models.json)과 [모델 보고서](../freeinference/outputs/models/models.md)에 전체 격자·후보별 holdout 점수·선택·환경·소스 해시·시간을 기록한다. 후보별 `best_iteration`(최적 회차)·`evaluated_rounds`(실제 평가 회차)·`hit_max_rounds`(실제 평가 회차의 상한 도달 여부), 조건·분위수별 및 전체 상한 도달 후보 수를 기록한다. 최적 회차가 상한보다 작아도 patience를 채우기 전에 상한에 도달할 수 있다. #39 커밋의 300회 산출물에서 집계값만 읽어 선택 반복 수·holdout 점수·validation 차이·CI의 변화를 자동 비교한다. #41 산출물과 A/L/B 선택 결과·모든 validation 구간 지표·B−L bootstrap CI·주 판정의 정확한 일치를 자동 대조한다. C−B·C−L은 요청 가중 mean pinball 차이·세션 가중 차이·같은 세션 bootstrap CI를 병기하는 보조 결과다. 요청 가중 CI 상한 < 0이면 해당 validation 개선, 0 포함이면 차이 미확인, 하한 > 0이면 악화로 해석하며 주 판정에 쓰지 않는다. 분할 상수·연구 질문·test 계약은 유지한다.

## 5. 분할과 평가

- 시간 분할(주 단위, 결정적): train = 1~10주(05-17~07-25), validation = 11~13주(07-26~08-15), **test = 14~16주(08-16~09-05)**. `freeinference.split`의 `TRAIN_WEEKS=(1,...,10)`, `VALIDATION_WEEKS=(11,12,13)`을 사용한다. 주차는 `loader.FIRST_WEEK=2026-05-17` 기준 UTC 7일 간격이다. 공개 데이터의 세션 키는 `(week, session_id)`이며, 같은 키의 첫 요청 `start_ms`가 속한 주로 세션 전체를 배정한다.
- 경계 규칙: train 세션이어도 `end_ms ≥ VALIDATION_START_MS=1785024000000`(2026-07-26 00:00 UTC)이거나 종료 시각이 결측이면 `train_target`에서 제외한다. `train` 행은 그대로 보존하고 특징·기준선 이력 계산은 변경하지 않는다. test 폴더·test 시각 요청은 기존 로더에서 거부한다.
- 하이퍼파라미터·조기 종료는 train 내부 시간 홀드아웃으로만 고른다. `FIT_WEEKS=(1,...,9)`, `HOLDOUT_WEEK=10`이며 `fit_target`은 `end_ms < HOLDOUT_START_MS=1784419200000`(2026-07-19 00:00 UTC)만 포함한다. 시각 결측도 제외한다. holdout 정답에도 validation 시작 경계를 적용한다. 기준선은 조정할 하이퍼파라미터가 없어 holdout을 쓰지 않고 경계 적용 train 전체로 맞춘다. validation은 비교 보고용, test는 최종 평가 Issue에서 승인 후 1회.
- 지표: q별 pinball `max(q(y−p_q), (q−1)(y−p_q))`와 세 분위수 평균, P50 MAE, P10–P90 coverage(`P10 ≤ y ≤ P90`)와 평균 폭(`P90−P10`). 요청 가중은 행마다 1, 세션 가중은 **각 평가 구간 안에서** 세션별 행 수의 역수이므로 세션마다 총 가중치 1이다. 전체·첫 요청(step 0)·후속 요청·harness·model별로 같은 지표와 행 수·세션 수를 기록한다.
- 주 판정 지표는 validation **요청 가중 mean pinball의 B−L 차이와 세션 bootstrap 95% CI**다. CI 상한 < 0이면 "validation에서 B가 L보다 개선", 0을 포함하면 "차이 미확인", 하한 > 0이면 validation 악화로 기록한다. 세션 가중은 보조로 함께 보고한다. 최종 결론은 사용자 승인 후 test 평가 Issue에서 정한다.
- 불확실성(피드백 2): `freeinference.evaluate.paired_bootstrap`은 예측기 이름 → P10/P50/P90 DataFrame 사전과 기준 예측기 이름, `cluster_columns`를 받는다. 기본 군집은 `(week, session_id)`, `BOOTSTRAP_B=2000`, `BOOTSTRAP_SEED=20261109`로 고정한다. validation 전체의 세션을 복원 추출해 **같은 추출을 모든 예측기·구간·가중에 공유**하며 재학습하지 않는다. 세션의 모든 행은 같은 횟수만큼 포함한다. 차이(예측기−기준)의 2.5·97.5 백분위(`linear`)를 95% CI로 기록한다. 일부 구간이 빈 추출은 그 구간 CI에서 제외하고 유효 반복 수를 기록한다. 모델 산출물은 L을 기준으로 B−L·C−L, B를 기준으로 C−B, A를 기준으로 L−A, fixed·previous_completion을 각각 기준으로 A/L/B/C 차이를 같은 함수·시드로 계산한다. 같은 함수와 시드로 계정 및 요청 시작 UTC 날짜의 겹치지 않는 일 블록 CI를 추가한다. 재학습 없이 train 등장·신규 계정과 validation 요청 수 상위 1·5개 계정 제외를 비교하고, 계정별 행 수 역수로 계정 가중 mean pinball을 계산한다. 세 군집의 요청 가중 CI 상한이 모두 0보다 작을 때 군집 방식에 견고하다고 기록한다. 일 블록은 21개로 불안정할 수 있다.
- `evaluate_predictions`와 `paired_bootstrap`은 동일 행 순서·인덱스의 정답/예측을 요구한다. 세션 가중은 군집 열을 바꿔도 유지되며, 군집 열만 바꾸면 계정·일 블록 추출을 재사용할 수 있다. 실데이터 실행은 `python -m freeinference.evaluate`; 집계 [JSON](../freeinference/outputs/baselines/baselines.json)과 [보고서](../freeinference/outputs/baselines/baselines.md)에 데이터 revision·코드 HEAD/소스 해시·명령·환경·소요 시간·경계 제외 수·지표·대체율·CI를 기록한다. 요청별 예측·정답·식별자는 내보내지 않으며 실데이터 실행은 저장소 검증 명령에 넣지 않는다.

10분 운영 뷰는 요청별 P50 합과 실측 합의 MAE·WAPE·평균 bias, P10/P90 합 범위의 포함률·평균 폭·폭/실측 중앙값을 전체·UTC 시간·주별로 보고한다. 실측 0 구간은 폭/실측 비율에서 제외한다. 분위수 합 범위는 완전 양의 상관을 가정한 보수적 범위이며 보정된 80% 합 구간이 아니다. P50 합도 합의 중앙값이 아니다. B−L·C−B의 MAE·WAPE 차이는 21개 UTC 일 블록 bootstrap으로 비교하며 주 가설 판정을 대신하지 않는다. [수요 뷰 결과](../freeinference/outputs/demand/demand.md).

## 6. 탐색 분석 (피드백 2·3)

- 요청→10분 시계열 그림: 개별 요청이 `start_ms` 기준 10분 구간의 입력(prompt) 합·출력(completion) 합으로 모이는 과정을 도식과 실제 시계열로 보인다.
- 세션 내: step 간 completion 자기상관, finish_reason·tool 범주별 다음 completion 분포, 상태 특징과 직전 completion lag의 상관(정보 중복 확인).
- 세션 간: 계정 내 세션 간 상관, 동시 활성 세션 수와 요청 completion의 관계, 주별 분포 이동.

10분 집계와 인접 step·동시 활성 세션·주별 분포의 결과 및 재현 명령은 [EDA 결과](../freeinference/outputs/eda/eda.md)에 있다. 상태 특징과 lag의 중복 및 계정 내 세션 간 상관은 이후 분석 범위다.

## 7. 제외 범위

다음 요청의 도착 시각·개수 예측, 메시지 의미·임베딩(원문 없음), prefix cache replay, RNN·Transformer, GPU·비용·SLO 효과 주장.

## 8. 진행 상황

작업은 Issue에서 관리한다([workflow.md](workflow.md)). A/L/B/C validation 비교와 세션·계정·UTC 일 블록 bootstrap 및 계정 구성 민감도를 1~13주에서 평가했다([모델 결과](../freeinference/outputs/models/models.md), [특징 집계](../freeinference/outputs/features/features.json), [기준선 결과](../freeinference/outputs/baselines/baselines.md)). 주 판정은 기존 세션 CI 기준 B−L을 유지한다. 일 블록은 21개이고 CI는 학습·선택 불확실성을 포함하지 않는다. 최종 test는 읽지 않았고 결론을 유보한다. 10분 수요 뷰는 validation에서 산출했다. 다음 순서: 사용자 승인 후 최종 test 1회.
