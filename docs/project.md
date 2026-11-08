# 프로젝트: 에이전트 세션 상태로 LLM 요청의 출력 토큰 수요 예측

- 과목: 시계열 데이터 분석 및 실습 (Group 5, "Forecasting LLM Token Demand with Conversation State")
- 최종 갱신일: 2026-11-09
- 현재 상태: **1~13주 감사·EDA·A/L/B 특징과 고정 분할·기준선 4종 validation 평가 구현.** 현재 입력 기반 연결과 엄격 완료 조건을 적용한다. [기준선 지표·세션 bootstrap CI](../freeinference/outputs/baselines/baselines.md), [특징 산출물](../freeinference/outputs/features/features.json), 연구 결정은 [ADR 0005](adr/0005-freeinference-agentic-trace.md)를 따른다. LightGBM 비교와 최종 test는 미실행이다.
- 이전 연구: WildChat 현재 프롬프트 응답 길이 예측(최종 test 완료)은 태그 `wildchat-final`에 보존한다. 이 문서의 수치·정의와 섞지 않는다.

## 1. 연구 질문

> 에이전트 세션에서 LLM 요청이 시작될 때, **세션 상태**(앞선 요청·tool 호출의 진행)가 현재 요청 정보와 **최근 수요(직전 출력 lag)** 이상으로 그 요청의 completion 토큰 분포(P10/P50/P90)를 더 잘 예측하는가?

실용 출력은 요청마다의 completion 토큰 P10/P50/P90이며, 이를 시작시각 기준 10분 구간으로 모은 **출력 토큰 수요 뷰**를 함께 제시한다. 이번 데이터는 실제 요청 시작·종료 시각이 있어 10분 입력·출력 토큰 시계열을 실측으로 만들 수 있다.

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
- 이력 후보는 같은 공개 주·세션의 이전 step 중 `end_ms < s_i`인 요청만이다. 그중 `(end_ms, step)` 순서가 가장 최근인 요청을 L의 직전 완료 요청으로 쓴다(종료 동률이면 큰 step). `sessions.parent_session_id`로 연결된 자식 세션은 섞지 않는다.
- 요청 j의 tool 결과는 다음 step 입력에 들어온다는 Issue #30의 처리 테이블 근거를 따른다. 특징에서는 j가 엄격 완료됐고 j+1의 입력이 이미 도착했으며 연결 근거가 확인될 때만 사용한다. 현재 step의 tool 호출은 제외한다. `latency_ms`는 tool 실행 시간이 아니라 다음 LLM 요청까지 간격이므로 별도 특징으로 넣지 않는다.

## 4. 비교 모델 (LightGBM 분위수 회귀, P10/P50/P90)

| 조건 | 입력 | 질문 |
|---|---|---|
| 기준선 | 고정 분위수, 모델별 분위수, 직전 completion, 세션 누적 평균 | 단순 규칙 대비 |
| A | 현재 요청: `prompt_tokens`, `role_tokens` 분해, `cache_read_tokens`, `n_messages`, tool 정의 수, model·provider·harness·stream·`step_trigger`·`waited_for`·`transition` | 현재 요청만으로 |
| L | A + 같은 세션 직전 완료 요청의 completion 1개 | 최근 수요 lag 통제 |
| B | L + 세션 상태: step·turn 위치, 이전 k개 completion/prompt lag·EWMA·차분, 직전 finish_reason, 직전 tool 호출 수·범주·outcome·result_tokens, 세션 경과·직전 종료 후 간격 | **주 비교: B − L** |
| C (보조) | B + 세션 간 정보: 같은 계정의 이전 세션 요약, 시작 시점 활성 세션 수·직전 10분 gateway 전체 출력 | 세션 간 효과 |

기준선은 `freeinference.baselines.Baselines`로 구현한다. P10/P50/P90 경험 분위수는 모두 NumPy의 `method="linear"`를 쓰며, 통계는 5절의 `train_target` 마스크가 참인 정답만으로 계산한다.

| 기준선 | 구체 정의 | 대체 규칙 |
|---|---|---|
| 고정 분위수 (`fixed`) | train 전체 completion의 경험 분위수 | 없음 |
| 모델별 (`model`) | `model`별 completion 경험 분위수 | `MIN_GROUP_ROWS=100` 미만 또는 처음 보는·결측 모델은 고정 분위수 |
| 직전 completion (`previous_completion`) | `completion_lag_1 × r_q`, `r_q = Q_q(y / max(lag, 1))` | lag 결측 또는 비율 학습 표본 없음이면 모델별 → 필요시 고정 분위수 |
| 세션 누적 평균 (`session_mean`) | 같은 공개 주·세션의 이전 step 중 `end_ms < start_ms`인 completion 전체의 평균 `m`에 `r_q = Q_q(y / max(m, 1))`를 곱함 | 평균 결측 또는 비율 학습 표본 없음이면 모델별 → 필요시 고정 분위수 |

두 이력 기준선의 `r_q`는 각 이력이 있는 train 정답에서 별도로 계산한다. 평균은 관측된 completion만 평균하며 최근 3개로 제한하지 않는다. `build_history`는 `build_features`와 `history_allowed` 연결 검사를 공유한다. 미확인 연결·직전 step 부재·직전 step 미완료이면 과거 완료 행이 있어도 이력 전체를 결측으로 둔다. 예측 세 분위수는 행별 정렬하며, 기준선별 직접 대체율과 최종 고정 분위수 대체율은 산출물에 기록한다. 현재·미래 정답은 예측에 쓰지 않지만 validation의 과거 엄격 완료 응답은 이후 요청 시점에 관측되므로 이력에 쓴다.

`freeinference.features.build_features(requests, sessions, tool_calls)`는 입력 행 순서·인덱스를 유지한 A/L/B DataFrame을 반환한다. 한 공개 주씩 호출하거나 세 테이블 모두에 `week`를 넣는다. A는 `current_input_features`로 생성하며 harness는 sessions에서 결합한다. C는 구현하지 않는다.

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

A 입력의 결측 목록·역할은 결측으로 유지하고, 명시적 빈 tool 정의 목록은 0이다. 첫 요청이나 연결 검사를 통과하지 못한 행은 A만 유지하며 **L 추가 특징과 B 추가 특징 전체를 결측**으로 둔다(step·turn·경과 포함). 정상 연결에서 이전 요청에 tool 호출이 없으면 호출 수·범주·outcome·result_tokens는 0이다. 완전한 특징 이름 목록과 주별·전체 결측률은 [생성 산출물](../freeinference/outputs/features/features.json)의 `feature_sets`, `missing_count`, `missing_rate`에 있다. 재현은 준비된 `FREEINFERENCE_DATA_DIR`와 분석 의존성이 있는 Python에서 `python -m freeinference.features`를 실행한다. 산출물에는 집계만 기록하며 요청별 특징·정답·식별자는 내보내지 않는다. 실행은 분할·학습·평가를 하지 않는다.

주 가설 검정은 **B − L** 하나로 미리 고정하고, 나머지는 보조 비교로 보고한다(다중 비교 완화).

## 5. 분할과 평가

- 시간 분할(주 단위, 결정적): train = 1~10주(05-17~07-25), validation = 11~13주(07-26~08-15), **test = 14~16주(08-16~09-05)**. `freeinference.split`의 `TRAIN_WEEKS=(1,...,10)`, `VALIDATION_WEEKS=(11,12,13)`을 사용한다. 주차는 `loader.FIRST_WEEK=2026-05-17` 기준 UTC 7일 간격이다. 공개 데이터의 세션 키는 `(week, session_id)`이며, 같은 키의 첫 요청 `start_ms`가 속한 주로 세션 전체를 배정한다.
- 경계 규칙: train 세션이어도 `end_ms ≥ VALIDATION_START_MS=1785024000000`(2026-07-26 00:00 UTC)이거나 종료 시각이 결측이면 `train_target`에서 제외한다. `train` 행은 그대로 보존하고 특징·기준선 이력 계산은 변경하지 않는다. test 폴더·test 시각 요청은 기존 로더에서 거부한다.
- 하이퍼파라미터·조기 종료는 train 내부 시간 홀드아웃으로만 고른다. `FIT_WEEKS=(1,...,9)`, `HOLDOUT_WEEK=10`이며 `fit_target`은 `end_ms < HOLDOUT_START_MS=1784419200000`(2026-07-19 00:00 UTC)만 포함한다. 시각 결측도 제외한다. holdout 정답에도 validation 시작 경계를 적용한다. 기준선은 조정할 하이퍼파라미터가 없어 holdout을 쓰지 않고 경계 적용 train 전체로 맞춘다. validation은 비교 보고용, test는 최종 평가 Issue에서 승인 후 1회.
- 지표: q별 pinball `max(q(y−p_q), (q−1)(y−p_q))`와 세 분위수 평균, P50 MAE, P10–P90 coverage(`P10 ≤ y ≤ P90`)와 평균 폭(`P90−P10`). 요청 가중은 행마다 1, 세션 가중은 **각 평가 구간 안에서** 세션별 행 수의 역수이므로 세션마다 총 가중치 1이다. 전체·첫 요청(step 0)·후속 요청·harness·model별로 같은 지표와 행 수·세션 수를 기록한다.
- 불확실성(피드백 2): `freeinference.evaluate.paired_bootstrap`은 예측기 이름 → P10/P50/P90 DataFrame 사전과 기준 예측기 이름, `cluster_columns`를 받는다. 기본 군집은 `(week, session_id)`, `BOOTSTRAP_B=2000`, `BOOTSTRAP_SEED=20261109`로 고정한다. validation 전체의 세션을 복원 추출해 **같은 추출을 모든 예측기·구간·가중에 공유**하며 재학습하지 않는다. 세션의 모든 행은 같은 횟수만큼 포함한다. 차이(예측기−기준)의 2.5·97.5 백분위(`linear`)를 95% CI로 기록한다. 일부 구간이 빈 추출은 그 구간 CI에서 제외하고 유효 반복 수를 기록한다. 현재 산출물의 기준은 `fixed`다. B−L은 이후 주 비교이며 계정 군집·일 단위 블록 bootstrap과 계정 분리 민감도는 이후 작업이다.
- `evaluate_predictions`와 `paired_bootstrap`은 동일 행 순서·인덱스의 정답/예측을 요구한다. 세션 가중은 군집 열을 바꿔도 유지되며, 군집 열만 바꾸면 계정·일 블록 추출을 재사용할 수 있다. 실데이터 실행은 `python -m freeinference.evaluate`; 집계 [JSON](../freeinference/outputs/baselines/baselines.json)과 [보고서](../freeinference/outputs/baselines/baselines.md)에 데이터 revision·코드 HEAD/소스 해시·명령·환경·소요 시간·경계 제외 수·지표·대체율·CI를 기록한다. 요청별 예측·정답·식별자는 내보내지 않으며 실데이터 실행은 저장소 검증 명령에 넣지 않는다.

## 6. 탐색 분석 (피드백 2·3)

- 요청→10분 시계열 그림: 개별 요청이 `start_ms` 기준 10분 구간의 입력(prompt) 합·출력(completion) 합으로 모이는 과정을 도식과 실제 시계열로 보인다.
- 세션 내: step 간 completion 자기상관, finish_reason·tool 범주별 다음 completion 분포, 상태 특징과 직전 completion lag의 상관(정보 중복 확인).
- 세션 간: 계정 내 세션 간 상관, 동시 활성 세션 수와 요청 completion의 관계, 주별 분포 이동.

10분 집계와 인접 step·동시 활성 세션·주별 분포의 결과 및 재현 명령은 [EDA 결과](../freeinference/outputs/eda/eda.md)에 있다. 상태 특징과 lag의 중복 및 계정 내 세션 간 상관은 이후 분석 범위다.

## 7. 제외 범위

다음 요청의 도착 시각·개수 예측, 메시지 의미·임베딩(원문 없음), prefix cache replay, RNN·Transformer, GPU·비용·SLO 효과 주장.

## 8. 진행 상황

작업은 Issue에서 관리한다([workflow.md](workflow.md)). 고정 분할·기준선 4종 validation 지표·세션 bootstrap 틀은 구현 및 실데이터 실행을 마쳤다([결과](../freeinference/outputs/baselines/baselines.md)). 다음 순서: 같은 holdout·평가 함수를 사용하는 A/L/B LightGBM validation 비교와 B−L CI → C 보조 비교 → 계정·일 블록 bootstrap 및 계정 분리 민감도 → 10분 수요 뷰 → 사용자 승인 후 최종 test 1회.
