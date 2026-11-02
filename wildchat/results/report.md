# WildChat 프롬프트 대 이력 GBDT 분위수 실험 (#7)

- 기록시각(UTC): 2026-11-02T15:43:28.943590+00:00; 실행 22.31초; seed 42.
- **test 미평가**: 사용자/오케스트레이터 지시로 최종 시험 평가를 보류함; test.jsonl은 읽지 않음. 최종 test 1회 평가 시점 승인이 필요하다.
- 아래 성능은 튜닝·조기 종료에 사용한 validation 결과다. 독립 시험 성능이나 일반화 입증으로 해석하지 않는다.

## 표본과 분할

- 실제 전처리 경로: `/workspace/data/wildchat-1m/derived/issue5-shard0`. manifest의 /tmp 경로는 과거 생성 위치다.
- WildChat-1M revision `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. #5의 준비된 표본을 그대로 사용했다. 추가 추출·행 제거는 없다.
- 전체 14개 샤드 중 첫 샤드 59,857개 원본 대화 기록 행의 결정적 부분 자료이며 전체 자료를 대표하지 않는다.
- 이 이슈의 실제 실행은 운영자의 자원·범위 결정으로 고정된 issue5-shard0를 사용한다.
- 동일 hash·가시 대화 전체·첫 user 텍스트의 정확 일치를 추이적으로 묶은 #5 그룹을 유지한다. 그룹 최초 기록시각의 70%/85% 경계, 같은 시각은 뒤 구간, 경계 걸침 그룹 전체 제외 규칙을 재사용한다.
- 경계(UTC): 2023-04-27T13:14:56+00:00 / 2023-05-01T01:30:58+00:00.
- 실제 읽은 train·validation의 그룹/원본 행 교집합은 0이고 기록시각 범위는 겹치지 않는다. test의 수·범위는 manifest 메타데이터만 인용하며 파일 검증도 수행하지 않는다.

| 분할 | 그룹 | 대화 기록 행 | 쌍 표본 | 첫 user 턴 | 기록시각 최소~최대(UTC) |
|---|---:|---:|---:|---:|---|
| train | 36,160 | 39,689 | 115,500 | 39,497 | 2023-04-09T00:02:53+00:00 ~ 2023-04-27T13:12:41+00:00 |
| validation | 7,755 | 8,329 | 22,968 | 8,300 | 2023-04-27T13:14:56+00:00 ~ 2023-05-01T01:30:03+00:00 |
| test | 7,786 | 8,647 | 23,222 | 미열람 | 2023-05-01T01:30:58+00:00 ~ 2023-05-04T07:49:52+00:00 |

A·B·단순 기준은 validation의 모든 동일 행을 같은 순서로 평가한다. 첫 턴과 이력 부족 행도 유지하며, 불완전한 이전 user 턴은 lag 위치를 차지한다.

## 입력·출력 분포와 관계

분위수는 #6과 같은 경험적 CDF 역함수 ceil(n×q) 방식이다. 상관은 입력/출력 쌍 단위 기술 통계이며 인과관계가 아니다. Spearman 동률에는 평균 순위를 사용한다. 상수 열의 상관은 JSON null로 기록한다.

| 분할·항목 | 수 | 평균 | 표준편차(모집단) | 최소 | P10 | P50 | P90 | P99 | 최대 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train input_tokens | 115,500.0000 | 103.9246 | 393.8128 | 1.0000 | 5.0000 | 15.0000 | 172.0000 | 1,818.0000 | 8,081.0000 |
| train output_tokens | 115,500.0000 | 290.3449 | 265.6481 | 1.0000 | 45.0000 | 229.0000 | 603.0000 | 1,257.0000 | 4,044.0000 |
| validation input_tokens | 22,968.0000 | 119.3074 | 424.6734 | 1.0000 | 6.0000 | 16.0000 | 197.0000 | 2,148.0000 | 7,780.0000 |
| validation output_tokens | 22,968.0000 | 298.5360 | 284.7729 | 1.0000 | 43.0000 | 233.0000 | 623.0000 | 1,366.0000 | 3,615.0000 |

| 분할 | Pearson(원척도) | Spearman | Pearson(log1p 입력, log1p 출력) |
|---|---:|---:|---:|
| train | 0.154797 | 0.241361 | 0.221961 |
| validation | 0.199147 | 0.219165 | 0.206386 |

## 특징 및 누출 방지

- A (3개): `input_tokens`, `input_chars`, `input_utf8_bytes`.
- B (31개): `input_tokens`, `input_chars`, `input_utf8_bytes`, `previous_user_turns`, `lag_1_input_tokens`, `lag_1_input_chars`, `lag_1_input_utf8_bytes`, `lag_1_output_tokens`, `lag_1_output_chars`, `lag_1_output_utf8_bytes`, `lag_1_turn_missing`, `lag_1_input_missing`, `lag_1_output_missing`, `lag_2_input_tokens`, `lag_2_input_chars`, `lag_2_input_utf8_bytes`, `lag_2_output_tokens`, `lag_2_output_chars`, `lag_2_output_utf8_bytes`, `lag_2_turn_missing`, `lag_2_input_missing`, `lag_2_output_missing`, `lag_3_input_tokens`, `lag_3_input_chars`, `lag_3_input_utf8_bytes`, `lag_3_output_tokens`, `lag_3_output_chars`, `lag_3_output_utf8_bytes`, `lag_3_turn_missing`, `lag_3_input_missing`, `lag_3_output_missing`.
- 간격 lag에 사용한 슬롯: []. 제출시각 의미가 검증되고 train에서 유효 관측이 있는 슬롯이 없어 간격 수치와 상수인 간격 결측 지표를 제외했다. assistant 시각이나 행 시각으로 대체하지 않는다.
- 현재 프롬프트 길이는 제출 직후 알려진 숫자다. 이전 인접 assistant 응답은 현재 프롬프트 전에 관측됐다고 가정한다. 원문·임베딩·현재 응답·미래 턴·전체 대화 집계·메타데이터는 특징에 넣지 않는다.
- 결측 lag는 NaN과 기존 결측 지표로 유지한다. 정규화·대체값 통계는 학습하지 않으며, LightGBM의 bin 경계는 train Dataset에서만 만든다. validation Dataset은 train을 reference로 사용한다.

## 학습·선택과 분위수 교차

- A·B 각각 P10/P50/P90, 같은 seed 42, 같은 탐색 공간 [{"num_leaves": 15, "min_data_in_leaf": 100}, {"num_leaves": 31, "min_data_in_leaf": 100}, {"num_leaves": 31, "min_data_in_leaf": 300}].
- 모델별 3설정 × 3분위수 = 9회. 각 최대 500회 boosting, learning_rate 0.05, validation 분위수 손실 40회 무개선 조기 종료(min_delta=0), 스레드 4.
- 각 분위수의 best iteration으로 validation을 예측한 뒤 음수를 0으로 자르고 행별 오름차순 정렬한다. 정렬 후 P10≤P50≤P90을 #6 metrics.evaluate에 전달한다. 교차율은 보정 전 원예측 기준이다. 정렬은 개별 분위수의 의미와 P50도 바꿀 수 있으며 보정 후 지표로 선택한다.
- validation의 3분위수 평균 pinball 최소 설정을 A·B 각각 선택한다. 동률은 탐색 순서의 첫 설정을 선택한다. validation을 학습 데이터에 합쳐 재학습하지 않는다. 선택된 모델의 시험 예측·평가와 모델 바이너리 저장은 수행하지 않는다.
- 단순 기준은 train 출력만으로 계산한 고정 P10/P50/P90: {'p10': 45.0, 'p50': 229.0, 'p90': 603.0}.

| 모델 | 후보 | 설정 | best iteration P10/P50/P90 | 검증 평균 pinball | 보정 전 교차율 | 선택 |
|---|---:|---|---|---:|---:|---|
| A | 0 | {'num_leaves': 15, 'min_data_in_leaf': 100} | 205/303/239 | 56.774009 | 0.000000 |  |
| A | 1 | {'num_leaves': 31, 'min_data_in_leaf': 100} | 131/187/171 | 56.734935 | 0.000000 | 예 |
| A | 2 | {'num_leaves': 31, 'min_data_in_leaf': 300} | 79/114/138 | 56.753888 | 0.000000 |  |
| B | 0 | {'num_leaves': 15, 'min_data_in_leaf': 100} | 500/467/250 | 47.354751 | 0.000958 |  |
| B | 1 | {'num_leaves': 31, 'min_data_in_leaf': 100} | 200/284/262 | 47.276701 | 0.000958 |  |
| B | 2 | {'num_leaves': 31, 'min_data_in_leaf': 300} | 190/487/201 | 47.183900 | 0.001132 | 예 |

## 동일 validation 표본의 성능

MAE와 log 오차는 P50 기준이며 log 오차는 mean(|log1p(y)−log1p(P50)|)다. pinball은 max(q(y−p), (q−1)(y−p)), 평균은 3분위수 산술평균이다. 커버리지는 양 끝을 포함하고 평균 폭은 P90−P10이다.

| 모델 | MAE | log 오차 | pinball P10 | P50 | P90 | 평균 pinball | P10~P90 커버리지 | 평균 폭 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 단순 기준 | 184.761799 | 0.801674 | 27.603775 | 92.380900 | 63.814303 | 61.266326 | 0.789577 | 558.000000 |
| A | 177.232888 | 0.770363 | 27.128279 | 88.616444 | 54.460080 | 56.734935 | 0.786529 | 529.399325 |
| B | 145.456297 | 0.616468 | 24.742934 | 72.728149 | 44.080618 | 47.183900 | 0.790230 | 441.838898 |
| B−A | -31.776590 | -0.153894 | -2.385345 | -15.888295 | -10.379463 | -9.551034 | 0.003701 | -87.560427 |

B−A 평균 pinball 차이는 -9.551034 (-16.83%)이다. 이 표본의 검증 분할에서 이력 추가 후 손실이 감소했다.
MAE·log·pinball 차이는 음수일수록 오차 감소다. 커버리지는 명목 0.8과 비교하고 폭은 커버리지와 함께 해석한다. 반복·연결 표본이 남아 있으므로 행을 독립 관측으로 보거나 통계적 유의성을 주장하지 않는다.

## 재현 정보와 한계

- 실행 패키지: python=3.13.5, lightgbm=4.7.0, numpy=2.5.4, scipy=1.18.1.
- 전처리 패키지: {"python": "3.13.5", "datasets": "5.1.0", "pyarrow": "26.0.0", "tiktoken": "0.14.0"}.
- 전체 LightGBM 파라미터·각 분위수 best iteration·후보별 지표·파일 SHA-256은 [experiment.json](experiment.json)에 기록한다. best iteration의 트리만 예측에 사용한다.
- 재현 명령(실행에 사용한 Python 경로; 다른 환경에서는 의존성이 설치된 Python으로 대체):

```sh
/workspace/venvs/wildchat/bin/python3 wildchat/train.py --prepared /workspace/data/wildchat-1m/derived/issue5-shard0 --output wildchat/results --seed 42 --max-rounds 500 --early-stopping-rounds 40 --threads 4
python3 -m unittest discover -s wildchat/tests
/workspace/venvs/wildchat/bin/python -m unittest discover -s wildchat/tests
python3 scripts/verify.py
```

- o200k_base로 user/assistant 가시 텍스트를 독립 계수한 재구성 토큰이다. 시스템·역할·툴·숨은 토큰, 제공자별 tokenizer와 서비스의 실제 사용량을 재현하지 않는다.
- 기록시각은 행의 마지막 턴 시각이며 실제 개별 요청 도착시각 순서를 보장하지 않는다. user 제출시각이 없어 실제 턴 간격의 효과는 평가하지 못한다. 다른 샤드와 연결되는 중복도 확인하지 못했다.
- 튜닝에 재사용한 검증 결과, 첫 샤드의 시기·분포·길이 편향, 분할 내 중복의 반복 가중 효과가 있다. 전체 데이터·독립 시험의 효과, 실제 GPU/비용 개선, 인과 효과로 확장하지 않는다.
- 미결정: 최종 test 1회 평가 시점 승인. 이 구현은 승인 전 test를 읽을 실행 경로를 제공하지 않는다.
