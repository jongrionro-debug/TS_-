# WildChat 요청 표본 준비

## 범위와 재현

- 출처: [AllenAI WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), [WildChat 논문](https://arxiv.org/abs/2405.01470), [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/).
- 지정 리비전: `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. 제공된 출처 정보이며 로컬 파일만으로 원격 리비전을 검증하지 않는다.
- 입력: `train-00000-of-00014.parquet` (59,857행, SHA-256 `abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c`).
- 이번 실행은 지정된 첫 샤드(14개 중 1개) 전체이며 전체 데이터 결과가 아니다. 미입력 샤드와의 연결은 알 수 없다.
- 파일 이름순·행 순서대로 모두 처리하며 추가 표본 추출은 없다. seed=42; 무작위 연산에 사용하지 않는다.
- 배치 크기 128; 두 번 순회하고 SQLite에는 원문 대신 해시·집계·행 번호만 저장한다.
- 실행 버전: python 3.13.5, datasets 5.1.0, pyarrow 26.0.0, tiktoken 0.14.0; tokenizer `o200k_base`.
- 실행 명령: `/workspace/venvs/wildchat/bin/python3 wildchat/prepare.py --input /workspace/data/wildchat-1m/data/train-00000-of-00014.parquet --output /tmp/wildchat-prepared-issue5-final --seed 42`
- 출력 JSONL과 manifest는 저장소 밖에 둔다. 표본은 원문 없이 숫자 특징·정답·관리 메타데이터만 포함한다.

## 예측 시점과 필드

현재 user 프롬프트 제출 직후를 예측 시점으로 삼는다. `prompt_features.input_tokens`는 현재 user 가시 텍스트의 입력 토큰이며 유일한 예측 정답은 `target.output_tokens`(바로 다음 assistant의 출력 토큰)이다. 각 메시지를 독립적으로 o200k_base로 계수하고 역할 구분자·숨은 시스템 메시지·제공자 직렬화 비용은 포함하지 않는다. chars는 Python Unicode 코드 포인트 수이며 utf8_bytes는 UTF-8 바이트 길이다.

모델 입력은 `prompt_features`와 `history_features`만 허용한다. 현재 응답·미래 메시지·대화 전체 통계·행 메타데이터는 특징에 없다. `metadata`의 split/group/행 번호/마지막 기록시각과 `diagnostics`는 모델 입력에서 제외한다. 마지막 기록시각은 미래 정보일 수 있으며 분할 관리에만 쓴다.

lag 1부터 3까지 이전 user 턴을 역순으로 참조한다. 입력·출력 토큰, 문자·바이트 길이와 결측 지표를 각각 저장한다. 이전의 인접 assistant 응답은 대화 순서상 현재 프롬프트 전에 이미 관측되었다고 가정한다. 제외된 과거 쌍도 턴 위치를 차지하며, 없는 값은 null로 유지하여 더 오래된 턴으로 당겨 채우지 않는다. 첫 턴에는 모든 lag가 결측이다.

간격 lag k는 이전 k번째 user와 그 직전 user 사이의 제출 간격이다. 현재 user나 assistant 완료시각을 사용하지 않는다. 실제 user 제출시각 검증 옵션: False. 검증되지 않았거나 시각이 없거나 역전되면 수치값을 제외(null)하고 결측 지표와 사유를 남긴다. 이번 입력에서 파싱 가능한 user 시각 0개, 결측·비정상 user 시각 172,418개다. assistant 완료시각이나 행의 마지막 턴 시각으로 대체하지 않는다.

## 유효 표본과 제외

- 대화 기록 행 59,857; user 후보 172,418; 구조상 유효 쌍 170,920; 분할 후 유효 표본 161,690.
- 바로 인접한 user→assistant이며 양쪽이 비어 있지 않은 문자열인 쌍만 유지한다. 아래 첫 사유 하나로만 후보를 제외한다. 행 간 중복은 삭제하지 않으며 같은 분할 안의 반복으로 유지한다.

| 쌍 제외 사유(우선순위순) | 후보 수 |
|---|---:|
| user_content_missing | 0 |
| user_content_empty_or_nontext | 1,498 |
| no_adjacent_assistant | 0 |
| assistant_content_missing | 0 |
| assistant_content_empty_or_nontext | 0 |

## 중복과 그룹 구성

동일 conversation_hash, 역할·가시 텍스트 목록 전체의 정확한 일치, 첫 user 텍스트의 정확한 일치 중 하나라도 연결되면 union-find로 추이적 연결 성분을 만든다. 선행 system 메시지가 있어도 첫 user를 찾는다. 공통 첫 프롬프트가 다른 후속 응답으로 갈라져도 한 그룹이다. 텍스트 정규화·부분 문자열·의미상 유사성은 다루지 않는다. 해시 충돌의 이론적 가능성과 공통 상용구에 의한 과도한 결합이 남는다.

| 연결 키 | 고유 키 | 반복 키 | 반복 키 소속 행 | 추가 사본 |
|---|---:|---:|---:|---:|
| declared_hash | 59,363 | 115 | 609 | 494 |
| visible_conversation | 59,363 | 115 | 609 | 494 |
| first_user_prefix | 51,650 | 3,301 | 11,035 | 7,734 |

최종 연결 성분 52,123개, 여러 행인 그룹 3,301개, 최대 그룹 492행.
공백만 있는 첫 user 텍스트 3행은 prefix 연결 키에서 제외한다. #4 감사의 prefix 판정은 공백 문자열을 포함했으므로 이 키의 고유 수가 다를 수 있다.

## 시간순 분할과 경계 제외

그룹 기록시각 범위는 소속 행의 마지막 턴 기록시각 최솟값~최댓값이다. 유효 표본이 있는 그룹 중 시각 결측·비정상 행이 하나라도 있는 그룹은 통째로 제외한다. 나머지 그룹을 최초 기록시각과 그룹 ID로 정렬하고 floor(0.70×그룹 수), floor(0.85×그룹 수) 위치의 최초 시각을 경계로 삼는다. 목표 70%/15%/15%는 제거 전 그룹 수 기준이다. 경계와 같은 시각은 모두 뒤 구간에 넣고, 그룹의 시각 범위가 경계를 넘으면 그룹 전체를 제외한다. 따라서 실제 그룹·행·표본 비율은 목표와 다를 수 있다.

- 경계 산정 대상 51,905그룹.
- train/validation 경계: 2023-04-27T13:14:56+00:00; validation/test 경계: 2023-05-01T01:30:58+00:00.
- 구간: train < 첫 경계, 첫 경계 ≤ validation < 둘째 경계, 둘째 경계 ≤ test. 그룹을 나눠 살리지 않는다.
- 이는 기록시각 순서이며 실제 개별 요청 도착시각 순서를 보장하지 않는다. 같은 대화의 과거 턴이 행 기록시각보다 앞설 수 있다.

| 분할 | 그룹 수 | 대화 기록 행 수 | 표본 수 | 기록시각 최소(UTC) | 기록시각 최대(UTC) |
|---|---:|---:|---:|---|---|
| train | 36,160 | 39,689 | 115,500 | 2023-04-09T00:02:53+00:00 | 2023-04-27T13:12:41+00:00 |
| validation | 7,755 | 8,329 | 22,968 | 2023-04-27T13:14:56+00:00 | 2023-05-01T01:30:03+00:00 |
| test | 7,786 | 8,647 | 23,222 | 2023-05-01T01:30:58+00:00 | 2023-05-04T07:49:52+00:00 |

| 그룹 제외 사유 | 그룹 수 | 대화 기록 행 수 | 구조상 유효 표본 수 |
|---|---:|---:|---:|
| crosses_time_boundary | 204 | 2,974 | 9,230 |
| no_valid_samples | 218 | 218 | 0 |
| missing_or_invalid_record_time | 0 | 0 | 0 |

## 간격 lag 제외와 검증

아래는 분할 후 표본의 각 lag 슬롯 집계다. 첫 턴/이력 부족과 미검증 시각을 구분하며, 모든 null에는 결측 지표가 있다.

| lag 슬롯: 제외 사유 | 수 |
|---|---:|
| lag_1: no_preceding_user_turn | 28,730 |
| lag_1: no_previous_turn | 56,410 |
| lag_1: user_submission_timestamp_missing_or_invalid | 76,550 |
| lag_2: no_preceding_user_turn | 19,009 |
| lag_2: no_previous_turn | 85,140 |
| lag_2: user_submission_timestamp_missing_or_invalid | 57,541 |
| lag_3: no_preceding_user_turn | 13,627 |
| lag_3: no_previous_turn | 104,149 |
| lag_3: user_submission_timestamp_missing_or_invalid | 43,914 |

- 수치 간격이 있는 lag 슬롯: 0.
- 분할 후 입력 토큰 합 17,460,314; 출력 정답 토큰 합 47,172,550.
- 실행 중 그룹별 단일 분할과 train < validation < test의 기록시각 범위 비중첩을 확인했다. 합성 단위 테스트는 추이적 그룹 연결·교집합·경계 제외·현재 응답/미래 변경에 대한 현재 특징 불변성·lag 순서와 결측을 검증한다.
- MAE·log 오차·pinball·커버리지와 모델 학습은 후속 평가 이슈 범위다. 분할 안에 중복을 유지하므로 후속 평가에서 반복 표본의 가중 효과를 고려해야 한다.
