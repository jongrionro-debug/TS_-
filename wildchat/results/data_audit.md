# WildChat 데이터 적합성 감사

## 출처와 실행 범위

- 출처: [AllenAI WildChat-1M 데이터 카드](https://huggingface.co/datasets/allenai/WildChat-1M), Zhao 등, [WildChat 논문](https://arxiv.org/abs/2405.01470). 데이터 카드의 라이선스: [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/). 원 출처와 라이선스를 표시한다.
- 지정 데이터 리비전: `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. 리비전은 사용자가 제공한 출처 정보이며 파일 자체에서 검증할 수 없다.
- 입력 파일 1개, 처리 행 59,857개; 파일명: `train-00000-of-00014.parquet`.
- 선택: 입력 파일을 이름순으로 모두 순회. 무작위 추출 아님; 시드 없음. 일부 샤드만 입력했다면 전체 데이터 결과가 아니다.
- 이번 입력은 14개 샤드 중 첫 샤드 하나이며 전체 데이터 결과가 아니다.
- 파일은 배치로 읽고 중복 집계는 임시 SQLite에 SHA-256 해시만 저장한다. 해시 충돌 가능성은 이론적으로 존재한다.
- 실행 버전: datasets 5.1.0, pyarrow 26.0.0, tiktoken 0.14.0; 토크나이저 o200k_base.
- 발견한 서로 다른 Parquet 스키마 수: 1. 행 단위 결측 표는 첫 파일의 스키마 기준이다.

## 필드와 결측

행은 대화 기록 하나이며 `conversation`은 메시지 목록이다. 데이터 카드는 `conversation_hash`를 대화 내용 해시로 설명하며 같은 내용을 가진 대화에서 중복될 수 있어 고유 키가 아니라고 명시한다. `model`, `turn`, `timestamp`, `language`, moderation, `toxic`, `redacted`와 지역·헤더 필드는 행 메타데이터다. 메시지는 `role`, `content`, `timestamp`, `turn_identifier`, 언어·지역·안전성 및 헤더를 갖는다. 결측은 null만 세며 빈 문자열은 별도다.

| 행 필드 | null 행 수 |
|---|---:|
| conversation | 0 |
| conversation_hash | 0 |
| country | 95 |
| detoxify_moderation | 0 |
| hashed_ip | 0 |
| header | 0 |
| language | 0 |
| model | 0 |
| openai_moderation | 0 |
| redacted | 0 |
| state | 10,012 |
| timestamp | 0 |
| toxic | 0 |
| turn | 0 |

메시지 객체 344,836개, null 메시지 0개.

| 메시지 역할 | 메시지 수 |
|---|---:|
| assistant | 172,418 |
| user | 172,418 |

| 메시지 필드 | null 메시지 수 |
|---|---:|
| content | 0 |
| role | 0 |
| timestamp | 172,418 |
| turn_identifier | 0 |
| language | 0 |
| country | 172,737 |
| state | 205,869 |
| hashed_ip | 172,418 |
| header | 172,418 |
| redacted | 0 |
| toxic | 0 |

## 중복·대화 구조

- 비어 있지 않은 conversation_hash: 고유 59,363, 반복 그룹 115, 반복 그룹의 행 609, 추가 사본 494. 동일 hash가 같은 실제 사용자 또는 독립 세션임을 보장하지 않는다.
- 모든 필드가 같은 행: 고유 59,857, 반복 그룹 0, 반복 그룹의 행 0, 추가 사본 0.
- 역할·가시 content 전체 목록이 같은 대화: 고유 59,363, 반복 그룹 115, 반복 그룹의 행 609, 추가 사본 494; 메타데이터는 무시.
- 첫 메시지가 비어 있지 않은 user일 때 첫 user 역할·텍스트가 정확히 같은 prefix: 고유 51,653, 반복 그룹 3,301, 반복 그룹의 행 11,035, 추가 사본 7,734. 이는 공유 prefix의 하한으로, 다른 시작 역할·부분 문자열·정규화·의미상 유사는 다루지 않는다.
- 행의 turn이 메시지 수와 일치 0행, user 수와 일치 59,857행, assistant 수와 일치 59,857행. 데이터 카드는 turn을 user와 assistant의 한 차례 상호작용 횟수로 정의한다. 관측 비교가 모든 예외의 생성 과정을 증명하지는 않는다.

## 유효 쌍과 제외

각 user 메시지를 후보로 삼고 바로 다음 메시지가 assistant이며 양쪽 content가 null도 공백만도 아닐 때 유효하다. 제외는 아래 순서의 첫 사유 하나로만 집계한다. 빈 문자열 판별 외의 텍스트 정규화는 하지 않는다. 행 간 중복은 제거하지 않으므로 같은 쌍의 반복이 포함된다.

- 유효 user→assistant 쌍 170,920개; user 후보 172,418개.
- 유효 쌍의 현재 user 텍스트 입력 토큰 합 18,082,289, assistant 텍스트 출력 토큰 합 48,708,661. 각각 가시 텍스트만 o200k_base로 별도 계수한 재구성값이며 제공자 사용량이 아니다.

| 제외 사유 (우선순위순) | 후보 수 |
|---|---:|
| user content 결측 | 0 |
| user content 빈 문자열/공백 | 1,498 |
| 바로 다음 assistant 없음 | 0 |
| assistant content 결측 | 0 |
| assistant content 빈 문자열/공백 | 0 |

## 시각과 후속 실험의 한계

- 역할별 timestamp 비결측: user 0개, assistant 172,418개.
- 메시지 timestamp 비결측 172,418개 중 행 timestamp와 동일 59,857개. 데이터 카드는 행 timestamp를 마지막 턴의 UTC 시각, assistant 메시지 timestamp를 백엔드가 전체 응답을 수신한 시각이라고 설명한다. 사용자 요청 시각은 문서화되지 않았다. 메시지 timestamp의 정밀도·누락 원인 및 실제 요청 도착과의 관계는 확정할 수 없다.
- 기록 시각을 각 user 요청 도착시각으로 가정하거나 턴 간격을 만들지 않는다. 중복·공유 prefix로 연결된 행을 같은 대화 그룹으로 묶고 그룹 기록시각 순으로 분할하는 방안은 전체 데이터에서 교차 그룹 누출과 시각 경계 처리를 검증한 뒤 확정한다. 결측 시각과 경계에 걸친 그룹의 처리도 후속 결정이 필요하다.
- 집계는 관측 파일에서만 성립한다. 가시 텍스트의 삭제·마스킹, 선택 편향 및 토크나이저와 실제 API의 차이를 고려해야 한다. 학습·성능 평가는 수행하지 않았다.
