# #23 H1: 같은 대화 순차 특징 (validation 전용)

B, B+H1, C_emb_PCA(k=32), C_emb_PCA+H1을 같은 정제 표본·분할에서 비교했다.
각 조건에 동일한 3개 후보와 train 내부 홀드아웃 선택·조기 종료 규칙을 적용했다. validation은 보고에만 썼다.
#19의 B 설정을 모든 조건에 고정했던 PCA 실험과 달리 이번에는 네 조건을 각각 내부 튜닝한다. Mac 탐색 코드의 재실행이 아니다.

- H1: 이전 유효 출력의 EWMA(α=0.3), 누적 평균·모집단 표준편차, 최근 두 출력 차이·원래 user 턴 간격당 기울기, 직전 유효 출력/입력 비율·비율 EWMA, 현재/직전 유효 입력 배율, 관측 수(9개).
- 같은 source 대화에서 특징을 먼저 기록한 뒤 현재 응답을 상태에 넣는다. 유효하지 않은 쌍은 제외하고 원래 턴 간격은 유지한다. 사용자·중복 그룹 사이에 상태를 공유하지 않는다.
- 이력이 없으면 count=0 외에는 NaN, 관측 1개면 std=0, 2개 미만이면 차이·기울기는 NaN. 입력 0의 비율은 NaN이고 비율 EWMA에서만 건너뛴다. EWMA는 첫 유효 값으로 초기화한다.
- 운영자 지시에 따라 기존 MiniLM 캐시를 재사용했다. 기존 SHA256·384차원 형식·키 유일성·train/validation 전체 키의 정확한 일치를 검사했다. 원문은 읽지 않았다.
- PCA(32)는 선택 시 내부 fit만, 최종 val 예측 시 전체 정제 train만으로 적합했다. 음수 예측만 0으로 자르고 분위수는 정렬하지 않았다.
- JSONL 접근은 train/validation 두 경로로 실행 중 제한했다. test 분할 접근·최종 평가 실행은 하지 않았다.

## 표본과 내부 선택

| 구간 | 행 | 사용자 | 첫 턴 |
|---|---:|---:|---:|
| outer/earlier | 81307 | 8702 | 27887 |
| outer/later | 12920 | 1736 | 4908 |
| inner/earlier | 52674 | 6328 | 18449 |
| inner/later | 10524 | 1837 | 3932 |

- outer 분리 검사: {"user_intersection": 0, "group_intersection": 0, "source_intersection": 0, "strict_chronology": true, "missing_users": 0}
- inner 분리 검사: {"user_intersection": 0, "group_intersection": 0, "source_intersection": 0, "strict_chronology": true, "missing_users": 0}

| 조건 | num_leaves | min_data_in_leaf | P10/P50/P90 반복 |
|---|---:|---:|---|
| B | 31 | 300 | 403/244/279 |
| B+H1 | 31 | 100 | 347/191/268 |
| C_emb_PCA | 31 | 300 | 315/500/384 |
| C_emb_PCA+H1 | 31 | 300 | 312/500/474 |

## validation 평균 pinball (P10/P50/P90 평균, 낮을수록 좋음)

행 가중은 행 전체 평균, 사용자 가중은 각 턴 구분 안의 사용자별 평균 후 사용자 동일 가중이다.
첫 턴은 turn_index==0, 후속 턴은 >0이다.

| 턴 | 가중 | B | B+H1 | C_emb_PCA | C_emb_PCA+H1 |
|---|---|---:|---:|---:|---:|
| first_turn | row | 72.124630 | 72.112678 | 64.646729 | 64.260688 |
| first_turn | user | 62.470312 | 62.525095 | 56.265649 | 56.038919 |
| followup_turn | row | 36.331568 | 36.143711 | 34.835691 | 34.596462 |
| followup_turn | user | 42.051782 | 41.912082 | 40.319427 | 40.148674 |
| all | row | 49.928499 | 49.807464 | 46.160193 | 45.865195 |
| all | user | 55.875767 | 55.905105 | 51.241735 | 51.075339 |

## H1 추가 차이와 사용자 paired bootstrap 95% CI

사용자 복원추출 1,000회, seed 42. 같은 추출을 네 조건·모든 턴·가중에 공유한다. 음수는 H1 개선이다.

| 턴 | 가중 | 차이 | 점추정 | 95% CI | 유효 반복 |
|---|---|---|---:|---|---:|
| first_turn | row | B+H1-B | -0.011951 | [-0.234133, 0.216068] | 1000 |
| first_turn | row | C_emb_PCA+H1-C_emb_PCA | -0.386041 | [-0.715176, -0.028842] | 1000 |
| first_turn | user | B+H1-B | 0.054782 | [-0.044794, 0.161670] | 1000 |
| first_turn | user | C_emb_PCA+H1-C_emb_PCA | -0.226731 | [-0.462993, -0.008083] | 1000 |
| followup_turn | row | B+H1-B | -0.187857 | [-0.353550, -0.012205] | 1000 |
| followup_turn | row | C_emb_PCA+H1-C_emb_PCA | -0.239228 | [-0.378874, -0.086743] | 1000 |
| followup_turn | user | B+H1-B | -0.139701 | [-0.398637, 0.127409] | 1000 |
| followup_turn | user | C_emb_PCA+H1-C_emb_PCA | -0.170754 | [-0.431728, 0.111103] | 1000 |
| all | row | B+H1-B | -0.121035 | [-0.257891, 0.014788] | 1000 |
| all | row | C_emb_PCA+H1-C_emb_PCA | -0.294999 | [-0.447423, -0.131609] | 1000 |
| all | user | B+H1-B | 0.029338 | [-0.095114, 0.154135] | 1000 |
| all | user | C_emb_PCA+H1-C_emb_PCA | -0.166397 | [-0.352599, 0.016207] | 1000 |

## 재현·한계

```sh
export WILDCHAT_DATA_DIR=/workspace/data/wildchat-1m
OPENBLAS_NUM_THREADS=1 /workspace/venvs/wildchat/bin/python wildchat/temporal.py --prepared /workspace/data/wildchat-1m/derived/issue5-shard0 --source /workspace/data/wildchat-1m/data/train-00000-of-00014.parquet --embeddings /workspace/data/wildchat-1m/derived/issue19-categories/embeddings.npz --output wildchat/results --embedding-reference wildchat/results/category_pca.json --seed 42 --inner-fraction 0.8 --threads 4 --max-rounds 500 --early-stopping-rounds 40 --bootstrap-replicates 1000
```

설정·후보별 내부 점수·패키지 버전·입력/코드 해시·키 정합성·분리 검사는 [temporal_h1.json](temporal_h1.json), 실행 로그는 [temporal_h1.log](temporal_h1.log)에 있다.

- 첫 샤드의 고정 표본이며 validation은 여러 탐색에서 이미 사용됐다. Mac 탐색의 유의성이나 독립 미관측 성능을 확증하는 평가가 아니다.
- CI는 고정된 학습 결과·평가 표본에 조건부다. 학습·선택 불확실성, 시간 블록 상관, IP 프록시 식별 오차를 포함하지 않는다.
- 첫 턴의 H1은 count=0 외에는 결측이다. 첫 턴 점수 차이는 공유 모델의 학습·선택 변화로 생길 수 있으며 첫 턴의 과거 응답 정보를 뜻하지 않는다.
- 과거 인접 응답은 현재 프롬프트 전에 관측됐다고 가정한다. 행 기록시각은 실제 요청 도착시각이 아니다. 실제 API 토큰·GPU 효과나 전체 WildChat으로 일반화하지 않는다.
