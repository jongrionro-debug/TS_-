# #19 후속: 범주 정보를 전역 B의 특징으로 추가 (validation 전용)

test 분할 파일은 읽지 않았다. 재분류 없이 기존 MiniLM 라벨을 재사용했다(유사도는 같은 모델로 재임베딩, argmax가 저장 라벨과 일치하는지 검사).

- C_feat_label: B 31개 특징 + 범주 라벨(LightGBM categorical).
- C_feat_sim: B 31개 특징 + 7개 범주 유사도(수치).
- 두 변형 모두 전역 B의 하이퍼파라미터와 분위수별 반복 수를 그대로 고정해 전체 purged train으로 재학습.
- C: category_c.md와 같은 범주별 별도 모델.

## validation 평균 pinball

| 턴 | 가중 | A | B | C | C_feat_label | C_feat_sim |
|---|---|---:|---:|---:|---:|---:|
| first_turn | row | 72.1761 | 72.1246 | 70.2425 | 70.1954 | 66.5699 |
| first_turn | user | 62.8032 | 62.4703 | 61.4877 | 61.3192 | 58.6292 |
| followup_turn | row | 51.5309 | 36.3316 | 36.3060 | 35.9713 | 35.4610 |
| followup_turn | user | 58.0626 | 42.0518 | 42.2348 | 41.5905 | 40.9417 |
| all | row | 59.3735 | 49.9285 | 49.1976 | 48.9722 | 47.2785 |
| all | user | 62.4498 | 55.8758 | 55.4524 | 55.0880 | 53.0072 |

## 사용자 bootstrap 95% CI (음수 = 앞 모델이 좋음)

| 턴 | 가중 | 차이 | 점추정 | 95% CI |
|---|---|---|---:|---|
| first_turn | row | C-B | -1.8822 | [-3.5492, -0.2217] |
| first_turn | row | C_feat_label-B | -1.9293 | [-3.1251, -0.7998] |
| first_turn | row | C_feat_sim-B | -5.5547 | [-7.5174, -3.6315] |
| first_turn | user | C-B | -0.9826 | [-1.7596, -0.1785] |
| first_turn | user | C_feat_label-B | -1.1512 | [-1.7558, -0.5221] |
| first_turn | user | C_feat_sim-B | -3.8412 | [-4.7106, -2.9215] |
| followup_turn | row | C-B | -0.0256 | [-0.6355, 0.4853] |
| followup_turn | row | C_feat_label-B | -0.3602 | [-0.6704, -0.1163] |
| followup_turn | row | C_feat_sim-B | -0.8706 | [-1.3845, -0.4797] |
| followup_turn | user | C-B | 0.1831 | [-0.3595, 0.8152] |
| followup_turn | user | C_feat_label-B | -0.4613 | [-0.8031, -0.1272] |
| followup_turn | user | C_feat_sim-B | -1.1101 | [-1.6102, -0.5830] |
| all | row | C-B | -0.7309 | [-1.5156, -0.0126] |
| all | row | C_feat_label-B | -0.9563 | [-1.4953, -0.4834] |
| all | row | C_feat_sim-B | -2.6500 | [-3.5620, -1.8474] |
| all | user | C-B | -0.4233 | [-1.0567, 0.2413] |
| all | user | C_feat_label-B | -0.7878 | [-1.2800, -0.2867] |
| all | user | C_feat_sim-B | -2.8686 | [-3.5317, -2.1675] |

- 유사도 파일 sims.jsonl은 저장소 밖(데이터 디렉터리)에 있다. 범주 분류 정확도는 train 200행 수기 라벨 대비 0.32로 낮다. validation은 이미 여러 번 본 기간이다.
