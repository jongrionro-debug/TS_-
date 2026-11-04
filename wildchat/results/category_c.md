# #19 조건 C: 프롬프트 의미 범주별 분위수 모델 (validation 전용)

test 분할 파일은 읽지 않았다. #13의 사용자 분리·train 내부 홀드아웃 선택 규칙을 그대로 사용했다.

- C: 범주별로 B와 같은 31개 특징의 LightGBM P10/P50/P90을 범주 내부 홀드아웃에서 선택·전체 범주 train으로 재학습한다. 내부 fit<500행 또는 홀드아웃<100행이면 전역 B.
- C_fallback: train 행 수 <5000인 범주는 전역 B(임계값은 결과를 보기 전 고정).
- C_inner_select: train 내부 홀드아웃에서 범주 모델이 전역 B보다 나을 때만 범주 모델 사용.

## 전체 validation 평균 pinball

| 턴 | 가중 | 행 | 사용자 | A | B | C | C_fallback | C_inner_select |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| first_turn | row | 4,908 | 1,726 | 72.1761 | 72.1246 | 70.2425 | 70.2425 | 71.8591 |
| first_turn | user | 4,908 | 1,726 | 62.8032 | 62.4703 | 61.4877 | 61.4877 | 62.0432 |
| followup_turn | row | 8,012 | 1,093 | 51.5309 | 36.3316 | 36.3060 | 36.3060 | 36.0640 |
| followup_turn | user | 8,012 | 1,093 | 58.0626 | 42.0518 | 42.2348 | 42.2348 | 41.9128 |
| all | row | 12,920 | 1,736 | 59.3735 | 49.9285 | 49.1976 | 49.1976 | 49.6617 |
| all | user | 12,920 | 1,736 | 62.4498 | 55.8758 | 55.4524 | 55.4524 | 55.7807 |

## 차이의 사용자 bootstrap 95% CI (음수 = 앞 모델이 좋음)

| 턴 | 가중 | 차이 | 점추정 | 95% CI |
|---|---|---|---:|---|
| first_turn | row | C-B | -1.8822 | [-3.5492, -0.2217] |
| first_turn | row | C_fallback-B | -1.8822 | [-3.5492, -0.2217] |
| first_turn | row | C_inner_select-B | -0.2656 | [-0.8074, 0.2986] |
| first_turn | row | C-A | -1.9336 | [-3.9921, 0.3589] |
| first_turn | user | C-B | -0.9826 | [-1.7596, -0.1785] |
| first_turn | user | C_fallback-B | -0.9826 | [-1.7596, -0.1785] |
| first_turn | user | C_inner_select-B | -0.4271 | [-0.9516, 0.1965] |
| first_turn | user | C-A | -1.3155 | [-2.1460, -0.4859] |
| followup_turn | row | C-B | -0.0256 | [-0.6355, 0.4853] |
| followup_turn | row | C_fallback-B | -0.0256 | [-0.6355, 0.4853] |
| followup_turn | row | C_inner_select-B | -0.2675 | [-0.8974, 0.1427] |
| followup_turn | row | C-A | -15.2249 | [-16.8349, -13.7277] |
| followup_turn | user | C-B | 0.1831 | [-0.3595, 0.8152] |
| followup_turn | user | C_fallback-B | 0.1831 | [-0.3595, 0.8152] |
| followup_turn | user | C_inner_select-B | -0.1389 | [-0.5739, 0.2793] |
| followup_turn | user | C-A | -15.8278 | [-18.0306, -14.0036] |
| all | row | C-B | -0.7309 | [-1.5156, -0.0126] |
| all | row | C_fallback-B | -0.7309 | [-1.5156, -0.0126] |
| all | row | C_inner_select-B | -0.2668 | [-0.6940, 0.0872] |
| all | row | C-A | -10.1759 | [-11.5636, -8.6754] |
| all | user | C-B | -0.4233 | [-1.0567, 0.2413] |
| all | user | C_fallback-B | -0.4233 | [-1.0567, 0.2413] |
| all | user | C_inner_select-B | -0.0950 | [-0.5146, 0.3790] |
| all | user | C-A | -6.9974 | [-7.9844, -6.0404] |

## 범주별 (행 가중, all 턴)

| 범주 | train 행 | val 행 | val 사용자 | val 첫 턴 | 자체 모델 | fallback | 내부선택 | A | B | C |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| chit_chat | 10,021 | 1,401 | 531 | 471 | True | False | True | 43.9690 | 37.8290 | 35.5765 |
| coding | 12,843 | 1,383 | 400 | 459 | True | False | True | 63.7562 | 54.5205 | 55.1527 |
| creative_writing | 9,578 | 1,926 | 477 | 1,109 | True | False | False | 82.2513 | 73.3370 | 69.3239 |
| editing_summarization | 17,802 | 3,006 | 828 | 1,050 | True | False | False | 63.4895 | 51.1101 | 51.2396 |
| knowledge_qa | 13,547 | 2,373 | 682 | 865 | True | False | True | 46.6240 | 38.4855 | 37.9944 |
| math_reasoning | 11,350 | 1,886 | 506 | 566 | True | False | False | 50.6428 | 40.6783 | 40.6841 |
| translation | 6,166 | 945 | 336 | 388 | True | False | False | 65.5174 | 56.8746 | 58.2855 |

- 모든 범주가 train 5,000행 이상이라 C_fallback은 C와 동일하다.

## 재현과 한계

```sh
/workspace/venvs/wildchat/bin/python wildchat/category.py --prepared /workspace/data/wildchat-1m/derived/issue5-shard0 --source /workspace/data/wildchat-1m/data/train-00000-of-00014.parquet --labels /workspace/data/wildchat-1m/derived/issue19-categories/labels.jsonl --output wildchat/results --seed 42 --inner-fraction 0.8 --max-rounds 500 --early-stopping-rounds 40 --threads 4 --bootstrap-replicates 1000 --min-fit-rows 500 --min-holdout-rows 100 --fallback-train-rows 5000
```

- 범주는 로컬 다국어 임베딩 zero-shot 분류(wildchat/classify.py)이며 train 200행 수기 라벨(category_handlabels.json) 대비 정확도 0.32로 낮다. 오분류가 많이 섞인다.
- validation은 #7·#13에서 이미 여러 번 본 기간이므로 C의 이점은 낙관적일 수 있다. test 재평가는 하지 않았다.
- 첫 샤드 고정 표본(train+validation 전체 행 분류, 표본 추출 없음) 결과다.
