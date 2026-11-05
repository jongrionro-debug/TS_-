# 시간 구조 가설 VAL 실험 요약

상태: 실제 VAL 학습·평가 완료. test.jsonl 미접근·test 정답 미사용. Seed 42.

기준선: #21 B (현재 길이+대화 내 길이 이력 31개). 동일 사용자 분리·시간 구간, 음수만 0 처리·분위수 무정렬.
하이퍼파라미터 31 leaves / min leaf 300, P10/P50/P90 403/244/279회 고정. 모든 H는 동일 기준선에 하나씩 추가하며 재튜닝하지 않았다.

B 기준선 재현: 과거 49.928499, 이번 49.928499, 재현 차이 +0.000000. 모든 가설 Δ는 이번 기준선에 대한 값이다.
C_emb_PCA k=32의 과거 46.062757은 임베딩 cache 부재로 재현하지 않았다. 사용자 지시에 따라 새 임베딩 생성 없이 H4_prompt_sequence도 건너뛰었다.

정제 train 81,307행/8,702사용자, VAL 12,920행/1,736사용자. 사용자·그룹·source 교집합 0.

- H1_sequence_lengths: 동일 대화의 이전 유효 응답 EWMA(α=.3), 누적 평균·표준편차, 마지막 두 응답 차이·턴당 기울기, 이전 출력/입력 비율·EWMA·현재 입력 배율·관측 수
- H2_reply_cadence: 현재 턴 이전 assistant 완료 시각끼리의 마지막 간격·누적 span·log1p와 결측 표시. 현재 요청까지의 대기시간이 아님
- H3_calendar_drift: 이전 assistant 완료 시각의 epoch day·월·UTC hour/weekday sin/cos와 hour×log1p(현재 입력). 첫 턴 결측

평균 P10/P50/P90 pinball (낮을수록 좋음). 사용자 가중은 각 구분 안의 사용자별 평균을 동일 가중.

| 턴 | 가중 | baseline | H1_sequence_lengths | H2_reply_cadence | H3_calendar_drift |
|---|---|---:|---:|---:|---:|
| all | row | 49.928499 | 49.780214 | 49.919513 | 50.063224 |
| all | user | 55.875767 | 55.785333 | 55.906758 | 56.003362 |
| first_turn | row | 72.124630 | 72.140488 | 72.173216 | 72.217982 |
| first_turn | user | 62.470312 | 62.459386 | 62.507725 | 62.550961 |
| followup_turn | row | 36.331568 | 36.082732 | 36.287314 | 36.491637 |
| followup_turn | user | 42.051782 | 41.732693 | 42.046458 | 42.187079 |

H−동일 기준선 차이 / 사용자 paired bootstrap 95% CI (1,000회·PCG64 seed 42·공유 추출·percentile 선형 보간):

| H | 턴 | 가중 | 차이 | 95% CI |
|---|---|---|---:|---|
| H1_sequence_lengths-baseline | all | row | -0.148285 | [-0.246487, -0.044299] |
| H2_reply_cadence-baseline | all | row | -0.008986 | [-0.072455, 0.057281] |
| H3_calendar_drift-baseline | all | row | 0.134725 | [0.073409, 0.197476] |
| H1_sequence_lengths-baseline | all | user | -0.090434 | [-0.187576, 0.005691] |
| H2_reply_cadence-baseline | all | user | 0.030991 | [-0.023507, 0.083077] |
| H3_calendar_drift-baseline | all | user | 0.127594 | [0.073918, 0.190938] |
| H1_sequence_lengths-baseline | first_turn | row | 0.015858 | [-0.045355, 0.072159] |
| H2_reply_cadence-baseline | first_turn | row | 0.048586 | [0.000563, 0.095765] |
| H3_calendar_drift-baseline | first_turn | row | 0.093353 | [0.019466, 0.168602] |
| H1_sequence_lengths-baseline | first_turn | user | -0.010927 | [-0.059754, 0.039689] |
| H2_reply_cadence-baseline | first_turn | user | 0.037412 | [-0.011280, 0.087927] |
| H3_calendar_drift-baseline | first_turn | user | 0.080649 | [0.025540, 0.137601] |
| H1_sequence_lengths-baseline | followup_turn | row | -0.248836 | [-0.397267, -0.090600] |
| H2_reply_cadence-baseline | followup_turn | row | -0.044254 | [-0.140044, 0.054649] |
| H3_calendar_drift-baseline | followup_turn | row | 0.160068 | [0.057492, 0.269941] |
| H1_sequence_lengths-baseline | followup_turn | user | -0.319090 | [-0.562586, -0.082333] |
| H2_reply_cadence-baseline | followup_turn | user | -0.005324 | [-0.108428, 0.087649] |
| H3_calendar_drift-baseline | followup_turn | user | 0.135297 | [0.024533, 0.251800] |

권장: **H1_sequence_lengths**. 전체 행 가중 CI가 0 아래인 후보 중 가장 낮은 점추정을 탐색적 권장으로 삼고, 없으면 기준선을 유지한다.
H1은 전체 행 가중과 후속 턴의 행/사용자 가중에서 작은 개선을 보였다. 전체 사용자 가중과 첫 턴의 H1 CI는 0을 포함한다. H2 전체 차이 CI는 0을 포함하고 H3는 손실이 증가했다.
시간 구조를 추가하려면 H1의 대화 내 수준·추세·출력/입력 비율 요약을 우선 권장한다. 묶음 특징 실험이므로 EWMA 하나의 효과로 분해할 수 없다. C_emb_PCA에 H1을 추가하는 효과는 아직 검증하지 않았다.
반복 사용한 VAL의 탐색 결과이며 다중비교 보정·새로운 일반화 증거는 아니다. 사용자 bootstrap은 고정 모델에 조건부이고 시간 블록·학습 불확실성을 반영하지 않는다.
입력 시각은 이전 assistant 완료의 하한이고 현재 요청 제출 시각이 아니다. 첫 턴에 시각이 없으므로 달력 특징을 제공하지 않는다. 모델 버전·대화 마지막 시각을 입력에 사용하지 않았다.

실행 시간(초): {"loading_features": 5.1502056659956, "baseline": 5.055709083972033, "H1_sequence_lengths": 5.434047458984423, "H2_reply_cadence": 4.916760041989619, "H3_calendar_drift": 5.459481166006299, "evaluation_bootstrap": 0.36037858299096115, "total": 26.379951374983648}

실행 환경: {"numpy": "2.5.4", "lightgbm": "4.7.0", "pyarrow": "26.0.1", "scipy": "1.18.1"}

파일: `wildchat/temporal_hypotheses.py`, `wildchat/temporal_causal.py`, `wildchat/temporal_baseline.json`, `wildchat/tests/test_temporal_hypotheses.py`, `wildchat/results/temporal_hypotheses.json`, `wildchat/results/temporal_hypotheses.md`, `wildchat/results/temporal_hypotheses_summary.md`.

검증·문제: 지정 /tmp/venv-temporal/bin/python으로 재실행. 설치 없음. 전체 합성 검사 50개 중 49개 PASS·1개 tokenizer 의존성 skip, 새 안전성 검사 8개 PASS. 구조·diff 형식 PASS. 공통 verify는 기존 TraceLab 링크 8건으로 FAIL. 기존 파일 18개 해시 유지.

기존 docs/project.md·wildchat/README.md는 사용자의 미커밋 변경 보존을 위해 수정하지 않았다. 이 VAL 전용 후속 실험의 정의·재현·결과는 요청된 두 결과 MD와 JSON에 둔다.

Notion: 기존 WildChat 실험 페이지의 쓰기는 직전 시도에서 승인 정책(approval policy: never)으로 차단됐다. 이번 실제 결과 요약도 미저장이며 완료된 로컬 실험과 구분한다. 미저장 인계안: “2026-11-06 VAL에서 B 재현, H1 대화 내 요약에 소폭 이득, H2 추가 개선 미확인·H3 손실 증가. H4는 cache 부재로 skip. 정본 wildchat/results/temporal_hypotheses.md 및 summary.md, 미커밋 로컬 산출물. 전체 사용자 가중 H1 CI는 0 포함. test 접근·Git 게시 없음.”
