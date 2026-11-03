## Summary

Closes #13

#7의 validation 재사용·사용자 중첩을 제거한 설정으로 A(현재 길이)와 B(길이+이력)를 재검증했다. 기존 산출물은 보존하고, 같은 시간 경계에서 사용자·대화 그룹을 분리한 뒤 train 내부에서만 설정과 조기 종료 횟수를 선택한다. 고정 분위수와 두 이력 기준선을 추가하고 분위수를 정렬하지 않는다.

정제 train 81,307행·8,702사용자, validation 12,920행·1,736사용자다. 외부·내부 분할 모두 사용자/대화 그룹/원본 행 겹침 0, 엄격한 기록시각 순서를 검증했다. 내부 학습/홀드아웃은 52,674/10,524행이다.

| 구분 | 가중 | A 평균 pinball | B 평균 pinball | B−A [사용자 bootstrap 95% CI] | B−최선 기준선 [95% CI] |
|---|---|---:|---:|---|---|
| 전체 | 행 | 59.373522 | 49.928499 | −9.445023 [−10.788199, −8.033169] | −6.530100 [−8.472556, −4.175292] |
| 전체 | 사용자 | 62.449831 | 55.875767 | −6.574064 [−7.469572, −5.736406] | −7.882293 [−9.260959, −6.454912] |
| 첫 턴 | 행 | 72.176077 | 72.124630 | −0.051448 [−1.165280, 1.269711] | −6.609437 [−10.799736, −1.975877] |
| 첫 턴 | 사용자 | 62.803227 | 62.470312 | −0.332915 [−0.803256, 0.109538] | −6.941416 [−8.516298, −5.479917] |
| 후속 턴 | 행 | 51.530918 | 36.331568 | −15.199349 [−16.967032, −13.413943] | −6.481499 [−7.711493, −5.358558] |
| 후속 턴 | 사용자 | 58.062599 | 42.051782 | −16.010817 [−18.370114, −14.050791] | −8.107047 [−10.113104, −6.253458] |

최선 기준선은 train 내부에서 턴·가중 방식별로 고정했다. 첫 턴은 fixed, 나머지는 cumulative_mean이며 validation에서도 같은 기준선이 최저 손실이었다. 전체 행 가중 평균 pinball은 fixed 63.160449, previous_ratio 57.097554, cumulative_mean 56.458599다. B의 손실 감소는 후속 턴에서 관측됐으며 첫 턴 B−A CI는 0을 포함한다.

## Acceptance Criteria

- [x] 사용자 겹침 0 — 외부·내부 분할의 실행 assertion과 [분할 로그](revalidation.log).
- [x] 5모델×3턴 구분×2가중, 전체 지표 — [결과 표](revalidation.md) 및 [기계 판독 결과](revalidation.json).
- [x] B−A·B−최선 기준선 사용자 bootstrap CI — 같은 사용자 추출을 모든 모델·턴·가중 방식에 적용, seed 42·1,000회. 개별 기준선 대비 CI도 제공.
- [x] 재현 명령·결과·PR 요약 — 이 문서와 [실행 절차](../README.md), 결과 보고서에 저장.

## Changes

- `wildchat/revalidate.py`: 허용된 두 준비 분할과 해당 원본 행의 사용자 메타데이터만 연결, 보수적 사용자·대화 그룹 제거, 내부 선택 후 전체 train 재학습, 기준선·두 가중 평가·사용자 bootstrap·로그 자동 저장.
- `wildchat/tests/test_revalidate.py`: 양쪽 사용자·그룹 제거, 시간 경계 동률, 원래 분위수 교차 평가, 비율 적합, 명시적 사용자 재표집과 CI 일치, 빈 구분, 누적 평균 인과성, 허용 파일 접근, validation 정답의 학습/조기 종료 유입 차단.
- `docs/prd.md`, `wildchat/README.md`: #13 평가 규칙과 재현 절차. 기존 #7 결과와 원본 데이터는 보존.

## Verification

```sh
/workspace/venvs/wildchat/bin/python wildchat/revalidate.py --prepared /workspace/data/wildchat-1m/derived/issue5-shard0 --source /workspace/data/wildchat-1m/data/train-00000-of-00014.parquet --output wildchat/results --seed 42 --inner-fraction 0.8 --max-rounds 500 --early-stopping-rounds 40 --threads 4 --bootstrap-replicates 1000
/workspace/venvs/wildchat/bin/python -m unittest wildchat.tests.test_revalidate wildchat.tests.test_metrics
python3 scripts/verify.py
```

같은 설정으로 실제 실험을 다시 실행해 분할·선택·30개 조합 지표·모든 bootstrap CI의 일치를 확인했다(수치 비교 허용오차 1e−10). [두 실행 대조 기록](revalidation_verification.json)의 최대 차이는 1.43e−14 미만이었다. LightGBM의 내부 손실 합산은 병렬 부동소수점 연산으로 끝자리 차이가 생길 수 있다. 수치 환경의 합성/지표 검증 10개 통과, 필수 저장소 검증 통과. 금지된 평가 분할을 만드는 기존 통합 검증은 실행하지 않고 위의 허용 분할 전용 검증을 실행했다.

## Risks

- 사용자 제거로 평가 모집단과 학습 규모가 바뀐다. 기존 −9.55와 비교해 각 보정 요인의 효과를 인과적으로 분해하지 않는다. 기존에 본 validation의 재검증이므로 완전히 새 기간의 검증도 아니다.
- hashed_ip는 사용자 프록시이며 기록시각은 실제 요청 도착시각이 아니다. CI는 고정 예측에 조건부이며 학습·선택 불확실성과 시간 블록 상관을 포함하지 않는다.
- B 원시 분위수 교차 20/12,920행(0.1548%), 음수 예측 17행이다. 음수만 0으로 자르고 교차를 그대로 평가한다. 행 가중 B coverage는 0.772833, 평균 폭은 459.959183이다.
- 사용자 연결 원본은 메타데이터 열만 읽고 파일명·행 수·선택 행 시각을 확인했다. 전체 원본 바이트 해시는 준비 manifest의 출처 주장으로 남긴다.

## Decisions needed (사람)

없음.

## Agent metadata

- worker: noxxcode (codex exec)
- difficulty / model: high / gpt-6.1-sol
- base_commit: 74266fe7b4c042b65d7c4993f755d86c83b4f64d
- branch: agent/issue-13-7-train
- 커밋·push·PR 게시는 작업 래퍼가 수행한다.
