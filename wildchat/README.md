# WildChat 응답 토큰 예측 — 조사부터 최종 평가까지

2026-11-03 기준 #4~#7 조사·전처리·탐색 실험, #13 사용자 분리 재검증, #15 최종 test 1회 평가를 완료했다. 현재 정의는 [PRD](../docs/prd.md), 최종 결과는 [test_final.md](results/test_final.md)와 [test_final.json](results/test_final.json)이다. 아래 명령은 단계별 실행 방법과 과거 실행 이력을 구분해 읽는다.

현재 비교는 고정 분위수·직전 응답 비율·대화 누적 평균·A·B의 5개이며, train 내부에서만 튜닝하고 사용자 분리·분위수 무정렬·사용자 bootstrap을 적용한다. #7의 validation 튜닝·3모델·분위수 정렬은 과거 탐색 규칙이다. **최종 test는 이미 한 번 사용했으므로 `finalize.py freeze/evaluate`를 재실행하지 않는다.**

정제 train 81,307행, validation 12,920행, 최종 test 11,942행·1,770사용자다. 최종 test 전체 행 가중 평균 pinball은 A 55.849263, B 45.455475, B−A −10.393788(사용자 bootstrap 95% CI [−12.732540, −8.279974])이다. 첫 턴 행 가중 차이의 CI는 0을 포함한다. 첫 샤드의 가시 텍스트 길이 예측 결과이며 실제 API 사용량이나 전체 14개 샤드의 성능을 뜻하지 않는다.

## 데이터 경로 설정

데이터 위치는 컴퓨터마다 다르므로 명령에서는 `$WILDCHAT_DATA_DIR`를 쓴다. 저장소에는 경로를 고정하지 않고 각 컴퓨터의 셸 설정에만 둔다.

```sh
# box(VM)
export WILDCHAT_DATA_DIR=/workspace/data/wildchat-1m
# 로컬 Mac (~/.zshrc)
export WILDCHAT_DATA_DIR=~/data/wildchat-1m
```

결과 파일(`results/*.json`, `results/*.md`)에 남은 `/workspace/...` 경로는 box에서 실행한 기록이므로 고치지 않는다.

## 데이터 조사 (#4)

[AllenAI WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), Zhao 등 [WildChat 논문](https://arxiv.org/abs/2405.01470). 데이터 카드의 라이선스는 [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/)이다. 출처 리비전은 `7d6490e462285cf85d91eabea0f9a954fbddcd1f`이며 데이터는 이 저장소에 포함하지 않는다.

이번 조사는 14개 샤드 중 `train-00000-of-00014.parquet` 하나를 그대로 순회했다. 무작위 표본이 아니며 시드는 없다. 첫 샤드는 59,857행이고 전체 데이터의 결과가 아니다. 후속 #5~#15도 이 샤드의 고정 표본만 사용했으며 전체 샤드 실행은 별도 범위 결정이 필요하다. 조사 CLI는 파일을 이름순으로 읽고 `--max-rows N`을 지정하면 그 순서의 앞 N행만 추가로 선택한다. 이 역시 무작위 표본이 아니다.

의존성은 [requirements.txt](requirements.txt)에 기록한다. `o200k_base` 인코딩 파일이 로컬 캐시에 없으면 최초 사용 시 다운로드가 필요하다. 네트워크가 없는 환경에서는 미리 캐시가 준비되어야 한다. 승인된 환경에서 실행한 명령은 다음과 같다.

```sh
/workspace/venvs/wildchat/bin/python wildchat/inspect.py --input "$WILDCHAT_DATA_DIR"/data/train-00000-of-00014.parquet --output wildchat/results/data_audit.md
python3 scripts/verify.py
```

다른 환경에서는 의존성이 설치된 Python을 사용하고 `--input`에 Parquet 파일 또는 디렉터리를 지정한다. 디렉터리는 하위의 모든 `.parquet` 파일을 이름순으로 읽는다. 배치와 임시 SQLite를 사용하므로 전체 원문을 메모리에 올리지 않으며, 임시 저장에는 원문의 SHA-256 해시만 쓴다. 결과는 [data_audit.md](results/data_audit.md)에 집계만 기록한다. 행 중복과 공유 prefix 판정의 정확한 기준과 한계도 보고서를 따른다.

유효 쌍은 바로 인접한 비어 있지 않은 user와 assistant 텍스트다. `o200k_base`로 각 텍스트를 따로 계수한 값은 API 실제 사용량이 아니다. `inspect.py`의 조사 단계는 학습·성능 평가를 포함하지 않는다. 저장소 밖의 데이터 및 토크나이저 캐시는 별도로 준비해야 한다. `/workspace/...`는 box에서 사용한 경로이며 로컬 Mac에 존재한다고 가정하지 않는다.

## 요청 표본 전처리 (#5)

[prepare.py](prepare.py)는 지정한 Parquet 파일 또는 디렉터리의 파일 전체를 두 번 배치 순회한다. 첫 순회에서 대화 hash·가시 내용 전체·첫 user 텍스트의 중복 관계를 union-find로 연결하고, 임시 SQLite에 해시·기록시각·집계만 저장한다. 둘째 순회에서 분할에 남은 대화의 토큰과 lag를 계산한다. 전체 원문을 메모리에 올리지 않으며 추가 추출은 없다. `--seed 42`는 재현 설정으로 기록하며 무작위 연산에는 사용하지 않는다.

의존성이 준비된 Python 환경에서 다음과 같이 실행한다. 출력 디렉터리는 저장소 밖의 비어 있는 디렉터리여야 하며 기존 산출물을 덮어쓰지 않는다. `--input`에 원본만 있는 디렉터리를 지정하면 이름순으로 모든 `.parquet` 파일을 처리한다.

```sh
python3 wildchat/prepare.py --input /path/to/wildchat --output /path/to/wildchat-prepared --seed 42
python3 -m unittest discover -s wildchat/tests
python3 scripts/verify.py
```

이 저장소의 실행 환경에서는 `/workspace/venvs/wildchat/bin/python`을 사용했다. 지정된 첫 샤드 전체(59,857행)를 실행한 결과와 정확한 명령·버전은 [preparation.md](results/preparation.md)에 있다. 전체 14개 샤드 결과가 아니며 다른 샤드에 존재하는 연결 관계는 이번 분할에서 확인할 수 없다. 새 전체 입력으로 실행할 때는 그룹과 경계를 처음부터 다시 만든다.

출력은 `train.jsonl`, `validation.jsonl`, `test.jsonl`, 그룹별 처리 결과 `groups.jsonl`, 재현 집계 `manifest.json`, 집계 보고서 `preparation.md`다. 원문은 내보내지 않으며 생성 표본과 manifest는 저장소에 커밋하지 않는다. 집계 보고서만 `wildchat/results/preparation.md`로 복사한다.

표본의 모델 입력은 `prompt_features`(현재 입력 토큰·문자·UTF-8 바이트 수)와 `history_features`(이전 user 턴의 입력·출력 토큰과 길이 lag, 결측 지표)다. 유일한 정답은 `target.output_tokens`다. `metadata`의 그룹·분할·원본 행 번호·메시지 위치·행 마지막 기록시각과 `diagnostics`의 간격 제외 사유는 모델 입력에 넣지 않는다. 현재 응답·미래 메시지·전체 대화에서 계산한 값도 특징에 사용하지 않는다. 이전 인접 assistant 응답은 현재 프롬프트 전에 관측됐다고 가정한다.

기본 `--lags 3`은 이전 user 턴 3개를 최신순으로 참조한다. 불완전한 과거 쌍도 턴 위치를 차지하며 알 수 없는 값은 null과 결측 지표로 남긴다. 턴 간격은 기본적으로 사용하지 않는다. `--verified-user-timestamps`는 user 메시지 시각이 **실제 프롬프트 제출시각임을 별도로 확인한 입력에만** 허용한다. 이 샤드의 user 시각은 모두 결측이므로 옵션을 켜도 간격 수치가 생성되지 않는다. assistant 완료시각이나 행 시각으로 대체하지 않는다.

분할은 시각이 유효하고 유효 쌍이 있는 그룹을 최초 기록시각 순으로 정렬한 뒤 그룹 수의 70%/85% 위치를 경계로 삼는다. 경계에 걸친 그룹은 통째로 제외하고 같은 시각은 뒤 구간에 둔다. 따라서 제거 후 비율은 70%/15%/15%와 다를 수 있다. 기록시각 순 분할이며 실제 요청 도착시각 순서의 보장은 아니다. 후속 #6/#7/#13/#15의 지표는 각 저장된 보고서에 있다.

테스트는 합성 대화만 사용한다. 표준 라이브러리 Python에서는 순서·결측·불변성·그룹 분리 테스트를 실행하고 실제 tokenizer/Parquet CLI 통합 테스트 1개는 건너뛴다. 위 WildChat 가상환경에서는 통합 테스트까지 모두 실행한다.

## 과거 탐색 실험: 프롬프트 대 이력 분위수 비교 (#7)

[train.py](train.py)는 “길이+대화이력 B가 길이만 A보다 출력 토큰을 더 잘 예측하는가”만 검증한다. 비교 대상은 고정 분위수 기준선·LightGBM A·LightGBM B 세 개다. A는 현재 프롬프트의 입력 토큰·문자·UTF-8 바이트 수 3개, B는 여기에 이전 턴 수·1~3턴 입력/출력 길이 lag·결측 지표 28개를 더한 총 31개를 사용한다. 원문·프롬프트 유형·주제·언어·임베딩 등 의미 특징은 사용하지 않는다. 결측 lag는 NaN으로 유지하고 첫 턴도 같은 평가 행에 포함한다. 간격 lag는 #5에서 user 제출시각 의미를 확인했고 train에 유효한 관측이 있는 슬롯만 사용한다. 이번 첫 샤드에는 해당 슬롯이 없다.

기본 실행은 지정된 prepared 디렉터리의 train·validation 전체를 사용한다. 이 이슈에서는 운영자 자원·범위 결정에 따라 `/workspace/data/wildchat-1m/derived/issue5-shard0` 고정 표본을 사용하며 전체 14개 샤드를 추가로 읽지 않는다. manifest의 과거 `/tmp` 경로를 따라가지 않는다.

```sh
python3 wildchat/train.py --prepared /path/to/wildchat-prepared --output wildchat/results --seed 42
python3 -m unittest discover -s wildchat/tests
python3 scripts/verify.py
```

당시 의존성이 준비된 box에서 사용한 명령은 다음과 같다. 로컬의 의존성 설치 여부는 별도로 확인한다.

```sh
PATH=/workspace/venvs/wildchat/bin:$PATH python3 wildchat/train.py --prepared "$WILDCHAT_DATA_DIR"/derived/issue5-shard0 --output wildchat/results --seed 42
/workspace/venvs/wildchat/bin/python -m unittest discover -s wildchat/tests
```

A·B에 동일한 3개 설정 × P10/P50/P90, seed, 최대 500회 boosting과 validation 손실 40회 무개선 조기 종료를 적용한다. `--max-rounds`, `--early-stopping-rounds`, `--threads`(기본 4)를 조정하면 두 모델에 공통 적용하며 결과에 기록한다. 음수 예측을 0으로 자르고 행별 정렬한 후, validation 평균 pinball로 각 모델의 설정을 선택한다. 단순 기준은 train 출력의 경험적 P10/P50/P90이며 #6의 동일 지표 함수를 사용한다. bin 경계는 train에서만 만들고 validation을 합쳐 재학습하지 않는다.

2026-11-03 사용자 결정에 따라 전체·첫 턴(`turn_index==0`)·후속 턴(`>0`)별 평균 pinball(주 지표), P50 MAE, P10~P90 커버리지, 평균 구간 폭을 보고한다. 각 지표와 B−A 차이에 #5 대화 그룹 부트스트랩 95% CI를 붙인다. `--bootstrap-replicates` 기본 1,000회, `--seed` 기본 42이며 NumPy PCG64를 사용한다. validation 전체의 G개 그룹에서 G개를 복원 추출하고 각 그룹의 모든 행을 유지한다. 같은 추출을 세 모델·세 턴 구분에 공통 적용하며, 각 반복의 표본별 평균과 대응 B−A 차이에서 2.5%/97.5% percentile(선형 보간)을 구한다. 턴 표본이 0인 반복은 제외하고 유효 반복 수를 기록한다. 부트스트랩 중 재학습·재선택은 하지 않는다.

하이퍼파라미터 선택·조기 종료·보고가 모두 같은 validation이라 절대 수치는 낙관적일 수 있음(A·B 동일 조건). CI는 고정된 선택 모델과 검증 표본에 조건부이며 이 선택 편향이나 모델 학습 불확실성을 보정하지 않는다. SciPy 1.18.1은 LightGBM 4.7.0의 필수 런타임 의존성이므로 버전 고정을 유지한다. 부트스트랩 자체는 NumPy만 사용한다.

당시 사용자 지시로 #7에서는 **test.jsonl을 읽지 않고 시험 평가를 보류했다**. `train.py`에는 test 접근 옵션이 없고 분할 표의 test 수·범위는 manifest 메타데이터만 인용한다. 이후 #15에서 승인된 최종 평가를 한 번 완료했으며 추가 승인을 기다리는 상태가 아니다. #7의 validation은 조기 종료·선택에 쓰인 자료이므로 이 결과는 독립 시험 성능이 아니다.

[experiment.json](results/experiment.json)과 [report.md](results/report.md)에 턴 구분별 기준·A·B 지표와 B−A·95% CI, train/validation 분포와 상관, 특징, 후보별 선택·교차율, 패키지·시드·명령·시각·해시와 한계를 저장한다. JSON의 `validation_by_cohort`가 분리 결과이며 기존 전체 지표 필드도 유지한다. 출력은 집계 두 파일뿐이며 표본·원문·모델 바이너리는 저장하지 않는다. 수치/LightGBM 환경이 없는 Python은 관련 테스트를 skip하고, 가상환경에서는 합성 학습과 그룹 부트스트랩·턴 분리·test 접근 차단·재현성까지 검증한다.

## 완료: 사용자 분리·train 내부 튜닝 재검증 (#13)

[revalidate.py](revalidate.py)는 기존 #7 산출물을 보존하고 별도 결과를 만든다. 기존 train/validation 시간 경계에서 양쪽에 등장한 `hashed_ip`와 ID 결측이 있는 원래 대화 그룹 전체를 제외한다. train 내부에도 시간순 80/20 홀드아웃과 같은 사용자 분리를 적용한다. A/B 설정·조기 종료 횟수 및 비교 기준선은 내부에서 고정하고, 전체 정제 train으로 재학습한 뒤 validation을 보고에만 쓴다. 분위수를 정렬하지 않으며 교차를 그대로 평가한다.

준비 표본에는 사용자 ID가 없어 원본 **단일 train Parquet 샤드의 `hashed_ip`·`timestamp` 열만** 연결한다. Python 객체로 변환하는 원본 행은 허용된 두 분할의 source_row_id뿐이며 대화 원문은 읽지 않는다. 소스 파일명·행 수·선택 행 시각을 준비 manifest와 대조한다. 원본 전체 바이트 해시는 재검증하지 않고 기존 manifest의 출처 해시를 기록한다. 의존성 추가·전처리 재실행·추가 추출이 없다.

```sh
/workspace/venvs/wildchat/bin/python wildchat/revalidate.py --prepared "$WILDCHAT_DATA_DIR"/derived/issue5-shard0 --source "$WILDCHAT_DATA_DIR"/data/train-00000-of-00014.parquet --output wildchat/results --seed 42 --inner-fraction 0.8 --max-rounds 500 --early-stopping-rounds 40 --threads 4 --bootstrap-replicates 1000
/workspace/venvs/wildchat/bin/python -m unittest wildchat.tests.test_revalidate wildchat.tests.test_metrics
python3 scripts/verify.py
```

고정 분위수·직전 응답 비율·대화 누적 평균·A·B의 30개 조합(3턴 구분×2가중×5모델)에 평균/분위수별 pinball, P50 MAE, coverage, 평균 구간 폭과 교차율을 제공한다. 누적 평균은 같은 원본 대화의 현재 턴 이전 유효 쌍 전체로 계산한다. 사용자 bootstrap 1,000회는 같은 사용자 추출을 모든 모델·턴·가중 방식에 공유한다. 최선 기준선은 턴·가중 방식별 내부 홀드아웃 점수로 미리 정하며 모든 개별 기준선 대비 CI도 보존한다.

결과는 [revalidation.md](results/revalidation.md), 전체 설정·수치는 [revalidation.json](results/revalidation.json), 실행·분할 검증은 [revalidation.log](results/revalidation.log)에 자동 저장된다. 이슈 PR용 요약은 [revalidation_pr.md](results/revalidation_pr.md), 실제 두 번 실행의 수치 대조 기록은 [revalidation_verification.json](results/revalidation_verification.json)에 있다. CI는 고정된 학습 결과에 조건부이며 IP 식별 오차·시간 블록 상관·학습 불확실성을 포함하지 않는다. 사용자 제거로 평가 모집단이 바뀌므로 기존 −9.55와의 차이를 보정 요인의 개별 효과로 해석하지 않는다.

## 완료: 승인된 최종 테스트 1회 평가 (#15)

2026-11-03 종현 님의 승인으로 #15에 한해 test 접근을 허용했고 같은 날 평가를 완료했다. 위 #7/#13의 보류 설명은 당시 실험에 적용된다. [finalize.py](finalize.py)는 PR #14의 고정 코드·선택 설정을 사용했다. 기준 커밋은 `922370cea7abc36e85379db4cee3f99cf3e8e4d6`이며 보존할 5개 코드 파일과 해시는 [고정 설정](results/test_final_config.json)의 `code_sha256`을 따른다. 이동 가능한 원격 브랜치 이름을 현재 상태의 증거로 사용하지 않는다.

학습은 #13의 정제 train 81,307행만 쓴다. validation을 합치지 않아 같은 학습 표본·모델 설정·반복 수를 유지한다. A의 P10/P50/P90은 231/175/175회, B는 403/244/279회로 고정한다. 조기 종료나 재튜닝을 하지 않는다. 기준선 추정값과 턴·가중별 최선 기준선도 #13 그대로다.

실행 이력은 다음과 같다. **이미 완료된 실행을 다시 실행하지 않는다.**

```sh
/workspace/venvs/wildchat/bin/python wildchat/finalize.py freeze
/workspace/venvs/wildchat/bin/python wildchat/finalize.py evaluate
```

첫 명령은 [고정 설정](results/test_final_config.json)과 [SHA256](results/test_final_config.sha256)을 테스트 접근 전에 만든다. 두 번째 명령은 코드·설정·환경·준비 manifest·train/validation 해시를 확인하고 #13의 train 정제와 기준선 추정을 재현한다. A/B를 같은 횟수로 학습한 후, 기존 validation의 모든 지표가 1e−10 이내로 일치하는지 확인한다. 이어 학습 모델 해시를 기록한 뒤 test를 딱 한 번 읽으면서 파일 해시도 계산한다. 원래 train 또는 validation과 사용자·그룹·원본 행이 겹치거나 ID가 결측인 test 대화 그룹 전체를 제외한다. 이 규칙은 테스트 전에 고정하며 학습 표본은 바꾸지 않는다.

출력 경로는 고정이다. [로그](results/test_final.log)를 독점 생성하며 로그 또는 결과가 존재하면 재실행을 거부한다. 실패한 시도도 로그를 남기므로 자동 재시도하지 않는다. 결과 확인 뒤 모델·설정·예측 코드 수정과 재평가는 금지한다. [최종 보고서](results/test_final.md), [기계 판독 결과](results/test_final.json), [PR 본문](results/test_final_pr.md)에 5모델×3턴 구분×2가중, 평균/분위수별 pinball·MAE·coverage·구간 폭·교차율과 B−A/기준선 대비 사용자 bootstrap 95% CI를 기록한다. 검증셋 대비 차이는 기간과 사용자 구성의 차이를 함께 포함한다.

검증은 실제 test 파일을 읽지 않는 합성 fixture만 사용한다. 기존 전처리 CLI 통합 테스트 1개는 합성 `test.jsonl`을 읽으므로 수치 환경의 아래 검증에서 제외한다. 표준 Python의 전체 discover에서는 해당 선택 의존성 통합 테스트가 skip된다.

```sh
/workspace/venvs/wildchat/bin/python -m unittest wildchat.tests.test_finalize wildchat.tests.test_revalidate wildchat.tests.test_experiment wildchat.tests.test_metrics wildchat.tests.test_prepare.FeatureTests wildchat.tests.test_prepare.GroupSplitTests
python3 -m unittest discover -s wildchat/tests
python3 scripts/verify.py
```

## H1: 같은 대화 순차 특징의 validation 재검증 (#23)

[temporal.py](temporal.py)는 B(31개), B+H1(40개), C_emb_PCA(k=32, 63개), C_emb_PCA+H1(72개)를 동일한 #13 정제 train/validation과 내부 분할에서 비교한다. 네 조건 각각에 같은 3개 LightGBM 후보·최대 500회·40회 조기 종료를 적용하고, 설정·분위수별 반복 수는 train 내부 홀드아웃 행 가중 평균 pinball로만 정한다. 전체 정제 train으로 재학습하며 validation은 보고에만 쓴다. #19의 전역 B 설정을 모든 PCA 조건에 고정한 실험과는 선택 절차가 다르다.

[temporal_features.py](temporal_features.py)의 H1은 같은 원본 대화의 이전 유효 응답만 누적한다. 출력 EWMA(α=0.3), 누적 평균·모집단 표준편차(ddof=0), 최근 두 출력 차이·원래 user 턴 간격당 기울기, 직전 유효 출력/입력 비율·그 EWMA, 현재/직전 유효 입력 배율, 관측 수의 9개다. 특징을 기록한 뒤 현재 응답을 갱신한다. 불완전한 쌍은 건너뛰되 원래 턴 위치를 유지하고 사용자나 중복 그룹 간에는 상태를 공유하지 않는다. EWMA는 첫 유효 값으로 초기화한다. 이력이 없으면 count=0 외에는 NaN, 1개면 std=0, 2개 미만이면 차이·기울기는 NaN이다. 입력이 0이면 해당 비율은 NaN이고 비율 EWMA에서만 제외한다.

이번 실행의 운영자 지시에 따라 기존 MiniLM 캐시를 재사용한다. `--embedding-reference`의 기존 SHA256, 캐시 형식·384차원·유한값·키 유일성·train/validation 전체 키와의 정확한 일치를 검사한다. 불일치하면 중단하며 자동 재생성하지 않는다. cache의 float16을 float32로 바꾸고 로컬 [PCA](category_pca.py)를 내부 fit 행/전체 정제 train에서 각각 적합한다. 원문이나 다른 분할을 읽을 필요가 없다. JSONL 접근은 실행 중 train/validation 두 경로로 제한한다.

```sh
export WILDCHAT_DATA_DIR=/workspace/data/wildchat-1m
OPENBLAS_NUM_THREADS=1 /workspace/venvs/wildchat/bin/python wildchat/temporal.py \
  --prepared "$WILDCHAT_DATA_DIR/derived/issue5-shard0" \
  --source "$WILDCHAT_DATA_DIR/data/train-00000-of-00014.parquet" \
  --embeddings "$WILDCHAT_DATA_DIR/derived/issue19-categories/embeddings.npz" \
  --embedding-reference wildchat/results/category_pca.json \
  --output wildchat/results --seed 42 --inner-fraction 0.8 --threads 4 \
  --max-rounds 500 --early-stopping-rounds 40 --bootstrap-replicates 1000
OPENBLAS_NUM_THREADS=1 /workspace/venvs/wildchat/bin/python -m unittest wildchat.tests.test_temporal wildchat.tests.test_revalidate wildchat.tests.test_metrics wildchat.tests.test_category wildchat.tests.test_prepare.FeatureTests wildchat.tests.test_prepare.GroupSplitTests
python3 -m unittest discover -s wildchat/tests
python3 scripts/verify.py
```

출력은 [temporal_h1.json](results/temporal_h1.json), [temporal_h1.md](results/temporal_h1.md), [temporal_h1.log](results/temporal_h1.log)다. 첫/후속/전체 턴×행/사용자 가중별 네 조건의 val pinball과 B+H1−B·C_emb_PCA+H1−C_emb_PCA를 보고한다. 사용자 paired bootstrap 1,000회(seed 42, PCG64)는 모든 조건·턴·가중에 같은 추출을 공유하며 percentile 선형 보간 95% CI를 계산한다. 분위수는 정렬하지 않는다. 테스트는 합성 자료만 사용하며 현재·미래 응답 변경 불변성, 대화 간 격리, invalid 쌍·0분모, 캐시 정합성, validation의 모델/PCA 적합 배제, paired bootstrap을 검사한다.

이미 반복 관측한 validation에 대한 후속 비교이며 독립 test 확인이나 Mac 탐색의 확증으로 해석하지 않는다. 첫 샤드 고정 표본·가시 텍스트 길이에 한정하고 CI는 학습·선택 불확실성을 포함하지 않는다. 설계는 [ADR 0004](../docs/decisions/0004-wildchat-h1-validation.md)를 따른다. 전체 `verify`를 막던 TraceLab 링크 오류 8건은 원본 릴리스 커밋 링크로 수정했으며 원본·로컬 해시 보존 방식은 [WORKFLOW §8](../WORKFLOW.md#8-검증-명령과-기존-한계)에 있다.
