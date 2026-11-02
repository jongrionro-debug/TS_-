# WildChat 데이터 조사와 요청 표본 준비

[AllenAI WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), Zhao 등 [WildChat 논문](https://arxiv.org/abs/2405.01470). 데이터 카드의 라이선스는 [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/)이다. 출처 리비전은 `7d6490e462285cf85d91eabea0f9a954fbddcd1f`이며 데이터는 이 저장소에 포함하지 않는다.

이번 조사는 14개 샤드 중 `train-00000-of-00014.parquet` 하나를 그대로 순회한다. 무작위 표본이 아니며 시드는 없다. 첫 샤드는 59,857행이고 전체 데이터의 결과가 아니다. 전체 데이터 위치와 전체 실행 여부는 미결정이다. 파일을 이름순으로 읽고 `--max-rows N`을 지정하면 그 순서의 앞 N행만 추가로 선택한다. 이 역시 무작위 표본이 아니다.

의존성은 [requirements.txt](requirements.txt)에 기록한다. `o200k_base` 인코딩 파일이 로컬 캐시에 없으면 최초 사용 시 다운로드가 필요하다. 네트워크가 없는 환경에서는 미리 캐시가 준비되어야 한다. 승인된 환경에서 실행한 명령은 다음과 같다.

```sh
/workspace/venvs/wildchat/bin/python wildchat/inspect.py --input /workspace/data/wildchat-1m/data/train-00000-of-00014.parquet --output wildchat/results/data_audit.md
python3 scripts/verify.py
```

다른 환경에서는 의존성이 설치된 Python을 사용하고 `--input`에 Parquet 파일 또는 디렉터리를 지정한다. 디렉터리는 하위의 모든 `.parquet` 파일을 이름순으로 읽는다. 배치와 임시 SQLite를 사용하므로 전체 원문을 메모리에 올리지 않으며, 임시 저장에는 원문의 SHA-256 해시만 쓴다. 결과는 [data_audit.md](results/data_audit.md)에 집계만 기록한다. 행 중복과 공유 prefix 판정의 정확한 기준과 한계도 보고서를 따른다.

유효 쌍은 바로 인접한 비어 있지 않은 user와 assistant 텍스트다. `o200k_base`로 각 텍스트를 따로 계수한 값은 API 실제 사용량이 아니다. 분석은 학습·성능 평가를 포함하지 않는다. 저장소 밖의 데이터 및 토크나이저 캐시는 별도로 준비해야 한다.

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

분할은 시각이 유효하고 유효 쌍이 있는 그룹을 최초 기록시각 순으로 정렬한 뒤 그룹 수의 70%/85% 위치를 경계로 삼는다. 경계에 걸친 그룹은 통째로 제외하고 같은 시각은 뒤 구간에 둔다. 따라서 제거 후 비율은 70%/15%/15%와 다를 수 있다. 기록시각 순 분할이며 실제 요청 도착시각 순서의 보장은 아니다. MAE·log 오차·pinball·커버리지는 후속 평가에서 산출한다.

테스트는 합성 대화만 사용한다. 표준 라이브러리 Python에서는 순서·결측·불변성·그룹 분리 테스트를 실행하고 실제 tokenizer/Parquet CLI 통합 테스트 1개는 건너뛴다. 위 WildChat 가상환경에서는 통합 테스트까지 모두 실행한다.
