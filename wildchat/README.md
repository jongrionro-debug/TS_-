# WildChat 데이터 조사

[AllenAI WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), Zhao 등 [WildChat 논문](https://arxiv.org/abs/2405.01470). 데이터 카드의 라이선스는 [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/)이다. 출처 리비전은 `7d6490e462285cf85d91eabea0f9a954fbddcd1f`이며 데이터는 이 저장소에 포함하지 않는다.

이번 조사는 14개 샤드 중 `train-00000-of-00014.parquet` 하나를 그대로 순회한다. 무작위 표본이 아니며 시드는 없다. 첫 샤드는 59,857행이고 전체 데이터의 결과가 아니다. 전체 데이터 위치와 전체 실행 여부는 미결정이다. 파일을 이름순으로 읽고 `--max-rows N`을 지정하면 그 순서의 앞 N행만 추가로 선택한다. 이 역시 무작위 표본이 아니다.

의존성은 [requirements.txt](requirements.txt)에 기록한다. `o200k_base` 인코딩 파일이 로컬 캐시에 없으면 최초 사용 시 다운로드가 필요하다. 네트워크가 없는 환경에서는 미리 캐시가 준비되어야 한다. 승인된 환경에서 실행한 명령은 다음과 같다.

```sh
/workspace/venvs/wildchat/bin/python wildchat/inspect.py --input /workspace/data/wildchat-1m/data/train-00000-of-00014.parquet --output wildchat/results/data_audit.md
python3 scripts/verify.py
```

다른 환경에서는 의존성이 설치된 Python을 사용하고 `--input`에 Parquet 파일 또는 디렉터리를 지정한다. 디렉터리는 하위의 모든 `.parquet` 파일을 이름순으로 읽는다. 배치와 임시 SQLite를 사용하므로 전체 원문을 메모리에 올리지 않으며, 임시 저장에는 원문의 SHA-256 해시만 쓴다. 결과는 [data_audit.md](results/data_audit.md)에 집계만 기록한다. 행 중복과 공유 prefix 판정의 정확한 기준과 한계도 보고서를 따른다.

유효 쌍은 바로 인접한 비어 있지 않은 user와 assistant 텍스트다. `o200k_base`로 각 텍스트를 따로 계수한 값은 API 실제 사용량이 아니다. 분석은 학습·성능 평가를 포함하지 않는다. 저장소 밖의 데이터 및 토크나이저 캐시는 별도로 준비해야 한다.
