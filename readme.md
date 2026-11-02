adr 폴더는 우리 기획에 특성 수정사항이 생겼을 경우에 adr.md를 작성한다. 큰 변경이 아니면 prd.md에 관련 내용이 있으면 해당 내용만 수정한다.

## 저장소 구조

- `docs/`: 프로젝트 기획서와 설계 결정(ADR)을 보관합니다.
- `eda/`: BurstGPT 데이터 탐색·분석 노트북을 보관합니다.
- `eda/results/`: 노트북에서 생성한 분석 요약, 표, 그림 등 산출물을 보관합니다.
- `output/`: 발표·제안 자료를 보관합니다.
- `ref/`: 참고 논문 등 연구 자료를 보관합니다.

## 재현 방법

1. 원본 데이터 `data/BurstGPT_3.csv`를 저장소의 `data/` 경로에 둡니다. `data/`는 Git에서 제외되므로 데이터 파일은 저장소에 포함되지 않습니다.
2. Jupyter 환경에서 `eda/eda.ipynb`를 위에서 아래 순서로 실행합니다.
3. 분석 결과는 `eda/results/`에 저장됩니다. 요약은 `eda/results/summary.md`와 `eda/results/summary.json`에서 확인할 수 있습니다.
