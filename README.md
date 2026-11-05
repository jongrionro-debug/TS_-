# 대화 이력을 활용한 LLM 응답 토큰 예측

시계열 데이터 분석 및 실습 과목 프로젝트. **현재 작업은 WildChat에서 현재 프롬프트의 길이만 쓰는 모델 A와 대화 이력을 추가한 모델 B의 응답 토큰 예측 비교**다. 2026-11-03 기준 사용자 분리 재검증(#13)과 최종 test 1회 평가(#15)를 완료했다.

정본은 [docs/project.md](docs/project.md), 작업 지침은 [AGENTS.md](AGENTS.md)와 [workflow.md](docs/workflow.md), 실험 실행 안내는 [wildchat/README.md](wildchat/README.md)다.

## 현재 결과

첫 Parquet 샤드의 고정 표본에서 사용자·대화 그룹을 분리하고 train 내부에서만 설정을 선택했다. 정제 train은 81,307행, validation은 12,920행, 최종 test는 11,942행·1,770사용자다. 입력/정답은 `o200k_base`로 가시 텍스트를 재계수한 길이이며 실제 API 사용량이 아니다.

최종 test 전체 행 가중 평균 pinball은 A 55.849263, B 45.455475, B−A −10.393788(사용자 bootstrap 95% CI [−12.732540, −8.279974])였다. 후속 턴의 개선이 크고 첫 턴 행 가중 차이의 CI는 0을 포함한다. 전체 14개 샤드·GPU 비용 절감으로 일반화하지 않는다. 상세 지표·한계는 [최종 보고서](wildchat/results/test_final.md), [기계 판독 결과](wildchat/results/test_final.json), [재검증 보고서](wildchat/results/revalidation.md)에 있다.

**최종 test 평가는 이미 한 번 완료됐다.** `finalize.py freeze/evaluate`는 실행 이력이며 재실행하지 않는다. 코드·설정·SHA256·로그를 보존한다.

## 프로젝트 구성

| 경로 | 역할 |
|---|---|
| `wildchat/` | 현재 요청별 응답 길이 실험·전처리·평가·테스트 |
| `wildchat/results/` | 조사 → 전처리 → 탐색 실험 → 재검증 → 최종 test의 집계 산출물 |
| `docs/project.md`, `docs/adr/` | 연구 정의와 채택한 설계 결정 |
| `eda/eda.ipynb`, `eda/results/` | BurstGPT의 다음 10분 집계 입력·출력 수요 EDA와 보존 설계 |
| `eda/agent_transition/` | TraceLab·Exgentic 후보 조사의 재현 코드·결과·README |
| `output/`, `ref/` | 기존 BurstGPT 발표·제안 자료와 참고 논문(PDF 보존) |
| `scripts/verify.py` | 저장소 JSON/ipynb·문서 상대 링크 검사 |

## 시작하기

저장소 루트에서 다음을 실행한다.

```sh
python3 scripts/verify.py
python3 -m unittest discover -s wildchat/tests
```

형식 검증은 표준 라이브러리만 사용한다. 기존 외부 문서 사본의 상대 링크 오류 8건은 [workflow.md §8](docs/workflow.md#8-검증-명령과-기존-한계)에 기록돼 있다. WildChat 의존성이 없는 Python에서는 수치·Parquet 통합 테스트가 skip된다. 학습 환경 버전은 [requirements.txt](wildchat/requirements.txt), 데이터 준비와 단계별 명령은 [WildChat README](wildchat/README.md)를 따른다. 원본·생성 표본·모델 바이너리는 커밋하지 않는다.

## BurstGPT 보존 트랙

BurstGPT는 다음 10분에 **도착하는 요청**의 입력·최종 출력 합계를 예측하는 별도 설계다. EDA는 완료됐으며 WildChat의 요청별 응답 예측과 목표·토큰 정의가 다르다. 기존 EDA를 재현할 필요가 있을 때만 다음 절차를 사용한다.

1. 원본 데이터 `data/BurstGPT_3.csv`의 존재를 확인합니다. `.gitignore` 대상이며 Git에는 포함되지 않습니다. 2026-11-03 점검한 로컬 Mac에는 있지만 다른 환경에도 있다고 가정하지 않습니다.
2. Jupyter 환경에서 `eda/eda.ipynb`를 위에서 아래 순서로 실행합니다.
3. 분석 결과는 `eda/results/`에 저장됩니다. 요약은 `eda/results/summary.md`와 `eda/results/summary.json`에서 확인할 수 있습니다.

## 검증

형식·링크는 `python3 scripts/verify.py`, 구조만 확인할 때는 `python3 scripts/check_context.py`를 실행한다. 의존성·skip·기존 링크 오류 한계는 위 시작하기를 따른다.

## 문서

[문서 안내](docs/README.md) · [현재 프로젝트](docs/project.md) · [작업 규칙](AGENTS.md) · [결정 기록](docs/adr/README.md)

작업 요청은 [Issue 템플릿](.github/ISSUE_TEMPLATE/task.yml), 변경·검증 결과는 [PR 템플릿](.github/pull_request_template.md)을 사용한다.
