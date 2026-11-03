# AGENTS.md — 에이전트용 지도 (100줄 이내 유지)

## 0. 정본과 현재 상태 (2026-11-03)
- 연구 정의의 정본은 [docs/prd.md](docs/prd.md), 작업 절차는 [WORKFLOW.md](WORKFLOW.md)다.
- 현재 작업은 WildChat의 **현재 프롬프트에 대한 응답 토큰 수 예측**이다. 길이만 쓰는 A와 대화 이력을 더한 B를 비교한다.
- #13 사용자 분리·train 내부 튜닝 재검증과 #15 최종 test 1회 평가가 완료됐다. 다음 할 일을 모델 구현이나 test 승인 대기로 안내하지 않는다.
- BurstGPT의 다음 10분 집계 입력·출력 수요 설계와 EDA는 별도 보존 트랙이다. WildChat 결과로 그 가설을 검증했다고 해석하지 않는다.
- 큰 연구 정의 변경은 먼저 기존 사용자 결정과 PRD를 확인하고, 미결정 사항만 질문한 뒤 `docs/decisions/`에 ADR을 기록한다. 작은 보완은 기존 문서를 갱신한다.

## 1. 필요한 진입점
| 알고 싶은 것 | 위치 |
|---|---|
| 연구 질문·현재 범위·BurstGPT 보존 설계 | [docs/prd.md](docs/prd.md) |
| 실행 환경·명령·단계별 산출물 | [wildchat/README.md](wildchat/README.md) |
| 최종 test 결과·설정·실행 증거 | `wildchat/results/test_final.md`, `.json`, `_config.json`, `_config.sha256`, `.log` |
| 사용자 분리·내부 튜닝·validation 결과 | `wildchat/revalidate.py`, `wildchat/results/revalidation.md`, `.json` |
| 전처리·과거 탐색 실험 | `wildchat/prepare.py`, `analyze.py`, `train.py`, `wildchat/results/` |
| 테스트 | `wildchat/tests/test_*.py` (unittest·합성 자료) |
| 설계 결정 | `docs/decisions/` (NNNN-*.md, 템플릿 0000) |
| BurstGPT EDA·요약·실행 환경 | `eda/eda.ipynb`, `eda/results/summary.md`, `summary.json`, `manifest.json` |
| 과거 전환 검토·발표 자료 | `docs/adr/가설1.md`, `제안1.md`, `eda/agent_transition/`, `output/` |

## 2. 명령과 환경
- 전체 형식·상대 링크 검증(필수): `python3 scripts/verify.py` (표준 라이브러리만 사용).
- 기존 테스트: `python3 -m unittest discover -s wildchat/tests`. 수치·Parquet 의존성이 없으면 관련 테스트가 skip되므로 통과/skip 수를 함께 보고한다.
- WildChat 의존성은 `wildchat/requirements.txt`에 고정돼 있다. box의 `/workspace/venvs/wildchat/bin/python`은 과거 실행 환경이며 로컬에 있다고 가정하지 않는다.
- 빌드·별도 린트·타입체크 설정은 없다. 새 의존성·테스트 프레임워크를 임의로 추가하지 않는다.
- BurstGPT 노트북은 `data/BurstGPT_3.csv`와 노트북 의존성이 필요하다. CSV는 gitignore 대상이며 환경별 존재 여부를 확인한다.
- 전체 검증에는 보존된 TraceLab 외부 문서 사본의 기존 링크 오류가 있다. [WORKFLOW.md §8](WORKFLOW.md#8-검증-명령과-기존-한계)를 따르고 실패를 통과로 보고하지 않는다.

## 3. 실험·자료 보존
- WildChat은 첫 Parquet 샤드의 고정 표본 결과다. 전체 14개 샤드나 실제 API 사용량·GPU 효과로 일반화하지 않는다.
- 현재 비교는 고정 분위수·직전 응답 비율·대화 누적 평균·A·B의 5개다. #7의 3모델·validation 튜닝·분위수 정렬은 과거 규칙이다.
- 현재 규칙은 사용자/대화 그룹 분리, train 내부 튜닝, 분위수 무정렬, 사용자 bootstrap이다. 세부 사항은 PRD와 #13/#15 산출물을 따른다.
- **`wildchat/finalize.py freeze/evaluate`는 이미 완료된 실행 이력이다. 재실행·재튜닝·실제 test 자료 재접근을 하지 않는다.** 합성 fixture 기반 테스트는 가능하다.
- `test_final_config.json`에 고정된 5개 코드 파일과 설정·결과·로그를 문서 수정 중 변경하지 않는다. 새 실험은 별도 범위 결정이 필요하다.
- `eda/results/*`, `wildchat/results/*`는 실행 산출물이다. 수치·생성 보고서를 손으로 바꾸지 않는다.
- 원문·생성 표본·모델 바이너리는 커밋하지 않는다. `ref/`, PDF, 데이터·대용량 파일, 출처 해시가 있는 `source/` 사본은 보존한다.
- 시크릿/.env/자격증명 파일 읽기·수정 금지. 사용자 ID·대화 원문을 채팅이나 보고서에 노출하지 않는다.

## 4. 작업과 Git
- 요청/이슈 AC 범위에서 최소 변경한다. 의미 있는 변경은 바꿀 동작·지킬 동작·확인 방법을 3줄 이내로 알린다.
- 한 메인 에이전트가 기본이다. 원본 프로젝트에서 작업하고 기존 변경을 보존하며, 새 worktree는 사용자 명시 요청 때만 만든다.
- 착수 시 `git status --short`, 현재 브랜치·HEAD를 확인한다. 기본 작업 브랜치는 `가설1`; 이슈 작업은 `agent/issue-<n>-<slug>`, 이슈 없는 작업은 `agent/<slug>`를 사용한다.
- Git 요청에는 필요한 브랜치·커밋·push·PR·병합·기본 브랜치 반영 권한이 포함된다(2026-10-30 사용자 위임). 단계별 재확인하지 않으며 이번 요청의 보류·범위 제한이 우선한다.
- 이슈 없이 받은 로컬 작업에 GitHub 이슈·게시를 자동으로 요구하지 않는다. 삭제·대용량·CI 변경이 필요하면 요청 범위와 권한을 먼저 확인한다.
- 막연한 불확실성은 `rg`, 관련 함수·산출물, `git log`, 관련 이슈/PR로 줄이고 결과를 크게 바꾸는 미결정 사항만 질문한다.

## 5. 완료 전 확인
1. 요청/AC별 근거, 필수 검증·관련 테스트 결과와 skip/기존 실패를 확인한다.
2. `git diff --check`, `git diff --stat`으로 범위와 기존 작업 보존을 확인하고 이번 임시 파일을 처리한다.
3. 영구 지식은 기존 PRD·ADR·지침에 반영한다. 별도 계획·요약 MD를 임의로 만들지 않는다.
4. 채팅 작업은 결과·위치·검증·남은 점을 한국어로 간결히 보고한다. box worker의 기계 인계는 WORKFLOW.md §5 JSON을 사용한다.
5. 하위 AGENTS.md가 있으면 해당 디렉터리 작업 시 우선 적용한다.
