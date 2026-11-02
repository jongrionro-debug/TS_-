# AGENTS.md — 에이전트용 지도 (100줄 이내 유지)

## 0. 팀 기획 규칙 (기존 agents.md 원문 유지)
- 우리 기획의 source of truth는 docs/prd.md이다.
- docs에는 prd.md, adr 폴더가 있다. (adr 폴더 = `docs/decisions/`)
- adr 폴더는 우리 기획에 특성 수정사항이 생겼을 경우에 adr.md(작성 전 질문)를 작성한다.
  큰 변경이 아니면 prd.md에 관련 내용이 있으면 해당 내용만 수정한다.

## 1. 이 저장소는
- 한 줄 설명: BurstGPT 대화형 로그로 다음 10분 입력·출력 토큰 수요를 단기 예측하는 시계열 연구 (과목 프로젝트)
- 스택: Python 3 + Jupyter 노트북 (eda/eda.ipynb), Markdown 문서. 빌드/테스트 프레임워크 없음
- 작업 절차는 WORKFLOW.md, 기획 원본은 docs/prd.md

## 2. 먼저 읽을 곳 (필요한 것만)
| 알고 싶은 것 | 위치 |
|---|---|
| 연구 질문·목표 정의·실험 범위 | docs/prd.md |
| 이슈→PR 절차, 완료 기준, handoff 스키마 | WORKFLOW.md |
| 과거 설계 결정 (ADR) | docs/decisions/ (NNNN-*.md, 템플릿 0000) |
| EDA 코드 | eda/eda.ipynb |
| EDA 산출물 요약 | eda/results/summary.md, summary.json, manifest.json |
| 발표/제안 자료 | output/ |
| 참고 논문 (PDF, 수정 금지) | ref/ |

## 3. 명령어
- 설치: 없음 (검증 스크립트는 표준 라이브러리만 사용)
- 검증(전체, 필수): `python3 scripts/verify.py`
  - 모든 *.json / *.ipynb 가 JSON으로 파싱되는지, *.md 의 상대 링크가 실제 파일을 가리키는지 확인
- 테스트/린트/타입체크: 아직 없음 — 추가하려면 이슈로 제안
- 노트북 실행: 원본 데이터 `data/BurstGPT_3.csv`(gitignore, box에 없음)가 필요 → 재실행이 필요하면 decisions_needed

## 4. 작업 규칙
- 새 의존성 추가 금지 (이슈에 허용 명시된 경우만)
- 기획/정의 변경은 docs/prd.md 해당 부분만 수정, 큰 변경은 docs/decisions/에 ADR
- eda/results/* 산출물은 노트북 재실행 결과물 — 손으로 수정 금지
- ref/*.pdf, output/*.pdf, data/ 등 대용량·바이너리 파일 수정·삭제 금지
- 문서는 한국어 기본, 수치는 eda/results 근거와 일치시킬 것

## 5. 작업 경계 (반드시)
- 이슈의 Acceptance Criteria 범위 밖 변경 금지 (발견 시 risks에 기록)
- 기본 브랜치(현재 `가설1`) 직접 push/merge 금지, 브랜치: agent/issue-<n>-<slug>
- 시크릿/.env/자격증명 파일 읽기·수정 금지
- 파일 삭제·대용량 파일 변경·CI 설정 변경은 decisions_needed로 보고
- 모호하면: 문서·코드 조사로 불확실성 줄이기 → 그래도 남으면 질문 1~3개

## 6. 맥락 찾는 법
- candidate_paths는 출발점일 뿐, rg로 자유롭게 탐색
- 관련 이력: `gh issue view <n>`, `gh pr view <n>`, `git log -S <문자열>`
- 과거 PR 내용은 이후 바뀌었을 수 있으니 현재 파일로 재확인

## 7. 끝내기 전 자기점검
1. AC 각 항목을 무엇으로 검증했는지
2. `python3 scripts/verify.py` 실행 결과
3. 범위 밖 변경 없음 확인 (`git diff --stat`)
4. 영구 지식 정리: 설계 결정→docs/decisions, 재발 방지 규칙→본 파일 또는 verify 스크립트
5. 마지막 출력은 handoff JSON (WORKFLOW.md §5 스키마)

## 8. 하위 AGENTS.md
- 하위 디렉터리에 AGENTS.md가 있으면 해당 디렉터리 작업 시 우선 적용
