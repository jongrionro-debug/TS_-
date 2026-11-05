# WORKFLOW.md — 로컬 협업과 이슈→PR 실행 계약

최종 갱신: 2026-11-03. 현재 연구·완료 단계는 [PRD](docs/prd.md), 실행 환경은 [WildChat README](wildchat/README.md)를 따른다. 이 문서는 저장소에서 확인할 수 있는 절차를 정의하며 box의 외부 자동화가 현재 실행 중이라고 보장하지 않는다.

## 0. 적용 범위와 사용자 위임
- 로컬 채팅 요청은 요청 자체가 작업 범위다. GitHub 이슈·Notion 카드 없이도 조사·문서 수정·검증을 진행한다.
- GitHub 이슈가 지정된 box 작업에는 아래 라벨·AC·handoff 계약을 적용한다. 한 메인 에이전트·원본 체크아웃이 기본이며 새 worktree는 사용자 명시 요청 때만 만든다.
- Git 관련 요청에는 필요한 브랜치 생성·전환, 커밋, push, PR 생성·병합과 기본 브랜치 반영 권한이 포함된다(2026-10-30 사용자 위임). 단계별 재확인은 하지 않고 이번 요청의 명시적 보류를 우선한다.
- 과거의 “머지는 사람만”, “기본 브랜치 반영 금지”는 위 위임으로 대체됐다. 로컬 문서 수정 요청만으로 외부 게시까지 자동 수행하지는 않는다.

## 1. 상태 (GitHub 라벨이 원본)
agent:triage → agent:ready → agent:in-progress → agent:review → merge → closed
                       ↘ agent:needs-decision / agent:blocked
- ready 조건: Goal·AC 존재, mode·difficulty 지정, Mac 필요 아님
- in-progress: worker가 잡은 순간 라벨 변경 + 코멘트(세션·브랜치)
- review: PR 생성 + 검증 결과 확정
- needs-decision: 조사 후에도 결과를 크게 바꾸는 선택이 남으면 현재 채널에 질문한다. Notion 카드 생성은 별도 요청/승인된 연동 작업일 때만 한다.
- blocked: 필요한 입력·환경이 없어 진행 불가. 같은 실패를 기계적으로 반복하지 않고 원인과 남은 작업을 보고한다.
- agent:needs-mac: box에서 실행 불가 (Mac 필요) → worker가 잡지 않음
- difficulty:low|medium|high: 모델·reasoning effort 선택 기준 (§7)

## 2. 실행 단위
- 1 이슈 = 1 브랜치(`agent/issue-<n>-<slug>`) = 1차 PR, 한 번에 한 이슈만. 이슈 없는 작업은 `agent/<slug>`를 사용한다.
- box의 기존 경로는 `/workspace/repos/<repo>`이며 로컬 Mac에서는 현재 원본 프로젝트 경로를 사용한다. `/workspace` 경로를 로컬 경로로 가정하지 않는다.
- 착수 시 작업 트리·브랜치·HEAD를 확인하고 기존 변경을 보존한다. 변경이 있다는 이유만으로 일괄 중단·reset하지 않으며 충돌하는 변경만 확인한다.
- 기본 작업 브랜치는 `가설1`이다. 이슈/PR 착수 시 필요한 최신화와 브랜치 전환은 기존 변경을 덮어쓰지 않는 범위에서 수행한다.
- Codex 세션은 교체 가능, 이슈가 작업 정체성. 세션 ID는 state와 PR 메타데이터에 기록

## 3. 구현 절차
1. AGENTS.md → 요청/이슈 → 관련 문서·진입점·핵심 로직·저장/외부 연동·테스트를 필요한 범위에서 추적한다.
2. 의미 있는 변경은 바꿀 동작·지킬 동작·확인 방법을 채팅에 3줄 이내로 알린다. 별도 계획 MD는 만들지 않는다.
3. 최소 변경 → `python3 scripts/verify.py` → 영향 범위의 기존 테스트를 실행한다. 문서 수치는 저장된 실행 결과와 대조한다.
4. 실패 원인을 요청/구현/환경/기존 실패로 구분한다. 통과를 위해 검증을 약화하거나 산출물 수치를 손으로 고치지 않는다.
5. Git 작업 요청이면 PR 템플릿과 검증 근거를 갖춰 요청된 커밋·push·PR·병합까지 진행한다. 로컬 작업은 변경과 검증을 보고한다.

## 4. 완료 정의 (DoD)
- 모든 AC에 검증 근거, verify 통과 (`python3 scripts/verify.py` exit 0)
- 기존 실패가 남으면 새 실패와 구분하고 원인·개수를 보고한다. 전체 DoD 통과로 표시하지 않는다.
- 범위 밖 변경 없음 또는 risks에 명시
- 문서 영향 있으면 docs/ 갱신
- `git diff --check`, 변경 범위·기존 작업 보존·이번 임시 파일 정리를 확인한다. 대체된 파일의 삭제와 관련 참조·진입점 수정·검증도 완료 조건이다.

## 5. Handoff JSON (box worker 기계 인계)

box worker의 최종 출력 형식이다. 일반 채팅 작업은 결과·핵심 위치·검증·남은 점을 한국어로 간결하게 보고하며 JSON을 강제하지 않는다.
```json
{
  "issue": 42, "status": "done|partial|blocked|needs_decision",
  "plan": ["..."], "changed_files": ["readme.md"],
  "tests_run": ["python3 scripts/verify.py"], "results": "0 errors",
  "ac_check": [{"ac": "...", "met": true, "evidence": "..."}],
  "risks": ["..."], "decisions_needed": ["..."],
  "followups": ["새 이슈 후보"]
}
```

## 6. 자료·실험·외부 작업 경계
- 시크릿·자격증명 접근, 요청 범위 밖 삭제·배포·외부 메시지 전송·새 의존성 추가를 하지 않는다. Git 요청의 push·PR·병합은 §0의 위임을 따른다.
- force-push는 자기 작업 브랜치 rebase 외에는 하지 않는다. Git 위임을 기존 작업이나 공유 브랜치 이력 삭제 권한으로 해석하지 않는다.
- 생성 산출물의 수치를 수작업으로 바꾸지 않는다. 현재 재현·검증·라이선스에 필요한 파일만 유지하며, 원본·PDF·출처 사본이라는 이유만으로 일괄 보존하지 않는다. #13/#15의 현재 근거인 고정 코드·설정·결과·로그는 유지한다.
- 대체된 제안·옛 설계·중복 보고서·미사용 코드와 산출물은 사용처와 Git 복구 가능 여부를 확인하고 작업 범위에서 삭제한다. archive/backup 폴더나 과거 설계 부록으로 옮겨 쌓지 않는다. 파일 내용은 Git, 중요한 결정은 ADR, 작업·검증 근거는 이슈·PR에 남긴다.
- 허용된 범위의 불필요한 커밋된 파일 삭제는 재승인받지 않는다. 미커밋 변경·미추적 파일·gitignore 데이터·유일한 원본은 복구 가능하다고 가정하지 않으며 별도 권한과 보존 수단을 확인한다. 삭제한 경로의 문서 링크·진입점도 함께 수정한다.
- #15 최종 test 평가는 완료됐다. `wildchat/finalize.py freeze/evaluate` 재실행·test 재접근·결과를 본 뒤 재튜닝하지 않는다. 고정 코드·해시·설정·로그는 보존한다.

## 7. 난이도 → 모델
| difficulty | 모델 | reasoning effort |
|---|---|---|
| low | light 모델 | low |
| medium | 기본 모델 | medium |
| high 또는 2회 실패 후 | strongest 모델 | high/xhigh |
box 자동화의 모델 이름은 저장소 밖 설정에서 관리하며 이 작업을 위해 해당 설정/자격증명을 읽지 않는다. 로컬 세션은 현재 제공된 모델을 사용하고 difficulty만으로 임의 전환·하위 에이전트 생성을 하지 않는다.

## 8. 검증 명령과 기존 한계
- `python3 scripts/verify.py`: 표준 라이브러리만 사용. JSON/ipynb 파싱과 Markdown 상대 링크의 존재를 검사하며 모델 성능·외부 URL·노트북 재현을 검증하지는 않는다.
- `python3 -m unittest discover -s wildchat/tests`: 합성 자료 기반 기존 테스트. NumPy·LightGBM·PyArrow·tiktoken 부재 시 관련 테스트는 skip된다. 실제 실험을 검증했다고 대신 보고하지 않는다.
- WildChat 의존성이 준비된 환경의 명령·선택 테스트는 [wildchat/README.md](wildchat/README.md)를 따른다. 문서 갱신을 위해 의존성을 새로 설치하거나 학습·최종 평가를 재실행할 필요는 없다.
- 2026-11-03 점검에서 TraceLab 외부 문서 사본의 upstream 기준 상대 링크 8건이 로컬에 없어 전체 verify가 실패했다. 2026-11-06 #23 검증 보완에서 `eda/agent_transition/tracelab/source/LICENSE-DATASET.md`와 `release_README.md`의 해당 링크 및 같은 LICENSE 대상의 배지 링크 1건(총 9건)을 기록된 릴리스 커밋 `fc06157`의 원본 URL로 교체하고 로컬 수정 안내를 붙였다. [provenance.json](eda/agent_transition/provenance.json)의 해당 `files` 항목에서 현재 사본의 bytes/SHA256을 기록하고, 수정 전 bytes/SHA256·출처 URL은 `upstream_copy`, 수정 이유는 `local_modification`에 구분해 보존한다. 검증 스크립트의 검사 범위는 유지하며 실제 실행 결과를 보고한다.
