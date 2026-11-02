# WORKFLOW.md — 이슈→PR 실행 계약

## 1. 상태 (GitHub 라벨이 원본)
agent:triage → agent:ready → agent:in-progress → agent:review → (사람 merge) → closed
                       ↘ agent:needs-decision / agent:blocked
- ready 조건: Goal·AC 존재, mode·difficulty 지정, Mac 필요 아님
- in-progress: worker가 잡은 순간 라벨 변경 + 코멘트(세션·브랜치)
- review: PR 생성 + 검증 결과 확정
- needs-decision: 사람 선택 필요 → Notion Agent Inbox에 카드 생성
- blocked: 3회 재시도 실패 / 환경 문제
- agent:needs-mac: box에서 실행 불가 (Mac 필요) → worker가 잡지 않음
- difficulty:low|medium|high: 모델·reasoning effort 선택 기준 (§7)

## 2. 실행 단위
- 1 이슈 = 1 브랜치(agent/issue-<n>-<slug>) = 1차 PR, 한 번에 한 이슈만
- box의 저장소 클론은 1개(/workspace/repos/<repo>), worktree 사용 안 함
- 착수 조건: 작업 트리 clean (아니면 중단·보고) → 기본 브랜치 checkout → pull → 브랜치 생성
- Codex 세션은 교체 가능, 이슈가 작업 정체성. 세션 ID는 state와 PR 메타데이터에 기록

## 3. 구현 절차
1. AGENTS.md → 이슈 → relevant_docs → 파일 탐색
2. 계획 5줄 이내 작성 (handoff.plan)
3. 구현 → `python3 scripts/verify.py`
4. 실패 시 최대 3회 수정 (codex exec resume)
5. PR 템플릿으로 PR 생성

## 4. 완료 정의 (DoD)
- 모든 AC에 검증 근거, verify 통과 (`python3 scripts/verify.py` exit 0)
- 범위 밖 변경 없음 또는 risks에 명시
- 문서 영향 있으면 docs/ 갱신
- 머지는 사람만

## 5. Handoff JSON (worker 최종 출력)
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

## 6. 금지
- 기본 브랜치 push, force-push(자기 브랜치 rebase 외), merge, 배포, 삭제성 변경, 외부 게시

## 7. 난이도 → 모델
| difficulty | 모델 | reasoning effort |
|---|---|---|
| low | light 모델 | low |
| medium | 기본 모델 | medium |
| high 또는 2회 실패 후 | strongest 모델 | high/xhigh |
모델 이름은 box의 /workspace/agent-ops/config/models.env에서 관리 (저장소에는 기록하지 않음).

## 8. 검증 명령
- `python3 scripts/verify.py` (표준 라이브러리만, 수 초 내 완료)
