# ADR-0001: GitHub Issue 기반 에이전트 작업 흐름 도입
- 상태: 제안 (Phase 0 PR 머지 시 채택)
- 날짜: 2026-11-02
- 관련: PR "Phase 0: agent workflow scaffolding"

## 맥락
코딩 에이전트(Codex)에게 작업을 맡길 때 작업 정의·상태·결과가 대화나 Notion에 흩어져 재현과 리뷰가 어려웠다.

## 결정
GitHub Issue(+agent:* 라벨)를 작업 원본으로, 저장소(AGENTS.md, WORKFLOW.md, docs/)를 지식 원본으로,
PR을 결과물로 삼는다. box의 work-issue.sh가 이슈 1건씩 Context Packet을 만들어 Codex를 실행하고,
`python3 scripts/verify.py`로 직접 검증한 뒤 PR을 연다. git worktree는 쓰지 않고 클론 1개에서
한 번에 한 이슈만 처리한다. 모델·effort는 이슈의 difficulty로 정한다. 머지는 사람만 한다.
기존 agents.md는 AGENTS.md로 이름을 바꾸고 내용을 §0에 보존했으며, "adr 폴더"는 docs/decisions/로 정한다.

## 대안
- A: Notion 카드 중심 — 사람이 보기 편하지만 코드·PR과 연결이 약함
- B: git worktree 병렬 — 처리량은 높지만 Phase 0에는 복잡도 대비 이득이 적음

## 결과
- 좋아지는 점: 작업·세션·PR이 이슈 번호로 추적됨, 검증 기준이 명시됨
- 감수하는 점: 테스트가 없어 verify는 형식 검사(JSON·링크) 수준. 노트북 재실행 검증은 데이터 부재로 불가
- AGENTS.md 반영: 완료
