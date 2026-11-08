# ADR-0001: GitHub Issue 기반 에이전트 작업 흐름 도입
- 상태: 채택 (Phase 0 커밋 `21132c8` 반영 완료; 현재 절차는 아래 갱신 참고)
- 후속 결정: [ADR 0003](0003-common-context-layout.md)이 공통 문서 경로·명명을, [ADR 0006](0006-geass-issue-workflow.md)이 box worker·라벨 상태 기계·handoff를 대체한다(2026-11-08). Issue 원본·PR 결과 원칙은 유지한다.
- 날짜: 2026-11-02
- 관련: PR "Phase 0: agent workflow scaffolding"

## 맥락
코딩 에이전트(Codex)에게 작업을 맡길 때 작업 정의·상태·결과가 대화나 Notion에 흩어져 재현과 리뷰가 어려웠다.

## 도입 당시 결정
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

## 현재 적용 상태 (2026-11-03 갱신)
- 이 문서는 box 자동화 도입 이력이다. 현재 로컬 협업과 기계 worker의 적용 범위는 [WORKFLOW.md](../workflow.md)를 따른다.
- 2026-10-30 사용자 위임에 따라 Git 요청은 필요한 커밋·push·PR·병합·기본 브랜치 반영까지 포함한다. 위 도입 당시의 “머지는 사람만”은 현재 제한으로 적용하지 않는다.
- 로컬 채팅 요청은 이슈 생성 없이 진행하며 기존 작업을 보존한다. 자동 Notion 카드 생성이나 별도 worktree를 요구하지 않는다.
- WildChat에 `unittest` 기반 테스트와 고정 의존성이 추가됐다. “테스트가 없다”는 위 결과는 도입 당시 상태이며 현재 검증 명령·환경 한계는 WORKFLOW.md §8을 따른다.
