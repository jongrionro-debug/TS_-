# 0006 · 작업 흐름을 geass 이슈 하네스로 바꾼다

- Status: Accepted
- Date: 2026-11-08
- Related Work: 2026-11-08 사용자 요청
- Approval: 2026-11-08 사용자 요청("/geass 방식을 활용해 issue 기반 작업 플로우로 변경")
- Supersedes: [ADR 0001](0001-agent-workflow.md)의 box worker·`agent:*` 라벨 상태 기계·handoff JSON. Issue를 작업 원본, PR을 결과로 삼는 원칙은 유지한다.

## 맥락
기존 흐름은 box의 외부 자동화와 `agent:*` 라벨 상태 기계에 의존했으나 현재 가동을 보장할 수 없고, 로컬 작업과 이슈 작업의 경계가 문서마다 달랐다. geass는 이슈 하나를 Codex 구현 → 고정 검증 → Claude 검수 → PR로 돌리는 최소 하네스다.

## 결정
- 작업은 `geass plan`으로 이슈를 만들고, 사람이 `harness` 라벨을 붙인 이슈만 `geass run`으로 실행한다. 라벨은 그 이슈 브랜치의 커밋·푸시·PR 승인이다.
- 검증 명령은 main의 `.harness/verify.toml`에 고정한다. 병합·이슈 닫기는 사람이 한다.
- 절차는 [workflow.md](../workflow.md)에 둔다.

## 대안
- box 자동화 유지: 가동 여부를 저장소에서 확인할 수 없다.
- 로컬 채팅만으로 진행: 작업·검증 기록이 이슈·PR에 남지 않는다.

## 영향
- 검증이 합성 fixture 테스트 중심이 되므로 첫 코드 Issue에서 테스트 구조와 의존성 고정을 정해야 한다.
- `agent:*`, `difficulty:*` 라벨은 더 쓰지 않는다. 기존 이슈의 라벨은 이력으로 둔다.
