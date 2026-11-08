# 0006 · 작업 흐름을 Opus 기획·Codex 구현·Opus 검수의 Issue→PR 스킬로 바꾼다

- Status: Accepted
- Date: 2026-11-08
- Related Work: 2026-11-08 사용자 요청
- Approval: 2026-11-08 사용자 요청("하네스로는 안 잡고 skill로", geass 미사용)
- Supersedes: [ADR 0001](0001-agent-workflow.md)의 box worker·`agent:*` 라벨 상태 기계·handoff JSON. Issue를 작업 원본, PR을 결과로 삼는 원칙은 유지한다.

## 맥락
기존 흐름은 box의 외부 자동화와 `agent:*` 라벨 상태 기계에 의존했으나 현재 가동을 보장할 수 없었다. geass 하네스도 검토했으나, 이 연구 저장소는 이슈마다 판단(시점 누출·평가 범위)이 크고 사람이 단계마다 결과를 보고 싶어 하므로 자동 루프보다 대화 안에서 진행하는 절차가 맞다.

## 결정
- 프로젝트 스킬 `.claude/skills/issue-flow`(`/issue-flow plan|implement|review|ship|status`)로 진행한다.
- 기획(Issue 초안)과 검수는 Claude Opus, 구현은 `codex exec`. 검수 지적 수정도 Codex에 다시 맡긴다(최대 3회차).
- Issue 게시, 커밋·푸시·PR은 해당 단계에서 사용자 승인 후 Claude가 한다. 병합은 항상 별도 승인.
- 검증은 `scripts/verify.py`와 `tests/` unittest를 검수자가 직접 재실행하고 CI에서도 실행한다. geass·`.harness/`는 쓰지 않는다.
- 절차는 [workflow.md](../workflow.md).

## 대안
- geass 하네스: 검증 고정·자동 PR은 장점이지만 단계별 확인이 어렵고 이 저장소에는 과하다.
- box 자동화 유지: 가동 여부를 저장소에서 확인할 수 없다.
- Claude 단독 구현: 구현·검수 분리가 사라진다.

## 영향
- 검증 명령이 기준 커밋에 고정되지 않으므로 검수자가 테스트 삭제·완화를 diff로 확인해야 한다.
- 이 세션이 Codex를 직접 실행하므로 Codex 로그인과 네트워크가 필요하다.
- `agent:*`, `difficulty:*` 라벨은 더 쓰지 않는다. 기존 이슈의 라벨은 이력으로 둔다.
