# 0007 · 작업 흐름을 다시 geass 이슈 하네스로 바꾼다

- Status: Accepted
- Date: 2026-11-08
- Related Work: 2026-11-08 사용자 요청, Issue #28·#29·#30
- Approval: 2026-11-08 사용자 요청("기어스로 해당 task들 다 해볼려고", 세 곳 수정 승인)
- Supersedes: [ADR 0006](0006-issue-flow-skill.md) 전체. Issue 원본·PR 결과 원칙(ADR 0001)은 유지한다.

## 맥락
ADR 0006은 판단이 큰 연구 저장소라 자동 루프보다 대화형 스킬이 맞다고 보고 issue-flow를 택했다. 이후 geass와 issue-flow를 같은 이슈로 비교하려 했으나(jongrionro-debug/geass#20), 사용자가 비교 없이 geass를 직접 써 보기로 했다.

## 결정
- 작업은 `geass plan`으로 이슈를 만들고, 사용자가 `harness` 라벨을 붙인 이슈만 `geass run`으로 실행한다. 라벨은 그 이슈 `geass/issue-<번호>` 브랜치의 커밋·푸시·PR 승인이다. 병합·이슈 닫기는 사람이 한다.
- 검증 명령은 main의 `.harness/verify.toml`에 고정한다.
- 실데이터는 `FREEINFERENCE_DATA_DIR`로 worktree 밖에서 읽고, 1~13주(train·validation)만 내려받아 둔다.
- issue-flow 스킬은 제거한다(Git 이력에 남는다). 절차는 [workflow.md](../workflow.md).

## 대안
- issue-flow 유지(ADR 0006): 단계별 확인은 쉽지만 사용자가 geass로 진행하기로 했다.
- geass·issue-flow 병행 비교: 이슈마다 두 번 구현·검수하는 비용 때문에 중단했다.

## 영향
- 검수 품질은 geass의 Claude 검수에 의존한다. geass #17·#18처럼 검수가 결함을 승인한 사례가 있어 PR 병합 전 사람 확인이 필요하다.
- 시점 누출·test 접근은 검수와 로더의 기본 거부(#28)로 막는다. 14~16주 데이터는 내려받지 않는다.
- 하네스가 멈추거나 사람이 개입하면 geass 저장소에 `failure` 이슈를 남길지 결정한다.
