# 작업 절차 — geass 이슈 흐름

최종 갱신: 2026-11-08. 결정 이유는 [ADR 0006](adr/0006-geass-issue-workflow.md). 하네스 자체의 동작은 geass 저장소(`~/Desktop/geass`, jongrionro-debug/geass)의 README를 따른다.

## 1. 흐름

```
geass plan "목표"      Claude가 이슈 초안 작성 → 라벨 없는 이슈 생성
(사람)                 본문 확인·수정 후 harness 라벨 부착 = 이 이슈의 브랜치 커밋·푸시·PR 승인
geass run <번호>       worktree(geass/issue-<번호>) → Codex 구현 → .harness/verify.toml 검증 → Claude 검수
                       APPROVE → 하네스가 커밋·푸시·PR(Closes #번호)
                       시도 초과·NEEDS_HUMAN → 이슈에 사유 댓글 후 멈춤 → 댓글로 답하고 run 재실행
(사람)                 PR 검토·병합
```

Claude Code에서는 `/geass doctor`, `/geass plan "목표"`, `/geass run <번호>`, `/geass status <번호>`, `/geass cleanup`을 쓴다.

## 2. 규칙

- 1 이슈 = 1 브랜치(`geass/issue-<번호>`) = 1 PR. 기준 브랜치는 `main`이다.
- `harness` 라벨 부착·PR 병합·이슈 닫기는 사람만 한다.
- 검증 명령은 기준 커밋의 [`.harness/verify.toml`](../.harness/verify.toml)에 고정된다. 에이전트는 `.harness/`를 바꿀 수 없으며, 바꾸려면 별도 사람 작업 PR로 한다.
- 이슈 본문에는 Goal, 바꿀 동작·지킬 동작, 완료 조건(검증 방법 포함), 제외 범위를 적는다. 대용량 다운로드·학습·최종 test처럼 비용·되돌릴 수 없는 단계는 이슈에 명시된 경우만 수행한다.
- 실제 trace가 필요한 실행(다운로드·학습)은 검증 명령에 넣지 않는다. 검증은 합성 fixture 테스트로 하고, 실제 실행 결과는 산출물과 PR 본문에 명령·환경과 함께 남긴다.
- 하네스가 멈췄거나 사람이 개입했으면 geass 저장소에 `failure` 이슈를 남길지 결정한다.

## 3. 사람이 직접 하는 작업

하네스 밖의 수동 작업(이번 재구성처럼 `.harness/` 변경 포함)은 `agent/<slug>` 브랜치와 PR로 하고, 커밋·푸시·PR·병합은 해당 요청에서 승인된 범위만 수행한다.

## 4. 검증 명령과 한계

- `python3 scripts/verify.py`: 표준 라이브러리만 사용. JSON 파싱, Markdown 상대 링크, 문서 구조([check_context.py](../scripts/check_context.py))를 검사한다.
- `python3 -m unittest discover -s tests`: 합성 fixture 테스트. 분석 의존성(pandas·pyarrow·lightgbm 등)이 없는 Python에서는 관련 테스트가 skip되며 skip 수를 보고한다.
- geass 검증은 로컬 Python에서 실행된다. 분석 의존성 고정은 첫 코드 Issue에서 `requirements.txt`로 정한다.
