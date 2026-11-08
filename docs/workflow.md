# 작업 절차 — Issue→PR 스킬 (Opus 기획·Codex 구현·Opus 검수)

최종 갱신: 2026-11-08. 결정 이유는 [ADR 0006](adr/0006-issue-flow-skill.md). 단계별 실행 지침은 [issue-flow 스킬](../.claude/skills/issue-flow/SKILL.md)이다.

## 1. 흐름

```
/issue-flow plan "목표"     Opus가 Issue 초안 → 사용자 확인 → 게시
/issue-flow implement <N>   브랜치 agent/issue-<N>-<slug> → codex exec 구현
/issue-flow review <N>      Opus가 검증 재실행 + diff·완료 조건 검수
                              CHANGES → 지적을 Codex에 넘겨 재구현(최대 3회차)
                              APPROVE → ship
/issue-flow ship <N>        사용자 승인 → 커밋·푸시·PR(Closes #N)
(사용자)                     PR 확인 → 병합 승인
```

## 2. 규칙

- 1 Issue = 1 브랜치(`agent/issue-<번호>-<slug>`) = 1 PR. 기준 브랜치는 `main`. Issue 없는 수동 작업은 `agent/<slug>`.
- Claude는 Issue 코드를 직접 구현하지 않고 기획·검수·커밋·PR을 맡는다. Codex는 커밋·푸시하지 않는다.
- Issue 게시, 커밋·푸시·PR, 병합은 각 단계에서 사용자 승인 후 수행한다. 병합은 항상 별도 승인.
- Issue 본문에는 Goal, 바꿀 동작·지킬 동작, 완료 조건(확인 방법 포함), 제외 범위를 적는다. 대용량 다운로드·학습·최종 test처럼 비용·되돌릴 수 없는 단계는 Issue에 명시된 경우만 수행한다.
- 실제 trace가 필요한 실행 결과는 산출물과 PR 본문에 명령·환경과 함께 남긴다. 테스트는 합성 fixture로 한다.

## 3. 검증 명령과 한계

- `python3 scripts/verify.py`: 표준 라이브러리만 사용. JSON 파싱, Markdown 상대 링크, 문서 구조([check_context.py](../scripts/check_context.py))를 검사한다.
- `python3 -m unittest discover -s tests`: 합성 fixture 테스트. 분석 의존성(pandas·pyarrow·lightgbm 등)이 없으면 관련 테스트는 skip되며 skip 수를 보고한다. 의존성 고정은 첫 코드 Issue에서 `requirements.txt`로 정한다.
- 검증 명령이 기준 커밋에 고정되지 않으므로, 검수자는 테스트 삭제·완화·skip 추가를 diff로 확인한다. 같은 명령이 CI에서도 실행된다.
