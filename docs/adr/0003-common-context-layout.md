# 0003 · 세 저장소의 맥락 문서 경로와 역할을 통일한다

- Status: Accepted
- Date: 2026-11-04
- Related Work: 현재 대화의 HCI_HANDSOME·TS_정확한·Hci_Korea 공통 문서 구조·규칙 요청
- Approval: 사용자가 세 레포의 안내·맥락 폴더·파일명과 규칙을 통일하고 Issue·PR·ADR을 함께 유지하도록 요청함
- Supersedes: [ADR 0001](0001-agent-workflow.md)의 `WORKFLOW.md`·`docs/decisions/` 경로와 별도 0000 템플릿만 대체. Issue→PR·box 인계 의미는 유지한다.

## 맥락

같은 역할의 문서가 master·prd·INDEX·활성 계획 등 여러 이름으로 생기고, ADR 위치와 작업 절차가 달라 현재 기준과 작업 기록을 찾기 어려웠다.

## 결정

1. 공통 경로·역할은 [문서 안내](../README.md), 작업 규칙은 [AGENTS](../../AGENTS.md)를 따른다. README는 시작, AGENTS는 규칙, docs/README는 안내, project는 현재 상태, ADR은 결정 이유를 맡는다.
2. Issue는 할 일·완료 조건, PR은 변경·실제 검증 결과를 관리한다. 같은 내용을 로컬 Issue/PR 사본·작업 보고서로 늘리지 않는다. 게시하지 않는 작업은 현재 요청·결과로 추적한다.
3. 상세 문서는 architecture·evaluation·workflow와 specs·sources만 필요할 때 사용한다. 코드·논문·실험 산출물의 기존 재현 계약은 유지한다.
4. 공통 구조 검사 `scripts/check_context.py`를 기존 검사·CI에 연결한다. 파일 추가로 규칙을 우회하지 않고 공통 구조 변경 요청과 ADR로 기준 자체를 갱신한다.

## 대안

- 레포마다 독자 명명: 현재 혼란을 반복하므로 채택하지 않는다.
- 모든 상세·연구 자료를 한 파일에 병합: 독립 계약·고정 근거의 역할이 사라지므로 채택하지 않는다.
- 세션마다 context·plan·summary 파일 추가: 정본이 다시 분산되므로 채택하지 않는다.

## 영향

`readme.md`를 `README.md`, `prd.md`를 `project.md`, `WORKFLOW.md`를 `docs/workflow.md`, `decisions/`를 `adr/`로 통일했다. 과거 BurstGPT 미실행 설계 부록·`제안1.md`·ADR로 잘못 분류된 `가설1.md`·0000 템플릿은 현재 정본에서 제거했다. 현재 EDA의 실행 안내·결과와 후보 조사의 재현 진입점은 유지한다. 제거 전 내용은 Git `ab66dc9`에서 복구한다. 고정 코드·설정·결과·해시·로그·외부 출처 원문은 변경하지 않는다.

이동 참조·앵커와 기존 검증을 확인한다. 구조 검사는 미래 파일명·경로의 이탈을 검출하며 문서 내용의 일치나 연구 품질까지 보장하지 않는다. 실제 원격 CI·브랜치 보호 적용은 별도 확인 대상이다.
