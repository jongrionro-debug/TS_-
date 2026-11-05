# 에이전트 데이터 조사 재현

TraceLab·Exgentic 후보의 **실행된 조사·예비 실험**을 확인하는 위치다. 현재 연구는 [WildChat](../../docs/project.md)이며, 아래 조사 결과를 현재 채택 데이터·확정 가설로 해석하지 않는다. 채택하지 않은 과거 제안은 Git 이력에서 확인한다.

## 출처와 결과

- [provenance.json](provenance.json): 출처·버전·라이선스·확보 범위·원문 해시. `source/` 파일은 이 출처 대조에 사용한다.
- TraceLab: [품질 집계](tracelab/results/audit.json), [시간·구성 진단](tracelab/results/temporal.json), [예비 실험 설정](tracelab/results/pilot_config.json), [지표](tracelab/results/pilot_metrics.csv), [독립 검증](tracelab/results/verification.json).
- Exgentic: [표본 품질](exgentic/results/audit.json), [요청 단위 점검](exgentic/results/request_unit_audit.json), [내부 호출 점검](exgentic/results/next_output_audit.json), [시간 구조 점검](exgentic/results/temporal_structure_audit.json).

TraceLab의 시간은 사용량 보고 기준이며 서버 요청 도착시각이 아니다. Exgentic은 벤치마크 실행·내부 호출 표본이며 실제 서비스 수요와 같다고 보장하지 않는다. 조사·구조 검증과 일반화된 예측 성능을 구분한다.

## 재현 진입점

| 목적 | 코드 |
|---|---|
| TraceLab 품질·시간·토큰 | [audit.py](tracelab/audit.py), [temporal.py](tracelab/temporal.py) |
| 보고 시각 기준 예비 실험 | [pilot_report_clock.py](tracelab/pilot_report_clock.py) |
| 저장 지표·정답·특징 재계산 | [verify.py](tracelab/verify.py) |
| Exgentic 표본·요청 단위 | [audit_sample.py](exgentic/audit_sample.py), [audit_request_units.py](exgentic/audit_request_units.py) |
| Exgentic 내부 호출·시간 구조 | [audit_next_output.py](exgentic/audit_next_output.py), [audit_temporal_structure.py](exgentic/audit_temporal_structure.py) |

환경은 예비 실험 설정의 패키지 버전을 확인한다. 기존 실행은 `/opt/anaconda3/bin/python3`을 사용했으며 다른 환경에 같은 경로가 있다고 가정하지 않는다. 재생성 도구는 기존 결과를 덮어쓸 수 있으므로 요청된 실행 범위를 확인한다. 이번 문서 구조 변경에서는 조사·학습·평가를 재실행하지 않았다.

외부 원문 사본의 상대 링크 8건은 원래 upstream 구조를 가리키며 로컬에 없다. 전체 문서 검사의 기존 실패와 원문 해시 보존 범위는 [workflow.md](../../docs/workflow.md#8-검증-명령과-기존-한계)를 따른다.
