# 12·13페이지 직접 수정용 시안

기존 발표의 16:9 비율, 크림색 배경, 짙은 제목, 흰 그래프 영역을 기준으로 만든 참고 시안이다. 그래프와 통계표의 제목·축·범례·단위는 영어로 통일하고 보조 주석은 슬라이드에서 제거했다. 분석 수치와 기획은 `docs/prd.md`를 따르며 원래 발표 PDF는 편집하지 않았다.

## 파일

- `slide_12.png`: 12페이지 전체 배치 참고 이미지.
- `slide_13.png`: 13페이지 전체 배치 참고 이미지.
- `slides_12_13_reference.pdf`: 위 두 장을 묶은 PDF. 제목과 본문을 텍스트로 선택할 수 있다.
- `assets/12_one_day.png`: 왼쪽 하루 그래프. 같은 이름의 SVG도 제공한다.
- `assets/12_full_period.png`: 오른쪽 약 110일 그래프. 같은 이름의 SVG도 제공한다.
- `assets/13_input_by_observed_order.png`: 대화 내 관측 요청 순서별 입력 길이. 같은 이름의 SVG도 제공한다.
- `assets/13_conversation_statistics.png`: 영어 대화 통계표.
- `assets/13_horizon_comparison.png`: 영어 5·10·15분 비교표.
- `charts_english.zip`: 위 그래프·표를 묶은 파일. 그래프는 PNG·SVG, 표는 PNG로 제공한다.

## 배치

- 배경 `#FFF6EA`, 제목·본문 `#3B332F`, 입력 선 `#267485`, 출력 선 `#BD6547`.
- 기존 발표 제목 글꼴을 유지한다. 시안에서는 비슷한 형태의 Arial Narrow Bold와 Apple SD Gothic Neo를 사용했다.
- 양쪽 바깥 여백 약 4.4%, 두 열 간격 약 2.2%, 각 열 너비 약 44.4%.
- 제목과 부제는 위쪽 약 20%, 그래프와 표는 가운데, 결론과 설계 반영은 아래쪽에 배치한다.
- 12페이지는 하루 변동을 읽기 쉽도록 왼쪽 y축을 확대하고 양쪽 눈금으로 범위를 표시한다. 부가적인 주석 문장은 넣지 않는다. 앞서 생성한 `12_target_timeseries_raw.png`는 좌우 같은 축 범위로 비교하는 별도 분석용 그림이다.

## 12페이지 붙여넣을 문구

제목:

> EDA ① TARGET DEMAND

부제:

> 다음 10분 누적 토큰 수요: 하루의 변동과 전체 기간의 수준 차이

왼쪽 그래프:

> 10-min token demand: one day

- Y axes: `Input tokens`, `Final output tokens`
- X axis: `Hour of day (UTC+8)`

오른쪽 그래프:

> 10-min token demand: 110 days

- Y axes: `Input tokens`, `Final output tokens`
- X axis: `Relative day`
- Legend: `Daily mean`

하단 핵심 문장:

> 토큰 수요는 하루 안에서도 변동하며, 전체 기간에서도 수준 차이가 있다.

설계 반영:

> 최근 수요 이력과 하루 중 시각을 공통 입력으로 사용하고, 예측 효과는 시간순 검증으로 평가한다.

## 13페이지 붙여넣을 문구

제목:

> EDA ② CONVERSATIONS

부제:

> 반복 대화와 상태 정보를 추가하고, 후속 간격을 참고해 10분 지평을 선택한다.

왼쪽 위 표:

| Conversation statistics | Value |
|---|---:|
| Unique conversation IDs | 55,920 |
| Conversations with 2+ requests | 64.53% |
| Requests in repeated conversations | 91.51% |
| Median observed follow-up gap | 131 sec |

왼쪽 아래 제목:

> Horizon comparison

비교 표에서 10분 행을 옅은 색으로 강조:

| Horizon | Follow-up gaps ≤ horizon (%) | Zero-demand windows (%) |
|---|---:|---:|
| 5 min | 66.1% | 22.3% |
| 10 min | 76.7% | 15.8% |
| 15 min | 81.6% | 12.6% |

표 아래 문장:

> 10분은 후속 요청 포괄 범위와 단기성을 고려한 설계 선택이다.

오른쪽 그래프 제목:

> Input length by observed request order

- Y axis: `Median input tokens`
- X axis: `Observed request order`

오른쪽 설명:

> 대화 상태를 추가할 동기
>
> 관측 순서에 따라 입력 길이의 중앙값이 다르다.
> 최근 대화의 수와 상태를 예측 후보로 사용한다.

하단 설계 반영:

> B　A + 최근 5·15·30분 활동 대화 수
>
> C　B + 최근성·관측 횟수·관측 연령·마지막 입력 길이·입력 길이 변화 요약

## 분석 메모와 재현

아래 내용은 슬라이드에 표시하지 않고 분석 기록으로 보관한다.

- 목표값은 매 1분 시점에서 계산한 다음 10분 도착 요청의 입력·최종 출력 토큰 합이며 인접 구간은 9분 겹친다. 출력은 구간 내 실제 생성량과 구분한다.
- 하루 그림은 첫 60% 기간에서 입력 목표 일평균의 중앙값에 가장 가까운 완전 관측일인 관측 6일째를 사용한다.
- 검은 선은 10분 목표값의 일별 평균이며 일별 총토큰이 아니다. 첫·마지막 부분 관측일은 일평균선에서 제외한다.
- 좌우 그래프의 y축 범위는 다르며 실제 눈금에 표시한다.
- 후속 간격 비율의 분모는 후속 요청이 관측된 쌍 177,697개다. 전체 요청의 재방문 확률이 아니다.
- 0 목표 비율은 5·10·15분 모두 완전한 동일 158,323개 시점에서 계산했고 입력·출력의 비율은 같다.
- 10분이 최적 지평이거나 예측 성능이 더 우수하다는 결론은 내리지 않는다.
- 요청 순서 그래프는 관측 순서별 집단 중앙값이며 모든 개별 대화가 동일하게 증가한다는 결론과 구분한다.

수치·선정 기준은 상위 폴더의 `checks.json`, `horizon_context.csv`, `readme.md`에 기록했다. 13페이지의 대화 통계는 `06_session_quality.csv`, 간격 중앙값은 `06_followup_gap_distribution.csv`, 선 그래프는 `06_input_length_by_observed_turn.csv`에서 가져왔다.

프로젝트 루트에서 다음 명령으로 다시 생성한다.

```sh
/opt/anaconda3/bin/python output/presentation/graph_revision_proposal/build_slide_mockups.py
```

새 PDF의 페이지 수·16:9 비율·핵심 수치·텍스트 영역 경계를 확인하고, 삭제 대상 주석이 없는지와 최종 PNG를 직접 검토한다.
