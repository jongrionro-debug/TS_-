# 시간 구조 가설 VAL 실험

상태: 실제 VAL 학습·평가 완료. test.jsonl 미접근·test 정답 미사용. Seed 42.

기준선: #21 B (현재 길이+대화 내 길이 이력 31개). 동일 사용자 분리·시간 구간, 음수만 0 처리·분위수 무정렬.
하이퍼파라미터 31 leaves / min leaf 300, P10/P50/P90 403/244/279회 고정. 모든 H는 동일 기준선에 하나씩 추가하며 재튜닝하지 않았다.

B 기준선 재현: 과거 49.928499, 이번 49.928499, 재현 차이 +0.000000. 모든 가설 Δ는 이번 기준선에 대한 값이다.
C_emb_PCA k=32의 과거 46.062757은 임베딩 cache 부재로 재현하지 않았다. 사용자 지시에 따라 새 임베딩 생성 없이 H4_prompt_sequence도 건너뛰었다.

정제 train 81,307행/8,702사용자, VAL 12,920행/1,736사용자. 사용자·그룹·source 교집합 0.

- H1_sequence_lengths: 동일 대화의 이전 유효 응답 EWMA(α=.3), 누적 평균·표준편차, 마지막 두 응답 차이·턴당 기울기, 이전 출력/입력 비율·EWMA·현재 입력 배율·관측 수
- H2_reply_cadence: 현재 턴 이전 assistant 완료 시각끼리의 마지막 간격·누적 span·log1p와 결측 표시. 현재 요청까지의 대기시간이 아님
- H3_calendar_drift: 이전 assistant 완료 시각의 epoch day·월·UTC hour/weekday sin/cos와 hour×log1p(현재 입력). 첫 턴 결측

평균 P10/P50/P90 pinball (낮을수록 좋음). 사용자 가중은 각 구분 안의 사용자별 평균을 동일 가중.

| 턴 | 가중 | baseline | H1_sequence_lengths | H2_reply_cadence | H3_calendar_drift |
|---|---|---:|---:|---:|---:|
| all | row | 49.928499 | 49.780214 | 49.919513 | 50.063224 |
| all | user | 55.875767 | 55.785333 | 55.906758 | 56.003362 |
| first_turn | row | 72.124630 | 72.140488 | 72.173216 | 72.217982 |
| first_turn | user | 62.470312 | 62.459386 | 62.507725 | 62.550961 |
| followup_turn | row | 36.331568 | 36.082732 | 36.287314 | 36.491637 |
| followup_turn | user | 42.051782 | 41.732693 | 42.046458 | 42.187079 |

H−동일 기준선 차이 / 사용자 paired bootstrap 95% CI (1,000회·PCG64 seed 42·공유 추출·percentile 선형 보간):

| H | 턴 | 가중 | 차이 | 95% CI |
|---|---|---|---:|---|
| H1_sequence_lengths-baseline | all | row | -0.148285 | [-0.246487, -0.044299] |
| H2_reply_cadence-baseline | all | row | -0.008986 | [-0.072455, 0.057281] |
| H3_calendar_drift-baseline | all | row | 0.134725 | [0.073409, 0.197476] |
| H1_sequence_lengths-baseline | all | user | -0.090434 | [-0.187576, 0.005691] |
| H2_reply_cadence-baseline | all | user | 0.030991 | [-0.023507, 0.083077] |
| H3_calendar_drift-baseline | all | user | 0.127594 | [0.073918, 0.190938] |
| H1_sequence_lengths-baseline | first_turn | row | 0.015858 | [-0.045355, 0.072159] |
| H2_reply_cadence-baseline | first_turn | row | 0.048586 | [0.000563, 0.095765] |
| H3_calendar_drift-baseline | first_turn | row | 0.093353 | [0.019466, 0.168602] |
| H1_sequence_lengths-baseline | first_turn | user | -0.010927 | [-0.059754, 0.039689] |
| H2_reply_cadence-baseline | first_turn | user | 0.037412 | [-0.011280, 0.087927] |
| H3_calendar_drift-baseline | first_turn | user | 0.080649 | [0.025540, 0.137601] |
| H1_sequence_lengths-baseline | followup_turn | row | -0.248836 | [-0.397267, -0.090600] |
| H2_reply_cadence-baseline | followup_turn | row | -0.044254 | [-0.140044, 0.054649] |
| H3_calendar_drift-baseline | followup_turn | row | 0.160068 | [0.057492, 0.269941] |
| H1_sequence_lengths-baseline | followup_turn | user | -0.319090 | [-0.562586, -0.082333] |
| H2_reply_cadence-baseline | followup_turn | user | -0.005324 | [-0.108428, 0.087649] |
| H3_calendar_drift-baseline | followup_turn | user | 0.135297 | [0.024533, 0.251800] |

권장: **H1_sequence_lengths**. 전체 행 가중 CI가 0 아래인 후보 중 가장 낮은 점추정을 탐색적 권장으로 삼고, 없으면 기준선을 유지한다.
H1은 전체 행 가중과 후속 턴의 행/사용자 가중에서 작은 개선을 보였다. 전체 사용자 가중과 첫 턴의 H1 CI는 0을 포함한다. H2 전체 차이 CI는 0을 포함하고 H3는 손실이 증가했다.
시간 구조를 추가하려면 H1의 대화 내 수준·추세·출력/입력 비율 요약을 우선 권장한다. 묶음 특징 실험이므로 EWMA 하나의 효과로 분해할 수 없다. C_emb_PCA에 H1을 추가하는 효과는 아직 검증하지 않았다.
반복 사용한 VAL의 탐색 결과이며 다중비교 보정·새로운 일반화 증거는 아니다. 사용자 bootstrap은 고정 모델에 조건부이고 시간 블록·학습 불확실성을 반영하지 않는다.
입력 시각은 이전 assistant 완료의 하한이고 현재 요청 제출 시각이 아니다. 첫 턴에 시각이 없으므로 달력 특징을 제공하지 않는다. 모델 버전·대화 마지막 시각을 입력에 사용하지 않았다.

실행 시간(초): {"loading_features": 5.1502056659956, "baseline": 5.055709083972033, "H1_sequence_lengths": 5.434047458984423, "H2_reply_cadence": 4.916760041989619, "H3_calendar_drift": 5.459481166006299, "evaluation_bootstrap": 0.36037858299096115, "total": 26.379951374983648}

실행 환경: {"numpy": "2.5.4", "lightgbm": "4.7.0", "pyarrow": "26.0.1", "scipy": "1.18.1"}

파일: `wildchat/temporal_hypotheses.py`, `wildchat/temporal_causal.py`, `wildchat/temporal_baseline.json`, `wildchat/tests/test_temporal_hypotheses.py`, `wildchat/results/temporal_hypotheses.json`, `wildchat/results/temporal_hypotheses.md`, `wildchat/results/temporal_hypotheses_summary.md`.

검증·문제: 지정 /tmp/venv-temporal/bin/python으로 재실행. 설치 없음. 전체 합성 검사 50개 중 49개 PASS·1개 tokenizer 의존성 skip, 새 안전성 검사 8개 PASS. 구조·diff 형식 PASS. 공통 verify는 기존 TraceLab 링크 8건으로 FAIL. 기존 파일 18개 해시 유지.

기존 docs/project.md·wildchat/README.md는 사용자의 미커밋 변경 보존을 위해 수정하지 않았다. 이 VAL 전용 후속 실험의 정의·재현·결과는 요청된 두 결과 MD와 JSON에 둔다.

Notion: 기존 WildChat 실험 페이지의 쓰기는 직전 시도에서 승인 정책(approval policy: never)으로 차단됐다. 이번 실제 결과 요약도 미저장이며 완료된 로컬 실험과 구분한다. 미저장 인계안: “2026-11-06 VAL에서 B 재현, H1 대화 내 요약에 소폭 이득, H2 추가 개선 미확인·H3 손실 증가. H4는 cache 부재로 skip. 정본 wildchat/results/temporal_hypotheses.md 및 summary.md, 미커밋 로컬 산출물. 전체 사용자 가중 H1 CI는 0 포함. test 접근·Git 게시 없음.”

## 누출 방지·재현

- `rv.load_allowed`의 train.jsonl/validation.jsonl 고정 허용 목록과 `purge_and_check`, `inner_split` 재사용. 사용자·그룹·source 교집합 0을 실행 검사한다.
- #21 `HistoryRows`, `temporal_metadata`, `evaluate`를 재사용했다. 같은 metadata projection에서 불필요한 header/country/model 컬럼을 제외했다.
- 이전 유효 턴의 응답/프롬프트만 스냅샷 후 상태에 반영한다. 원본 source별 상태, 현재·미래 응답/다른 대화·다른 분할 미사용. H1 이력은 lag slot이 아닌 이전 유효 표본 전체다.
- raw shard에서 hashed_ip/recorded timestamp/message role·timestamp만 projection한다. Arrow metadata batch는 압축 단위로 다른 원본 행을 포함할 수 있지만, train/VAL source ID만 take한 뒤 Python 객체로 만든다. raw text/응답 target 컬럼과 test.jsonl은 요청하지 않는다.
- 모든 가설의 파라미터/반복 수는 기존 B의 train 내부 선택에 고정했다. 새 sweep·VAL 조기 종료·VAL 재튜닝이 없다. H1은 이전 유효 응답 전체, H2는 이미 끝난 두 응답 사이 cadence이며 현재 요청까지의 간격이 아니다.
- H1 EWMA는 첫 관측값으로 초기화하고 α=.3으로 갱신한다. std는 ddof=0, 차이·기울기는 직전 두 유효 응답 기준(기울기 분모는 원래 turn_index 차이). 출력/입력과 현재/이전 입력 비율의 분모는 max(이전 입력 토큰,1)이다.
- H2 gap은 마지막 두 과거 assistant 완료 시각 차이, span은 첫 과거 완료부터 마지막 과거 완료까지다. 현재 assistant 시간은 사용하지 않는다. H3는 anchor의 epoch day·월·UTC hour/weekday sin/cos와 hour×log1p 현재 입력이다.
- 첫 턴에는 과거 특징이 결측이지만 전체 모델을 재학습하므로 첫 턴 예측도 소폭 변할 수 있다. H3의 악화는 이 고정 특징·모델의 결과이며 시간 drift가 없다는 증명은 아니다.
- 전체 기계 판독 결과는 temporal_hypotheses.json: P10/P50/P90 개별 loss, coverage, crossing, split 검사, 입력/코드 해시, 환경·시간 포함.

문제·제한: 브랜치 생성이 sandbox의 .git 쓰기 제한으로 실패하여 main의 새 파일만 사용했다. 기존 사용자 변경을 수정하지 않았다. Notion/기존 정본 MD는 사용자 미커밋 파일 보존 제약에 따라 별도 인계한다.

```sh
/tmp/venv-temporal/bin/python '/Users/nojonghyeon/Desktop/TS_정확한/wildchat/temporal_hypotheses.py' --prepared /Users/nojonghyeon/data/wildchat-1m/derived/issue5-shard0 --source /Users/nojonghyeon/data/wildchat-1m/data/train-00000-of-00014.parquet --threads 4 --output /tmp/temporal-val-iuiufx3q
```

```json
{
  "outer": {
    "cutoff": "2023-04-27T13:14:56+00:00",
    "before": {
      "earlier": {
        "samples": 115500,
        "users": 9403,
        "groups": 36160,
        "conversations": 39689,
        "first_turn_samples": 39497,
        "recorded_at_min": "2023-04-09T00:02:53+00:00",
        "recorded_at_max": "2023-04-27T13:12:41+00:00"
      },
      "later": {
        "samples": 22968,
        "users": 2415,
        "groups": 7755,
        "conversations": 8329,
        "first_turn_samples": 8300,
        "recorded_at_min": "2023-04-27T13:14:56+00:00",
        "recorded_at_max": "2023-05-01T01:30:03+00:00"
      }
    },
    "shared_users_before_purge": 672,
    "later_rows_with_shared_user_before_purge": 10021,
    "removed_original_groups": 13707,
    "removed_group_reasons_nonexclusive": {
      "missing_user": 0,
      "crosses_time_boundary": 0,
      "shared_user": 13707
    },
    "removed_rows": {
      "earlier": 34193,
      "later": 10048
    },
    "after": {
      "earlier": {
        "samples": 81307,
        "users": 8702,
        "groups": 25593,
        "conversations": 28041,
        "first_turn_samples": 27887,
        "recorded_at_min": "2023-04-09T00:03:20+00:00",
        "recorded_at_max": "2023-04-27T13:12:41+00:00"
      },
      "later": {
        "samples": 12920,
        "users": 1736,
        "groups": 4615,
        "conversations": 4930,
        "first_turn_samples": 4908,
        "recorded_at_min": "2023-04-27T13:14:56+00:00",
        "recorded_at_max": "2023-05-01T01:30:03+00:00"
      }
    },
    "checks": {
      "user_intersection": 0,
      "group_intersection": 0,
      "source_intersection": 0,
      "strict_chronology": true,
      "missing_users": 0
    }
  },
  "inner": {
    "cutoff": "2023-04-24T02:55:51+00:00",
    "before": {
      "earlier": {
        "samples": 65395,
        "users": 6840,
        "groups": 20474,
        "conversations": 22407,
        "first_turn_samples": 22286,
        "recorded_at_min": "2023-04-09T00:03:20+00:00",
        "recorded_at_max": "2023-04-24T02:52:53+00:00"
      },
      "later": {
        "samples": 15912,
        "users": 2340,
        "groups": 5161,
        "conversations": 5634,
        "first_turn_samples": 5601,
        "recorded_at_min": "2023-04-24T02:55:51+00:00",
        "recorded_at_max": "2023-04-27T13:12:41+00:00"
      }
    },
    "shared_users_before_purge": 478,
    "later_rows_with_shared_user_before_purge": 5236,
    "removed_original_groups": 4995,
    "removed_group_reasons_nonexclusive": {
      "missing_user": 0,
      "crosses_time_boundary": 42,
      "shared_user": 4970
    },
    "removed_rows": {
      "earlier": 12721,
      "later": 5388
    },
    "after": {
      "earlier": {
        "samples": 52674,
        "users": 6328,
        "groups": 16968,
        "conversations": 18551,
        "first_turn_samples": 18449,
        "recorded_at_min": "2023-04-09T00:03:20+00:00",
        "recorded_at_max": "2023-04-24T02:50:40+00:00"
      },
      "later": {
        "samples": 10524,
        "users": 1837,
        "groups": 3630,
        "conversations": 3961,
        "first_turn_samples": 3932,
        "recorded_at_min": "2023-04-24T02:55:51+00:00",
        "recorded_at_max": "2023-04-27T13:12:41+00:00"
      }
    },
    "checks": {
      "user_intersection": 0,
      "group_intersection": 0,
      "source_intersection": 0,
      "strict_chronology": true,
      "missing_users": 0
    }
  }
}
```

## 메타데이터·특징 감사

```json
{
  "projected_columns": [
    "hashed_ip",
    "timestamp",
    "conversation.list.element.role",
    "conversation.list.element.timestamp"
  ],
  "selected_sources": 48018,
  "user_messages": 139495,
  "user_timestamps_present": 0,
  "assistant_timestamps_missing": 0,
  "raw_text_or_target_columns": false,
  "python_metadata_sources_outside_allowlist": 0
}
```

```json
{
  "train": {
    "H1_sequence_lengths": {
      "features": 40,
      "first_turn_rows": 27887,
      "anchor_present_rows": 53420,
      "new_numeric_nonmissing_rows": 53266
    },
    "H2_reply_cadence": {
      "features": 36,
      "first_turn_rows": 27887,
      "anchor_present_rows": 53420,
      "new_numeric_nonmissing_rows": 38496
    },
    "H3_calendar_drift": {
      "features": 39,
      "first_turn_rows": 27887,
      "anchor_present_rows": 53420,
      "new_numeric_nonmissing_rows": 53420
    }
  },
  "validation": {
    "H1_sequence_lengths": {
      "features": 40,
      "first_turn_rows": 4908,
      "anchor_present_rows": 8012,
      "new_numeric_nonmissing_rows": 7990
    },
    "H2_reply_cadence": {
      "features": 36,
      "first_turn_rows": 4908,
      "anchor_present_rows": 8012,
      "new_numeric_nonmissing_rows": 5739
    },
    "H3_calendar_drift": {
      "features": 39,
      "first_turn_rows": 4908,
      "anchor_present_rows": 8012,
      "new_numeric_nonmissing_rows": 8012
    }
  }
}
```

## 표본 수·분위수별 pinball

| 턴 | 가중 | 조건 | 행/사용자 | P10 loss | P50 loss | P90 loss |
|---|---|---|---|---:|---:|---:|
| all | row | baseline | 12920/1736 | 26.082470 | 76.950577 | 46.752451 |
| all | row | H1_sequence_lengths | 12920/1736 | 26.043787 | 76.701924 | 46.594932 |
| all | row | H2_reply_cadence | 12920/1736 | 26.032325 | 76.994473 | 46.731740 |
| all | row | H3_calendar_drift | 12920/1736 | 26.189456 | 77.051960 | 46.948255 |
| all | user | baseline | 12920/1736 | 28.208048 | 86.206128 | 53.213126 |
| all | user | H1_sequence_lengths | 12920/1736 | 28.143885 | 86.080510 | 53.131605 |
| all | user | H2_reply_cadence | 12920/1736 | 28.189888 | 86.308957 | 53.221428 |
| all | user | H3_calendar_drift | 12920/1736 | 28.264556 | 86.319610 | 53.425919 |
| first_turn | row | baseline | 4908/1726 | 33.300349 | 114.440223 | 68.633318 |
| first_turn | row | H1_sequence_lengths | 4908/1726 | 33.317217 | 114.500897 | 68.603349 |
| first_turn | row | H2_reply_cadence | 4908/1726 | 33.304868 | 114.552119 | 68.662661 |
| first_turn | row | H3_calendar_drift | 4908/1726 | 33.279533 | 114.521030 | 68.853384 |
| first_turn | user | baseline | 4908/1726 | 29.039913 | 97.560885 | 60.810140 |
| first_turn | user | H1_sequence_lengths | 4908/1726 | 29.041461 | 97.551856 | 60.784840 |
| first_turn | user | H2_reply_cadence | 4908/1726 | 29.027567 | 97.641107 | 60.854500 |
| first_turn | user | H3_calendar_drift | 4908/1726 | 29.037809 | 97.596109 | 61.018966 |
| followup_turn | row | baseline | 8012/1093 | 21.660933 | 53.985127 | 33.348644 |
| followup_turn | row | H1_sequence_lengths | 8012/1093 | 21.588220 | 53.546986 | 33.112991 |
| followup_turn | row | H2_reply_cadence | 8012/1093 | 21.577302 | 53.987368 | 33.297272 |
| followup_turn | row | H3_calendar_drift | 8012/1093 | 21.846208 | 54.099115 | 33.529587 |
| followup_turn | user | baseline | 8012/1093 | 24.144210 | 62.445702 | 39.565436 |
| followup_turn | user | H1_sequence_lengths | 8012/1093 | 23.900040 | 61.989939 | 39.308099 |
| followup_turn | user | H2_reply_cadence | 8012/1093 | 24.085182 | 62.513214 | 39.540978 |
| followup_turn | user | H3_calendar_drift | 8012/1093 | 24.284667 | 62.555949 | 39.720622 |

## 환경 준비·검증 기록

```json
{
  "summary": "지정 /tmp/venv-temporal/bin/python으로 재실행. 설치 없음. 전체 합성 검사 50개 중 49개 PASS·1개 tokenizer 의존성 skip, 새 안전성 검사 8개 PASS. 구조·diff 형식 PASS. 공통 verify는 기존 TraceLab 링크 8건으로 FAIL. 기존 파일 18개 해시 유지.",
  "checks": [
    {
      "command": "/tmp/venv-temporal/bin/python -m unittest discover -s wildchat/tests",
      "exit_code": 0,
      "tests_run": 50,
      "passed": 49,
      "skipped": 1,
      "elapsed_seconds": 1.517,
      "skip_reason": "tiktoken absent; test_prepare.IntegrationTests.test_cli_real_tokenizer_reproducible_outputs_and_report"
    },
    {
      "command": "/tmp/venv-temporal/bin/python scripts/check_context.py",
      "exit_code": 0,
      "errors": 0
    },
    {
      "command": "git diff --check",
      "exit_code": 0
    },
    {
      "command": "/tmp/venv-temporal/bin/python scripts/verify.py",
      "exit_code": 1,
      "known_TraceLab_relative_link_errors": 8,
      "json_ipynb_files": 26,
      "relative_md_links": 145
    },
    {
      "check": "pre-existing modified/untracked file hashes",
      "files_checked": 18,
      "changed": 0
    },
    {
      "check": "current runner hash equals actual training runner hash",
      "passed": true
    }
  ],
  "environment_setup": {
    "used_existing_venv": "/tmp/venv-temporal",
    "python_version": "3.13.5",
    "pip_install_this_run": false,
    "homebrew_python_used_this_run": false,
    "repo_venv_used_this_run": false,
    "pyarrow_used": "26.0.1",
    "unused_missing_dependency": "tiktoken; token counts reused from prepared data, only tokenizer fixture skipped"
  },
  "git": {
    "branch": "main",
    "HEAD": "ab66dc9aef5784371c97eb2cb9631e19ed141220",
    "requested_branch": "exp/temporal-hypotheses",
    "branch_creation": "failed: .git/refs write restriction; existing main preserved",
    "commit_push_pr_merge": false
  },
  "notion": {
    "status": "pending_unsaved",
    "target": "https://app.notion.com/p/40ff30f64d938197b478f5df54622bab",
    "blocker": "Prior write returned: MCP tool call requires approval, but approval policy is never. No new write was attempted after that denial."
  }
}
```

추가 생성 파일: wildchat/results/temporal_hypotheses.log, temporal_hypotheses_DONE.txt. 기존 파일 삭제 없음. 이번에 만들었던 설치 실패용 빈 venv와 임시 파일만 정리했다. 실제 실행 환경 /private/tmp/venv-temporal은 기존 환경이므로 보존했다.
