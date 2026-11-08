"""Aggregate the prepared 1–13 week release without exporting identifiers."""
import json
import shlex
import sys
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from .loader import REVISION, data_dir, load_table, weeks

OUTPUT = Path(__file__).parent / "outputs" / "audit"
REQUEST_COLUMNS = ["session_id", "step", "start_ms", "end_ms", "link_reason", "model", "provider"]
SESSION_COLUMNS = ["session_id", "user_id", "harness"]


def counts(series):
    result = series.fillna("<missing>").value_counts(dropna=False)
    return {str(key): int(value) for key, value in result.items()}


def distribution(series):
    values = counts(series)
    total = len(series)
    return {key: {"count": value, "share": value / total if total else 0} for key, value in values.items()}


def audit(root=None):
    paths = weeks(root)
    if len(paths) != 13:
        raise ValueError("Audit requires exactly the 13 train/validation weeks")
    expected = datetime(2026, 5, 17).date()
    if any(path.name[:10] != (expected + timedelta(days=7 * i)).isoformat() for i, path in enumerate(paths)):
        raise ValueError("Audit requires consecutive weeks 1–13")
    weekly = []
    all_rows = []
    excluded_total = Counter()
    for week in paths:
        requests = load_table(week, "requests", REQUEST_COLUMNS)
        sessions = load_table(week, "sessions", SESSION_COLUMNS)
        if sessions.session_id.isna().any() or sessions.session_id.duplicated().any():
            raise ValueError("Sessions must have unique non-null session IDs within each week")
        rows = requests.merge(sessions, on="session_id", how="left", validate="many_to_one", indicator=True)
        if (rows["_merge"] != "both").any() or rows.session_id.isna().any():
            raise ValueError("Requests without a matching session")
        rows = rows.drop(columns="_merge")
        rows["week"] = week.name
        boundary_start = int(datetime.combine(datetime.fromisoformat(week.name[:10]).date(), time(), timezone.utc).timestamp() * 1000)
        boundary_end = boundary_start + 7 * 24 * 60 * 60 * 1000
        bounds = rows.groupby("session_id").agg(first_start=("start_ms", "min"), last_end=("end_ms", "max"))
        cross = (((bounds.first_start < boundary_start) & (bounds.last_end >= boundary_start)) |
                 ((bounds.first_start < boundary_end) & (bounds.last_end >= boundary_end))).sum()
        manifest = json.loads((week / "manifest.json").read_text(encoding="utf-8"))
        excluded = manifest["excluded_sessions"]
        excluded_total.update(excluded)
        account_requests = rows.groupby("user_id", dropna=False).size().sort_values(ascending=False)
        weekly.append({"week": week.name, "requests": len(rows), "sessions": len(sessions),
                       "accounts": int(sessions.user_id.nunique()),
                       "top_1_account_request_share": float(account_requests.iloc[:1].sum() / len(rows)) if len(rows) else 0,
                       "top_5_account_request_share": float(account_requests.iloc[:5].sum() / len(rows)) if len(rows) else 0,
                       "distributions": {key: distribution(rows[key]) for key in ("harness", "model", "provider")},
                       "excluded_sessions": excluded, "sessions_crossing_week_boundary": int(cross)})
        all_rows.append(rows)
    rows = pd.concat(all_rows, ignore_index=True)
    # Week is part of the key because released sessions are reconstructed per week.
    rows = rows.sort_values(["week", "session_id", "step"], kind="stable")
    groups = rows.groupby(["week", "session_id"], sort=False)
    previous_end = groups.end_ms.shift()
    previous_step = groups.step.shift()
    adjacent = previous_end.notna() & rows.end_ms.notna()
    inversions = adjacent & (rows.end_ms < previous_end)
    overlapping = adjacent & (rows.start_ms <= previous_end)
    link = distribution(rows.link_reason)
    # A preceding row is only a preceding step when its step number is k - 1.
    eligible = (rows.step > 0) & (previous_step == rows.step - 1) & previous_end.notna() & rows.start_ms.notna()
    link_pairs = {}
    for reason in sorted(rows.loc[rows.step > 0, "link_reason"].fillna("<missing>").unique()):
        mask = (rows.step > 0) & (rows.link_reason.fillna("<missing>") == reason)
        pairs = mask & eligible
        completed = pairs & (previous_end < rows.start_ms)
        n = int(pairs.sum())
        link_pairs[reason] = {"later_step_requests": int(mask.sum()),
                              "adjacent_pairs": n,
                              "strictly_completed": int(completed.sum()),
                              "strictly_completed_rate": float(completed.sum() / n) if n else None,
                              "not_strictly_completed": n - int(completed.sum()),
                              "unassessable_requests": int(mask.sum()) - n}
    account_requests = rows.groupby("user_id", dropna=False).size().sort_values(ascending=False)
    return {"revision": REVISION, "scope": "weeks 1–13, train and validation only", "weeks": weekly,
            "overall": {"requests": len(rows), "sessions": sum(w["sessions"] for w in weekly),
                        "accounts": int(rows.user_id.nunique()),
                        "top_1_account_request_share": float(account_requests.iloc[:1].sum() / len(rows)),
                        "top_5_account_request_share": float(account_requests.iloc[:5].sum() / len(rows)),
                        "distributions": {key: distribution(rows[key]) for key in ("harness", "model", "provider")},
                        "excluded_sessions": dict(excluded_total),
                        "end_before_start": int((rows.end_ms < rows.start_ms).sum()),
                        "missing_start_or_end": int((rows.start_ms.isna() | rows.end_ms.isna()).sum()),
                        "adjacent_step_pairs_with_end_times": int(adjacent.sum()),
                        "adjacent_step_end_order_inversions": int(inversions.sum()),
                        "adjacent_step_end_order_inversion_rate": float(inversions.sum() / adjacent.sum()) if adjacent.sum() else None,
                        "adjacent_step_pairs_not_strictly_completed": int(overlapping.sum()),
                        "adjacent_step_pairs_not_strictly_completed_rate": float(overlapping.sum() / adjacent.sum()) if adjacent.sum() else None,
                        "sessions_crossing_week_boundary": sum(w["sessions_crossing_week_boundary"] for w in weekly),
                        "link_reason": link,
                        "link_reason_adjacent_pairs": link_pairs,
                        "link_reason_by_first_step": {"step_0": distribution(rows.loc[rows.step == 0, "link_reason"]),
                                                       "later_steps": distribution(rows.loc[rows.step > 0, "link_reason"])},
                        "duplicate_session_step_rows": int(rows.duplicated(["week", "session_id", "step"]).sum()),
                        "missing_steps": int(rows.step.isna().sum())}}


def render(result):
    o = result["overall"]
    lines = ["# FreeInference 1~13주 데이터 감사", "",
             f"- 실제 실행: `{shlex.quote(sys.executable)} -m freeinference.audit` (`FREEINFERENCE_DATA_DIR`에 사전 준비 경로 설정)",
             f"- revision: `{REVISION}`; 1~10주 train, 11~13주 validation. test는 읽지 않았다.",
             f"- 요청 {o['requests']:,}건, 주별 세션 합계 {o['sessions']:,}개, 계정 {o['accounts']:,}개. 상위 1·5개 계정 요청 점유율 {o['top_1_account_request_share']:.2%}·{o['top_5_account_request_share']:.2%}.",
             "- 분포는 요청 수 기준이며 주별 count와 share는 JSON에 기록한다. 계정 수는 비결측 고유 ID 기준이다.", "",
             "| 주 | 요청 | 세션 | 계정 | 상위 1 / 5 계정 요청 점유율 | 주 경계 초과 세션 | 제외 세션 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for w in result['weeks']:
        lines.append(f"| {w['week']} | {w['requests']:,} | {w['sessions']:,} | {w['accounts']:,} | {w['top_1_account_request_share']:.2%} / {w['top_5_account_request_share']:.2%} | {w['sessions_crossing_week_boundary']:,} | {sum(w['excluded_sessions'].values()):,} |")
    lines += ["", "## 시간 정합성", "",
              f"- 종료가 시작보다 빠른 요청: {o['end_before_start']:,}건; 시작·종료 결측: {o['missing_start_or_end']:,}건. 중복 세션·step 행: {o['duplicate_session_step_rows']:,}건; step 결측: {o['missing_steps']:,}건.",
              f"- 세션 내 step 정렬 후 인접 쌍 {o['adjacent_step_pairs_with_end_times']:,}개 중 종료시각 역전 {o['adjacent_step_end_order_inversions']:,}개 ({o['adjacent_step_end_order_inversion_rate']:.2%}). 같은 종료시각은 역전이 아니다.",
              f"- 인접 쌍에서 이전 요청이 현재 시작 시점에 엄격히 완료되지 않은 경우(이전 종료 ≥ 현재 시작): {o['adjacent_step_pairs_not_strictly_completed']:,}개 ({o['adjacent_step_pairs_not_strictly_completed_rate']:.4%}). 이는 병렬·동시 요청의 지표이며 step만으로 이력을 인정할 수 없다.",
              f"- 주별 폴더 시작·끝 경계를 실제로 가로지르는 세션: {o['sessions_crossing_week_boundary']:,}개(주별 세션 합). 공개 데이터는 주 경계에서 세션을 분할하므로 실제 이어진 세션은 이 수치로 식별할 수 없다.", "",
              "## 세션 연결과 제외", ""]
    for key, value in o['link_reason'].items():
        lines.append(f"- `link_reason` {key}: {value['count']:,}건 ({value['share']:.2%}).")
    first_missing = o["link_reason_by_first_step"]["step_0"].get("<missing>", {}).get("count", 0)
    later_missing = o["link_reason_by_first_step"]["later_steps"].get("<missing>", {}).get("count", 0)
    lines.append(f"- 연결 근거 결측: step 0에서 {first_missing:,}건, 이후 step에서 {later_missing:,}건. step별 전체 분포는 JSON에 기록한다.")
    pairs = o["link_reason_adjacent_pairs"]
    lines += ["", "직전 step과의 쌍(이전 step 또는 시각이 없는 요청은 평가 불가):", ""]
    for reason, value in pairs.items():
        rate = f"{value['strictly_completed_rate']:.4%}" if value['strictly_completed_rate'] is not None else "평가 불가"
        lines.append(f"- `{reason}`: {value['adjacent_pairs']:,}쌍 중 엄격 완료 {value['strictly_completed']:,}쌍 ({rate}), 미완료 {value['not_strictly_completed']:,}쌍, 평가 불가 {value['unassessable_requests']:,}건.")
    total = sum(v["adjacent_pairs"] for v in pairs.values())
    complete = sum(v["strictly_completed"] for v in pairs.values())
    unassessable = sum(v["unassessable_requests"] for v in pairs.values())
    # This threshold assesses timing of the proxy, not availability of the reconstructed link.
    verdict = "조건부 성립" if total and not unassessable and complete / total >= 0.999 else "불성립"
    rate = f"{complete / total:.4%}" if total else "평가 불가"
    lines += ["", "공개 요청 테이블의 `waited_for`, `step_trigger`, `transition`, `cause`는 요청 ID가 아닌 범주형 값이어서 실제 연결 대상 요청을 식별하지 못한다. 따라서 직전 step을 대리로 사용했다.", "",
              f"판정: **{verdict}**. 기준은 평가 가능한 직전 step 쌍의 99.9% 이상이 `prev_end_ms < start_ms`를 만족하고 평가 불가 후속 요청이 없는 것이다. 전체 {total:,}쌍 중 {complete:,}쌍 ({rate})이 충족하고 {total - complete:,}쌍은 충족하지 않으며 평가 불가는 {unassessable:,}건이다. 이력에는 반드시 엄격 완료된 요청만 포함하고 위반 쌍은 제외한다. 다만 직전 step은 실제 연결 대상의 대리이며 이 수치만으로 사후 재구성된 연결의 시작 시점 가용성을 입증할 수 없다. 특징 생성 전에 현재 입력의 연결 단서와 이력 생성 시점을 대조하는 누출 방지 테스트로 이를 검증하고, 검증되지 않은 연결은 사용하지 않는다.", "",
              "manifest 제외 사유 합계:"]
    for key, value in o['excluded_sessions'].items():
        lines.append(f"- {key}: {value:,}개")
    lines += ["", "## 대표성", "", "무료 gateway의 공개 세션에 한정되며 제외된 세션은 분포에 포함되지 않는다. harness·모델·provider의 전체 및 주별 요청 수와 점유율은 JSON에 있어 주간 구성 변화를 비교할 수 있다. 계정 식별자와 tool 인자는 출력하지 않는다.", ""]
    for key in ("harness", "model", "provider"):
        first = result["weeks"][0]["distributions"][key]
        last = result["weeks"][-1]["distributions"][key]
        first_name = max(first, key=lambda name: first[name]["count"])
        last_name = max(last, key=lambda name: last[name]["count"])
        lines.append(f"- {key}: 1주 최다 {first_name} {first[first_name]['share']:.2%} → 13주 최다 {last_name} {last[last_name]['share']:.2%} (각 주 요청 기준).")
    lines.append("")
    return "\n".join(lines)


def main():
    result = audit(data_dir())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "audit.md").write_text(render(result), encoding="utf-8")
    print(f"Wrote aggregate audit to {OUTPUT}")


if __name__ == "__main__":
    main()
