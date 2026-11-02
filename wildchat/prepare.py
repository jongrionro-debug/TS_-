#!/usr/bin/env python3
"""Prepare prompt-level WildChat targets and purged chronological group splits."""
import sys
from pathlib import Path

# inspect.py in this directory must not shadow the standard-library module.
if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import argparse
from array import array
from collections import Counter
from contextlib import closing, ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import sqlite3
import tempfile

REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
SPLITS = ("train", "validation", "test")
PAIR_REASONS = (
    "user_content_missing", "user_content_empty_or_nontext", "no_adjacent_assistant",
    "assistant_content_missing", "assistant_content_empty_or_nontext",
)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def digest(value):
    """Exact JSON identity; no text normalization or semantic matching."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).digest()


def timestamp_us(value):
    """Accept timezone-aware timestamps only; never guess units/timezones."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    delta = value.astimezone(timezone.utc) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def iso_time(value):
    if value is None:
        return None
    from datetime import timedelta
    return (EPOCH + timedelta(microseconds=value)).isoformat()


def pair_reason(messages, index):
    user = messages[index].get("content")
    assistant = messages[index + 1] if index + 1 < len(messages) else None
    if user is None:
        return PAIR_REASONS[0]
    if not isinstance(user, str) or not user.strip():
        return PAIR_REASONS[1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        return PAIR_REASONS[2]
    if assistant.get("content") is None:
        return PAIR_REASONS[3]
    if not isinstance(assistant["content"], str) or not assistant["content"].strip():
        return PAIR_REASONS[4]
    return None


def messages_of(row):
    value = row.get("conversation")
    return value if isinstance(value, list) else []


def identity_keys(row):
    """Connect same declared hash, exact visible conversation, or first user text.

    The first user is found even after leading system messages. Sharing that
    exact nonblank text is a deliberately conservative prefix relation; it also
    connects forks whose later messages differ. Substrings/fuzzy matches are out
    of scope. Never use this whole-conversation information as model features.
    """
    messages = messages_of(row)
    declared = row.get("conversation_hash")
    if declared is not None and str(declared).strip():
        yield "declared_hash", digest(str(declared))
    visible = [(m.get("role"), m.get("content")) if isinstance(m, dict) else None
               for m in messages]
    if visible:
        yield "visible_conversation", digest(visible)
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            text = message.get("content")
            if isinstance(text, str) and text.strip():
                yield "first_user_prefix", digest(["user", text])
            break


class UnionFind:
    def __init__(self):
        self.parent = array("q")

    def add(self):
        index = len(self.parent)
        self.parent.append(index)
        return index

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def index_rows(db, rows):
    """First streaming pass: hashes/counts only in scratch SQLite, no raw text."""
    db.executescript("""
        CREATE TABLE identities (
            kind TEXT, digest BLOB, owner INTEGER, n INTEGER,
            PRIMARY KEY (kind, digest)) WITHOUT ROWID;
        CREATE TABLE records (
            id INTEGER PRIMARY KEY, recorded_us INTEGER, valid_samples INTEGER,
            group_id INTEGER);
    """)
    uf, exclusions = UnionFind(), Counter()
    counts = Counter(dict.fromkeys(("conversations", "user_candidates", "structurally_valid_samples",
                                   "user_timestamp_usable", "user_timestamp_missing_or_invalid",
                                   "assistant_timestamp_usable", "assistant_timestamp_missing_or_invalid",
                                   "first_user_whitespace_only"), 0))
    for row in rows:
        row_id = uf.add()
        valid = 0
        first_user_seen = False
        for index, message in enumerate(messages_of(row)):
            if isinstance(message, dict) and message.get("role") in ("user", "assistant"):
                role = message["role"]
                usable = timestamp_us(message.get("timestamp")) is not None
                counts[f"{role}_timestamp_usable"] += int(usable)
                counts[f"{role}_timestamp_missing_or_invalid"] += int(not usable)
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            if not first_user_seen:
                text = message.get("content")
                counts["first_user_whitespace_only"] += int(isinstance(text, str) and bool(text) and not text.strip())
                first_user_seen = True
            counts["user_candidates"] += 1
            reason = pair_reason(messages_of(row), index)
            if reason:
                exclusions[reason] += 1
            else:
                valid += 1
        for kind, key in identity_keys(row):
            match = db.execute("SELECT owner FROM identities WHERE kind=? AND digest=?", (kind, key)).fetchone()
            if match:
                uf.union(row_id, match[0])
                db.execute("UPDATE identities SET n=n+1 WHERE kind=? AND digest=?", (kind, key))
            else:
                db.execute("INSERT INTO identities VALUES (?, ?, ?, 1)", (kind, key, row_id))
        db.execute("INSERT INTO records VALUES (?, ?, ?, NULL)",
                   (row_id, timestamp_us(row.get("timestamp")), valid))
        counts["conversations"] += 1
        counts["structurally_valid_samples"] += valid
        if row_id % 2048 == 0:
            db.commit()
    db.executemany("UPDATE records SET group_id=? WHERE id=?",
                   ((uf.find(i), i) for i in range(len(uf.parent))))
    db.executescript("""
        CREATE INDEX records_group ON records(group_id);
        CREATE TABLE groups AS
            SELECT group_id, MIN(recorded_us) AS start_us, MAX(recorded_us) AS end_us,
                   SUM(recorded_us IS NULL) AS missing_times,
                   COUNT(*) AS conversations, SUM(valid_samples) AS samples,
                   CAST(NULL AS TEXT) AS split, CAST(NULL AS TEXT) AS exclusion
            FROM records GROUP BY group_id;
        CREATE UNIQUE INDEX groups_id ON groups(group_id);
        CREATE INDEX groups_time ON groups(start_us, group_id);
    """)
    return counts, exclusions


def split_for_interval(start, end, cuts):
    """Half-open windows: equal boundary timestamps always go to the later split."""
    if cuts is None:
        return None
    lower, upper = cuts
    if end < lower:
        return "train"
    if start >= lower and end < upper:
        return "validation"
    if start >= upper:
        return "test"
    return None


def assign_splits(db):
    db.execute("UPDATE groups SET exclusion='no_valid_samples' WHERE samples=0")
    db.execute("UPDATE groups SET exclusion='missing_or_invalid_record_time' WHERE samples>0 AND missing_times>0")
    eligible = db.execute("SELECT COUNT(*) FROM groups WHERE exclusion IS NULL").fetchone()[0]
    cuts = None
    if eligible:
        cuts = tuple(db.execute(
            "SELECT start_us FROM groups WHERE exclusion IS NULL ORDER BY start_us, group_id LIMIT 1 OFFSET ?",
            (min(math.floor(eligible * fraction), eligible - 1),)).fetchone()[0]
            for fraction in (0.70, 0.85))
        for group_id, start, end in db.execute(
                "SELECT group_id, start_us, end_us FROM groups WHERE exclusion IS NULL ORDER BY group_id"):
            split = split_for_interval(start, end, cuts)
            db.execute("UPDATE groups SET split=?, exclusion=? WHERE group_id=?",
                       (split, None if split else "crosses_time_boundary", group_id))
    db.commit()
    return {"eligible_groups_before_purge": eligible,
            "train_validation_cutoff": iso_time(cuts[0]) if cuts else None,
            "validation_test_cutoff": iso_time(cuts[1]) if cuts else None}


def text_metrics(text, encoding):
    if not isinstance(text, str):
        return {"tokens": None, "chars": None, "utf8_bytes": None}
    return {"tokens": len(encoding.encode(text, disallowed_special=())),
            "chars": len(text), "utf8_bytes": len(text.encode("utf-8"))}


def turn_interval(history, position, verified_user_timestamps):
    if position < 0:
        return None, "no_previous_turn"
    if position == 0:
        return None, "no_preceding_user_turn"
    previous, current = history[position - 1]["time"], history[position]["time"]
    if previous is None or current is None:
        return None, "user_submission_timestamp_missing_or_invalid"
    if not verified_user_timestamps:
        return None, "user_submission_timestamps_unverified"
    if current < previous:
        return None, "user_submission_timestamps_nonmonotonic"
    return (current - previous) / 1_000_000, None


def iter_samples(row, encoding, lags=3, verified_user_timestamps=False):
    """Snapshot features BEFORE inspecting current response metrics.

    A lag is a previous user turn in message order, including malformed pairs.
    We assume an adjacent earlier assistant response has already been observed.
    No row metadata, current response, later messages, or whole-chat aggregates
    are included in prompt_features/history_features.
    """
    history = []
    messages = messages_of(row)
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        prompt = text_metrics(message.get("content"), encoding)
        features = {"previous_user_turns": len(history)}
        reasons = {}
        for lag in range(1, lags + 1):
            position = len(history) - lag
            prior = history[position] if position >= 0 else None
            prefix = f"lag_{lag}_"
            features[prefix + "turn_missing"] = prior is None
            for side in ("input", "output"):
                metrics = prior[side] if prior else dict.fromkeys(("tokens", "chars", "utf8_bytes"))
                features[prefix + side + "_missing"] = metrics["tokens"] is None
                for name, value in metrics.items():
                    features[prefix + side + "_" + name] = value
            interval, reason = turn_interval(history, position, verified_user_timestamps)
            features[prefix + "turn_interval_seconds"] = interval
            features[prefix + "turn_interval_missing"] = interval is None
            if reason:
                reasons[str(lag)] = reason
        reason = pair_reason(messages, index)
        # Only now read the response. It becomes history for the NEXT user turn.
        next_message = messages[index + 1] if index + 1 < len(messages) else None
        response = text_metrics(next_message.get("content"), encoding) if (
            isinstance(next_message, dict) and next_message.get("role") == "assistant") else text_metrics(None, encoding)
        if reason is None:
            yield {
                "user_message_index": index,
                "turn_index": len(history),
                "prompt_features": {"input_" + k: v for k, v in prompt.items()},
                "history_features": features,
                "target": {"output_tokens": response["tokens"]},
                "diagnostics": {"interval_exclusion_reasons": reasons},
            }
        history.append({"input": prompt, "output": response,
                        "time": timestamp_us(message.get("timestamp"))})


def iter_parquet(files, batch_size):
    import pyarrow.parquet as pq
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size,
                                         columns=["conversation_hash", "timestamp", "conversation"]):
            yield from batch.to_pylist()


def write_json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def group_summary(db):
    result = {}
    for kind in ("declared_hash", "visible_conversation", "first_user_prefix"):
        row = db.execute("""SELECT COUNT(*), COALESCE(SUM(n>1),0),
            COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0)
            FROM identities WHERE kind=?""", (kind,)).fetchone()
        result[kind] = {"unique_keys": row[0], "repeated_keys": row[1],
                        "rows_in_repeated_keys": row[2], "extra_copies": row[2] - row[1]}
    count, multi, maximum = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(conversations>1),0), COALESCE(MAX(conversations),0) FROM groups").fetchone()
    result["connected_components"] = {"groups": count, "multirow_groups": multi, "largest_group_rows": maximum}
    return result


def partition_summary(db):
    result = {}
    for split in SPLITS:
        row = db.execute("""SELECT COUNT(*), COALESCE(SUM(conversations),0), COALESCE(SUM(samples),0),
            MIN(start_us), MAX(end_us) FROM groups WHERE split=?""", (split,)).fetchone()
        result[split] = dict(zip(("groups", "conversations", "samples", "recorded_at_min", "recorded_at_max"),
                                (*row[:3], iso_time(row[3]), iso_time(row[4]))))
    excluded = {}
    for reason, groups, conversations, samples in db.execute("""SELECT exclusion, COUNT(*),
            SUM(conversations), SUM(samples) FROM groups WHERE exclusion IS NOT NULL GROUP BY exclusion"""):
        excluded[reason] = {"groups": groups, "conversations": conversations, "samples": samples}
    return result, excluded


def validate_partitions(db):
    """One SQL group assignment per record; assert strict record-time ordering."""
    previous_end = None
    for split in SPLITS:
        start, end = db.execute("SELECT MIN(start_us), MAX(end_us) FROM groups WHERE split=?", (split,)).fetchone()
        if start is not None:
            if previous_end is not None and previous_end >= start:
                raise AssertionError("overlapping record-time windows")
            previous_end = end
    # Every identity's members have one union-find root, including transitive joins.
    # Kept records inherit only that root's split, never a per-sample assignment.
    if db.execute("SELECT COUNT(*) FROM records r LEFT JOIN groups g USING(group_id) WHERE g.group_id IS NULL").fetchone()[0]:
        raise AssertionError("unassigned conversation")


def render_report(manifest):
    m = manifest
    lines = ["# WildChat 요청 표본 준비", "", "## 범위와 재현", "",
        "- 출처: [AllenAI WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), "
        "[WildChat 논문](https://arxiv.org/abs/2405.01470), [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/).",
        f'- 지정 리비전: `{REVISION}`. 제공된 출처 정보이며 로컬 파일만으로 원격 리비전을 검증하지 않는다.',
        "- 입력: " + ", ".join(f'`{f["name"]}` ({f["rows"]:,}행, SHA-256 `{f["sha256"]}`)' for f in m["input_files"]) + ".",
        "- 이번 실행은 지정된 첫 샤드(14개 중 1개) 전체이며 전체 데이터 결과가 아니다. "
        "미입력 샤드와의 연결은 알 수 없다." if m["first_shard_only"] else
        "- 지정 입력 파일 전체를 처리했다. 데이터셋 전체인지 파일 목록을 확인해야 한다.",
        f'- 파일 이름순·행 순서대로 모두 처리하며 추가 표본 추출은 없다. seed={m["seed"]}; 무작위 연산에 사용하지 않는다.',
        f'- 배치 크기 {m["batch_size"]}; 두 번 순회하고 SQLite에는 원문 대신 해시·집계·행 번호만 저장한다.',
        "- 실행 버전: " + ", ".join(f"{k} {v}" for k, v in m["versions"].items()) + "; tokenizer `o200k_base`.",
        f'- 실행 명령: `{m["command"]}`',
        "- 출력 JSONL과 manifest는 저장소 밖에 둔다. 표본은 원문 없이 숫자 특징·정답·관리 메타데이터만 포함한다.",
        "", "## 예측 시점과 필드", "",
        "현재 user 프롬프트 제출 직후를 예측 시점으로 삼는다. `prompt_features.input_tokens`는 현재 user 가시 텍스트의 입력 토큰이며 "
        "유일한 예측 정답은 `target.output_tokens`(바로 다음 assistant의 출력 토큰)이다. 각 메시지를 독립적으로 o200k_base로 계수하고 "
        "역할 구분자·숨은 시스템 메시지·제공자 직렬화 비용은 포함하지 않는다. chars는 Python Unicode 코드 포인트 수이며 utf8_bytes는 UTF-8 바이트 길이다.",
        "",
        "모델 입력은 `prompt_features`와 `history_features`만 허용한다. 현재 응답·미래 메시지·대화 전체 통계·행 메타데이터는 특징에 없다. "
        "`metadata`의 split/group/행 번호/마지막 기록시각과 `diagnostics`는 모델 입력에서 제외한다. 마지막 기록시각은 미래 정보일 수 있으며 분할 관리에만 쓴다.",
        "",
        f'lag 1부터 {m["lags"]}까지 이전 user 턴을 역순으로 참조한다. 입력·출력 토큰, 문자·바이트 길이와 결측 지표를 각각 저장한다. '
        "이전의 인접 assistant 응답은 대화 순서상 현재 프롬프트 전에 이미 관측되었다고 가정한다. "
        "제외된 과거 쌍도 턴 위치를 차지하며, 없는 값은 null로 유지하여 더 오래된 턴으로 당겨 채우지 않는다. 첫 턴에는 모든 lag가 결측이다.",
        "",
        "간격 lag k는 이전 k번째 user와 그 직전 user 사이의 제출 간격이다. 현재 user나 assistant 완료시각을 사용하지 않는다. "
        f'실제 user 제출시각 검증 옵션: {m["verified_user_timestamps"]}. '
        "검증되지 않았거나 시각이 없거나 역전되면 수치값을 제외(null)하고 결측 지표와 사유를 남긴다. "
        f'이번 입력에서 파싱 가능한 user 시각 {m["counts"].get("user_timestamp_usable", 0):,}개, '
        f'결측·비정상 user 시각 {m["counts"].get("user_timestamp_missing_or_invalid", 0):,}개다. '
        "assistant 완료시각이나 행의 마지막 턴 시각으로 대체하지 않는다.",
        "", "## 유효 표본과 제외", "",
        f'- 대화 기록 행 {m["counts"]["conversations"]:,}; user 후보 {m["counts"]["user_candidates"]:,}; '
        f'구조상 유효 쌍 {m["counts"]["structurally_valid_samples"]:,}; 분할 후 유효 표본 {m["written_samples"]:,}.',
        "- 바로 인접한 user→assistant이며 양쪽이 비어 있지 않은 문자열인 쌍만 유지한다. 아래 첫 사유 하나로만 후보를 제외한다. "
        "행 간 중복은 삭제하지 않으며 같은 분할 안의 반복으로 유지한다.", "",
        "| 쌍 제외 사유(우선순위순) | 후보 수 |", "|---|---:|"]
    lines += [f'| {reason} | {m["pair_exclusions"].get(reason, 0):,} |' for reason in PAIR_REASONS]
    lines += ["", "## 중복과 그룹 구성", "",
        "동일 conversation_hash, 역할·가시 텍스트 목록 전체의 정확한 일치, 첫 user 텍스트의 정확한 일치 중 "
        "하나라도 연결되면 union-find로 추이적 연결 성분을 만든다. 선행 system 메시지가 있어도 첫 user를 찾는다. "
        "공통 첫 프롬프트가 다른 후속 응답으로 갈라져도 한 그룹이다. 텍스트 정규화·부분 문자열·의미상 유사성은 다루지 않는다. "
        "해시 충돌의 이론적 가능성과 공통 상용구에 의한 과도한 결합이 남는다.", "",
        "| 연결 키 | 고유 키 | 반복 키 | 반복 키 소속 행 | 추가 사본 |", "|---|---:|---:|---:|---:|"]
    for key in ("declared_hash", "visible_conversation", "first_user_prefix"):
        v = m["grouping"][key]
        lines.append(f'| {key} | {v["unique_keys"]:,} | {v["repeated_keys"]:,} | {v["rows_in_repeated_keys"]:,} | {v["extra_copies"]:,} |')
    v = m["grouping"]["connected_components"]
    lines += ["", f'최종 연결 성분 {v["groups"]:,}개, 여러 행인 그룹 {v["multirow_groups"]:,}개, 최대 그룹 {v["largest_group_rows"]:,}행.',
        f'공백만 있는 첫 user 텍스트 {m["counts"].get("first_user_whitespace_only", 0):,}행은 prefix 연결 키에서 제외한다. '
        "#4 감사의 prefix 판정은 공백 문자열을 포함했으므로 이 키의 고유 수가 다를 수 있다.",
        "", "## 시간순 분할과 경계 제외", "",
        "그룹 기록시각 범위는 소속 행의 마지막 턴 기록시각 최솟값~최댓값이다. 유효 표본이 있는 그룹 중 "
        "시각 결측·비정상 행이 하나라도 있는 그룹은 통째로 제외한다. 나머지 그룹을 최초 기록시각과 그룹 ID로 정렬하고 "
        "floor(0.70×그룹 수), floor(0.85×그룹 수) 위치의 최초 시각을 경계로 삼는다. "
        "목표 70%/15%/15%는 제거 전 그룹 수 기준이다. 경계와 같은 시각은 모두 뒤 구간에 넣고, "
        "그룹의 시각 범위가 경계를 넘으면 그룹 전체를 제외한다. 따라서 실제 그룹·행·표본 비율은 목표와 다를 수 있다.",
        "",
        f'- 경계 산정 대상 {m["boundaries"]["eligible_groups_before_purge"]:,}그룹.',
        f'- train/validation 경계: {m["boundaries"]["train_validation_cutoff"]}; validation/test 경계: {m["boundaries"]["validation_test_cutoff"]}.',
        "- 구간: train < 첫 경계, 첫 경계 ≤ validation < 둘째 경계, 둘째 경계 ≤ test. 그룹을 나눠 살리지 않는다.",
        "- 이는 기록시각 순서이며 실제 개별 요청 도착시각 순서를 보장하지 않는다. 같은 대화의 과거 턴이 행 기록시각보다 앞설 수 있다.",
        "", "| 분할 | 그룹 수 | 대화 기록 행 수 | 표본 수 | 기록시각 최소(UTC) | 기록시각 최대(UTC) |", "|---|---:|---:|---:|---|---|"]
    for split, v in m["partitions"].items():
        lines.append(f'| {split} | {v["groups"]:,} | {v["conversations"]:,} | {v["samples"]:,} | {v["recorded_at_min"]} | {v["recorded_at_max"]} |')
    lines += ["", "| 그룹 제외 사유 | 그룹 수 | 대화 기록 행 수 | 구조상 유효 표본 수 |", "|---|---:|---:|---:|"]
    for reason, v in m["group_exclusions"].items():
        lines.append(f'| {reason} | {v["groups"]:,} | {v["conversations"]:,} | {v["samples"]:,} |')
    if "missing_or_invalid_record_time" not in m["group_exclusions"]:
        lines.append("| missing_or_invalid_record_time | 0 | 0 | 0 |")
    if not m["group_exclusions"]:
        lines.append("| 없음 | 0 | 0 | 0 |")
    lines += ["", "## 간격 lag 제외와 검증", "",
        "아래는 분할 후 표본의 각 lag 슬롯 집계다. 첫 턴/이력 부족과 미검증 시각을 구분하며, 모든 null에는 결측 지표가 있다.", "",
        "| lag 슬롯: 제외 사유 | 수 |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(m["interval_exclusions"].items())]
    lines += ["", f'- 수치 간격이 있는 lag 슬롯: {m["observed_interval_slots"]:,}.',
        f'- 분할 후 입력 토큰 합 {m["input_tokens"]:,}; 출력 정답 토큰 합 {m["output_tokens"]:,}.',
        "- 실행 중 그룹별 단일 분할과 train < validation < test의 기록시각 범위 비중첩을 확인했다. "
        "합성 단위 테스트는 추이적 그룹 연결·교집합·경계 제외·현재 응답/미래 변경에 대한 현재 특징 불변성·lag 순서와 결측을 검증한다.",
        "- MAE·log 오차·pinball·커버리지와 모델 학습은 후속 평가 이슈 범위다. "
        "분할 안에 중복을 유지하므로 후속 평가에서 반복 표본의 가중 효과를 고려해야 한다.", ""]
    return "\n".join(lines)


def run(args):
    import pyarrow.parquet as pq
    import tiktoken
    files = ([args.input] if args.input.is_file() and args.input.suffix == ".parquet" else
             sorted(args.input.rglob("*.parquet")) if args.input.is_dir() else [])
    if not files:
        raise ValueError("input must be a Parquet file or a directory containing Parquet files")
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("prepared data output must be outside the repository")
    if any(output == path.resolve().parent or output in path.resolve().parents for path in files):
        raise ValueError("output must not contain input files (use a separate derived directory)")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be empty; existing artifacts are never overwritten")
    encoding = tiktoken.get_encoding("o200k_base")
    fingerprints = []
    for path in files:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        fingerprints.append({"name": path.name, "sha256": h.hexdigest(),
                             "rows": pq.ParquetFile(path).metadata.num_rows})
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wildchat-prepare-") as tmp:
        with closing(sqlite3.connect(str(Path(tmp) / "index.sqlite"))) as db:
            db.execute("PRAGMA cache_size=-32768")
            counts, pair_exclusions = index_rows(db, iter_parquet(files, args.batch_size))
            boundaries = assign_splits(db)
            validate_partitions(db)
            partitions, group_exclusions = partition_summary(db)
            print(f'Indexed {counts["conversations"]:,} conversations; tokenizing kept groups', flush=True)
            interval_exclusions = Counter()
            written = Counter()
            input_tokens = output_tokens = observed_intervals = 0
            with ExitStack() as stack:
                handles = {split: stack.enter_context((output / f"{split}.jsonl").open("w", encoding="utf-8")) for split in SPLITS}
                for row_id, row in enumerate(iter_parquet(files, args.batch_size)):
                    group_id, split, recorded = db.execute("""SELECT r.group_id, g.split, r.recorded_us
                        FROM records r JOIN groups g USING(group_id) WHERE r.id=?""", (row_id,)).fetchone()
                    if split is None:
                        continue
                    for sample in iter_samples(row, encoding, args.lags, args.verified_user_timestamps):
                        index, turn = sample.pop("user_message_index"), sample.pop("turn_index")
                        sample["metadata"] = {"source_row_id": row_id, "group_id": group_id, "split": split,
                                              "user_message_index": index, "turn_index": turn,
                                              "recorded_at": iso_time(recorded)}
                        write_json_line(handles[split], sample)
                        written[split] += 1
                        input_tokens += sample["prompt_features"]["input_tokens"]
                        output_tokens += sample["target"]["output_tokens"]
                        reasons = sample["diagnostics"]["interval_exclusion_reasons"]
                        interval_exclusions.update(f"lag_{lag}: {reason}" for lag, reason in reasons.items())
                        observed_intervals += args.lags - len(reasons)
                    if row_id % 10000 == 0:
                        print(f"Processed source row {row_id:,}", flush=True)
            for split in SPLITS:
                if written[split] != partitions[split]["samples"]:
                    raise AssertionError("pair counts changed between passes")
            with (output / "groups.jsonl").open("w", encoding="utf-8") as handle:
                for group in db.execute("SELECT group_id,start_us,end_us,conversations,samples,split,exclusion FROM groups ORDER BY group_id"):
                    write_json_line(handle, dict(zip(("group_id", "start_us", "end_us", "conversations", "samples", "split", "exclusion"), group)))
            manifest = {"revision": REVISION, "input_files": fingerprints,
                        "first_shard_only": len(files) == 1 and files[0].name == "train-00000-of-00014.parquet",
                        "seed": args.seed, "batch_size": args.batch_size, "lags": args.lags,
                        "verified_user_timestamps": args.verified_user_timestamps,
                        "versions": {"python": sys.version.split()[0], **{name: importlib.metadata.version(name)
                                     for name in ("datasets", "pyarrow", "tiktoken")}},
                        "command": " ".join([sys.executable, "wildchat/prepare.py", *sys.argv[1:]]),
                        "counts": dict(counts), "pair_exclusions": dict(pair_exclusions),
                        "grouping": group_summary(db), "boundaries": boundaries,
                        "partitions": partitions, "group_exclusions": group_exclusions,
                        "written_samples": sum(written.values()), "input_tokens": input_tokens,
                        "output_tokens": output_tokens, "interval_exclusions": dict(interval_exclusions),
                        "observed_interval_slots": observed_intervals}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "preparation.md").write_text(render_report(manifest), encoding="utf-8")
    print(f'Wrote {sum(written.values()):,} samples and preparation.md to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="empty directory outside repository")
    parser.add_argument("--seed", type=int, default=42, help="recorded for reproducibility; no random sampling")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--verified-user-timestamps", action="store_true",
                        help="ONLY when user message timestamps are verified actual submission times")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.lags <= 0:
        parser.error("batch-size and lags must be positive")
    try:
        run(args)
    except (ValueError, ImportError) as error:
        parser.exit(2, f"prepare: {error}\n")


if __name__ == "__main__":
    main()
