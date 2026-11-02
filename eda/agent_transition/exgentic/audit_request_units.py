"""Audit user-request boundaries and trial-total candidates in the local shard.

One row here is a benchmark trial, NOT a verified human turn. Labels are recorded
usage proxies; missing conversation links and request boundaries remain unknown.
No raw message content, model inference or credential access is performed.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source/train-00000-of-00039.parquet"
RESULTS = ROOT / "results"
EXPECTED_HASH = "5cfde1c01460d2f7d90fdc6c245278cee20f32101d5275b6cf3d06f30a5f8f60"
INITIAL_FEATURES = [
    "initial_model", "harness", "initial_input_chars", "initial_message_count",
    "initial_user_text_chars", "initial_tool_def_count", "initial_tool_defs_chars",
    "initial_tool_defs_present",
]
HISTORY_FIELDS = ["user_turn_index", "previous_request_input_tokens", "previous_request_output_tokens"]
BOUNDARY_FIELDS = {"user_id", "conversation_id", "user_turn_id", "request_id",
                   "message_origin", "human_input_id", "request_start", "request_end"}


def time_key(span):
    time = datetime.fromisoformat(span["start_time"].replace("Z", "+00:00"))
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    return time.astimezone(timezone.utc), span["span_id"]


def user_texts(messages):
    texts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        text = "\n".join(part["content"] for part in message.get("parts", [])
                         if part.get("type") == "text" and isinstance(part.get("content"), str))
        if text:
            texts.append(text)
    return texts


def build_candidates(rows):
    records = []
    schema = {"session": set(), "span": set(), "attribute": set(), "message": set()}
    unique_text_counts, messages_per_call = Counter(), Counter()
    counts = Counter()
    for row in rows:
        schema["session"].update(row)
        spans = sorted(row["spans"], key=time_key)
        first = spans[0]["attributes"]
        initial = json.loads(first["gen_ai.input.messages"])
        texts = user_texts(initial)
        definitions = json.loads(first["gen_ai.tool.definitions"]) if first.get("gen_ai.tool.definitions") else None
        seen_text_hashes = set()
        input_sum = output_sum = errors = successes = missing_outputs = finish_calls = 0
        response_ids = []
        for span in spans:
            schema["span"].update(span)
            a = span["attributes"]
            schema["attribute"].update(a)
            messages = json.loads(a["gen_ai.input.messages"])
            for message in messages:
                schema["message"].update(message)
            current_texts = user_texts(messages)
            messages_per_call[len(current_texts)] += 1
            seen_text_hashes.update(hashlib.sha256(text.encode()).hexdigest() for text in current_texts)
            counts["calls_same_user_text_sequence_as_initial"] += current_texts == texts
            counts["calls_different_user_text_sequence_from_initial"] += current_texts != texts
            i, o = a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]
            assert isinstance(i, int) and i >= 0 and isinstance(o, int) and o >= 0
            input_sum += i
            output_sum += o
            status = (span.get("status") or {}).get("code")
            errors += status == 2
            successes += status == 1
            counts["error_zero_usage_calls"] += status == 2 and i == 0 and o == 0
            missing_outputs += a.get("gen_ai.output.messages") is None
            output = json.loads(a["gen_ai.output.messages"]) if a.get("gen_ai.output.messages") else []
            finish_calls += sum(part.get("type") == "tool_call"
                                and part.get("name") == "mcp__environment__finish"
                                for message in output for part in message.get("parts", []))
            if a.get("gen_ai.response.id") is not None:
                response_ids.append(a["gen_ai.response.id"])
        unique_text_counts[len(seen_text_hashes)] += 1
        assert len(response_ids) == len(set(response_ids)), "Duplicate response ID inside a trial"
        records.append({
            "recorded_session_id": row["session_id"], "benchmark": row["benchmark"],
            "initial_input_sha256": hashlib.sha256(first["gen_ai.input.messages"].encode()).hexdigest(),
            "initial_model": first["gen_ai.request.model"], "harness": row["harness"],
            "initial_input_chars": len(first["gen_ai.input.messages"]),
            "initial_message_count": len(initial),
            "initial_user_text_chars": sum(len(text) for text in texts),
            "initial_tool_def_count": len(definitions) if definitions is not None else None,
            "initial_tool_defs_chars": len(first["gen_ai.tool.definitions"]) if definitions is not None else None,
            "initial_tool_defs_present": definitions is not None,
            # Unknown human-turn information is never imputed as an observed first turn.
            "user_turn_index": None,
            "previous_request_input_tokens": None,
            "previous_request_output_tokens": None,
            "request_boundary_verified": False, "completion_verified": False,
            "recorded_input_tokens_sum": input_sum,
            "recorded_output_tokens_sum": output_sum,
            "recorded_total_tokens_sum": input_sum + output_sum,
            "matches_row_total_tokens": input_sum + output_sum == row["total_tokens"],
            "recorded_call_count": len(spans), "recorded_success_calls": successes,
            "recorded_error_calls": errors, "recorded_missing_output_calls": missing_outputs,
            "finish_tool_observed": finish_calls > 0,
            "distinct_user_text_blocks": len(seen_text_hashes),
            "candidate_only": True,
        })
    return pd.DataFrame(records), schema, unique_text_counts, messages_per_call, counts


def summarize(frame, schema, unique_text_counts, messages_per_call, counts):
    return {
        "scope": "Local shard 0, benchmark-trial candidates only; not verified human-request labels",
        "source_sha256": EXPECTED_HASH,
        "recorded_sessions": len(frame), "recorded_calls": int(frame.recorded_call_count.sum()),
        "candidate_trial_aggregates": len(frame),
        "verified_user_request_units": int(frame.request_boundary_verified.sum()),
        "verified_multi_request_conversations": 0,
        "interpretation": "Zero means not established by this audit, not proof of absence in all source data.",
        "benchmark_sessions": frame.benchmark.value_counts().to_dict(),
        "recorded_success_calls": int(frame.recorded_success_calls.sum()),
        "recorded_error_calls": int(frame.recorded_error_calls.sum()),
        "recorded_missing_output_calls": int(frame.recorded_missing_output_calls.sum()),
        "counts": dict(counts),
        "sessions_matching_recorded_total": int(frame.matches_row_total_tokens.sum()),
        "sessions_with_finish_tool": int(frame.finish_tool_observed.sum()),
        "distinct_initial_inputs": int(frame.initial_input_sha256.nunique()),
        "distinct_user_text_blocks_per_session": dict(unique_text_counts),
        "user_text_messages_per_call": dict(messages_per_call),
        "schema_keys": {name: sorted(values) for name, values in schema.items()},
        "explicit_boundary_field_matches": {name: sorted(values & BOUNDARY_FIELDS) for name, values in schema.items()},
        "initial_feature_columns": INITIAL_FEATURES,
        "unavailable_history_fields": HISTORY_FIELDS,
        "trial_usage_summary": frame[["recorded_input_tokens_sum", "recorded_output_tokens_sum"]].describe().to_dict(),
        "checks": {
            "source_hash_verified": True,
            "unique_recorded_session_ids": bool(frame.recorded_session_id.is_unique),
            "unverified_candidates_not_marked_as_training_labels": bool(frame.candidate_only.all()),
            "missing_user_turn_history_retained_as_missing": bool(frame[HISTORY_FIELDS].isna().all().all()),
            "all_internal_calls_aggregated_within_their_recorded_trial": True,
            "raw_messages_not_exported": True,
        },
        "limits": [
            "A benchmark session is a possible proxy for one initial user request; that mapping and task completion are not verified.",
            "Repeated role=user messages and distinct text blocks do not establish new human prompts.",
            "No conversation/user-turn identity, human message origin or user-level completion marker was established.",
            "Failure usage zero may be missing accounting; observed totals are not guaranteed complete billed usage.",
            "No task-success assessment or complete child-agent call tree validation was performed.",
            "The 43.8% statistic and prior ACF/MAE refer to internal calls, not user-request cycles.",
        ],
    }


if __name__ == "__main__":
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_HASH
    rows = pq.read_table(SOURCE).to_pylist()
    frame, schema, unique_text_counts, messages_per_call, counts = build_candidates(rows)
    audit = summarize(frame, schema, unique_text_counts, messages_per_call, counts)
    RESULTS.mkdir(exist_ok=True)
    frame.to_parquet(RESULTS / "request_unit_candidates.parquet", index=False)
    (RESULTS / "request_unit_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items()
                      if key not in {"schema_keys", "trial_usage_summary", "limits", "user_text_messages_per_call"}},
                     ensure_ascii=False, indent=2))
