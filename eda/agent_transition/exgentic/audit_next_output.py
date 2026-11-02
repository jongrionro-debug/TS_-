"""Audit pre-generation features in the local shard; no model/API calls or raw text export."""
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
EXPECTED_SHA256 = "5cfde1c01460d2f7d90fdc6c245278cee20f32101d5275b6cf3d06f30a5f8f60"
INPUT_FEATURES = [
    "model", "harness", "input_chars", "input_message_count", "input_text_chars",
    "input_assistant_message_count", "latest_user_text_chars", "tool_def_count",
    "tool_defs_chars", "has_tool_defs", "input_tool_response_count",
    "input_tool_response_chars", "input_history_tool_call_count", "input_thinking_chars",
    "history_call_count", "history_usage_count", "has_prev_usage", "prev_in_tok",
    "prev_out_tok", "recent3_out_mean", "past_cum_in_tok", "past_cum_out_tok",
    "prev_duration_s", "past_error_count",
]


def decode_list(raw, field, counts):
    if raw is None:
        counts[f"{field}_missing"] += 1
        return []
    try:
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError("Expected an array")
    except (TypeError, ValueError):
        counts[f"{field}_invalid"] += 1
        return []
    counts[f"{field}_parsed"] += 1
    return value


def parts(messages):
    return [part for message in messages for part in message.get("parts", [])]


def char_count(value):
    if value is None:
        return 0
    return len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))


def to_utc(raw, field, counts):
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        counts[f"{field}_naive_assumed_utc"] += 1
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def run():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, "Source changed; reassess before comparing audit counts"
    sessions = pq.read_table(SOURCE).to_pylist()
    counts, attr_keys, roles, part_types = Counter(), Counter(), Counter(), Counter()
    models, statuses, finish_reasons, tool_counts = Counter(), Counter(), Counter(), Counter()
    initial_hashes = Counter()
    records = []
    for session in sessions:
        spans = []
        for raw_span in session["spans"]:
            a = raw_span["attributes"]
            attr_keys.update(a.keys())
            messages = decode_list(a.get("gen_ai.input.messages"), "input_messages", counts)
            output = decode_list(a.get("gen_ai.output.messages"), "output_messages", counts)
            definitions = decode_list(a.get("gen_ai.tool.definitions"), "tool_definitions", counts)
            start = to_utc(raw_span["start_time"], "start_time", counts)
            end = to_utc(raw_span["end_time"], "end_time", counts)
            assert end >= start
            code = (raw_span.get("status") or {}).get("code")
            statuses[str(code)] += 1
            models[a.get("gen_ai.request.model")] += 1
            finish_reasons[str(a.get("gen_ai.response.finish_reasons"))] += 1
            spans.append({"raw": raw_span, "attrs": a, "start": start, "end": end,
                          "input": messages, "output": output, "definitions": definitions,
                          "status": code})
            for side, collection in (("input", messages), ("output", output)):
                for message in collection:
                    roles[f"{side}:{message.get('role')}"] += 1
                part_types.update(f"{side}:{p.get('type')}" for p in parts(collection))
            for part in parts(output):
                if part.get("type") == "tool_call":
                    tool_counts[part.get("name")] += 1
        spans.sort(key=lambda span: (span["start"], span["raw"]["span_id"]))
        initial_hash = hashlib.sha256(
            spans[0]["attrs"]["gen_ai.input.messages"].encode()
        ).hexdigest()
        initial_hashes[initial_hash] += 1
        for idx, span in enumerate(spans):
            a, messages = span["attrs"], span["input"]
            # Filter by completion time, not adjacent row or final session totals.
            history = sorted(
                [s for s in spans if s is not span and s["end"] < span["start"]],
                key=lambda s: (s["end"], s["raw"]["span_id"]),
            )
            successful_history = [s for s in history if s["status"] == 1]
            for previous in history:
                assert previous["end"] < span["start"]
            prev = successful_history[-1] if successful_history else None
            iparts, oparts = parts(messages), parts(span["output"])
            tool_responses = [p for p in iparts if p.get("type") == "tool_call_response"]
            plans = [p for p in iparts if p.get("type") == "tool_call" and p.get("name") == "TodoWrite"]
            counts["input_has_todo_plan"] += bool(plans)
            if plans:
                arguments = plans[-1].get("arguments")
                counts["input_latest_todo_has_structured_list"] += (
                    isinstance(arguments, dict) and isinstance(arguments.get("todos"), list)
                )
            user_texts = [
                "\n".join(p.get("content", "") for p in message.get("parts", [])
                          if p.get("type") == "text" and isinstance(p.get("content"), str))
                for message in messages if message.get("role") == "user"
            ]
            user_texts = [text for text in user_texts if text]
            counts["input_has_user_text"] += bool(user_texts)
            counts["input_has_tool_response"] += bool(tool_responses)
            counts["input_has_thinking"] += any(p.get("type") == "thinking" for p in iparts)
            counts["input_has_task_marker"] += any("Task:" in text for text in user_texts)
            counts["output_has_tool_call"] += any(p.get("type") == "tool_call" for p in oparts)
            counts["output_has_thinking"] += any(p.get("type") == "thinking" for p in oparts)
            if idx:
                counts["adjacent_overlaps"] += spans[idx - 1]["end"] > span["start"]
                counts["adjacent_equal_boundary"] += spans[idx - 1]["end"] == span["start"]
                previous_calls = {p.get("id") for p in parts(spans[idx - 1]["output"])
                                  if p.get("type") == "tool_call"}
                response_ids = {p.get("id") for p in tool_responses}
                if previous_calls:
                    counts["previous_has_tool_call"] += 1
                    counts["previous_tool_ids_in_input"] += previous_calls.issubset(response_ids)
            label = a.get("gen_ai.usage.output_tokens")
            eligible = (span["status"] == 1 and bool(span["output"])
                        and isinstance(label, int) and label >= 0 and bool(messages))
            past_in = [s["attrs"]["gen_ai.usage.input_tokens"] for s in successful_history]
            past_out = [s["attrs"]["gen_ai.usage.output_tokens"] for s in successful_history]
            record = {
                # These metadata/label columns are never estimator inputs.
                "session_id": session["session_id"], "initial_input_sha256": initial_hash,
                "span_id": span["raw"]["span_id"], "benchmark": session["benchmark"],
                "prediction_time": span["start"], "label_eligible": eligible,
                "target_output_tokens": label if eligible else None,
                # Features below use request fields or strictly completed history only.
                "model": a.get("gen_ai.request.model"), "harness": session["harness"],
                "input_chars": len(a["gen_ai.input.messages"]),
                "input_message_count": len(messages),
                "input_text_chars": sum(char_count(p.get("content")) for p in iparts
                                        if p.get("type") == "text"),
                "input_assistant_message_count": sum(m.get("role") == "assistant" for m in messages),
                "latest_user_text_chars": len(user_texts[-1]) if user_texts else 0,
                "tool_def_count": len(span["definitions"]),
                "tool_defs_chars": char_count(a.get("gen_ai.tool.definitions")),
                "has_tool_defs": a.get("gen_ai.tool.definitions") is not None,
                "input_tool_response_count": len(tool_responses),
                "input_tool_response_chars": sum(char_count(p.get("result")) for p in tool_responses),
                "input_history_tool_call_count": sum(p.get("type") == "tool_call" for p in iparts),
                "input_thinking_chars": sum(char_count(p.get("thinking")) for p in iparts
                                            if p.get("type") == "thinking"),
                "history_call_count": len(history), "history_usage_count": len(successful_history),
                "has_prev_usage": prev is not None,
                "prev_in_tok": past_in[-1] if past_in else None,
                "prev_out_tok": past_out[-1] if past_out else None,
                "recent3_out_mean": sum(past_out[-3:]) / len(past_out[-3:]) if past_out else None,
                "past_cum_in_tok": sum(past_in), "past_cum_out_tok": sum(past_out),
                "prev_duration_s": (prev["end"] - prev["start"]).total_seconds() if prev else None,
                "past_error_count": sum(s["status"] == 2 for s in history),
            }
            assert set(record) == set(INPUT_FEATURES) | {
                "session_id", "initial_input_sha256", "span_id", "benchmark", "prediction_time",
                "label_eligible", "target_output_tokens",
            }
            records.append(record)
    frame = pd.DataFrame(records)
    labeled = frame[frame.label_eligible]
    audit = {
        "scope": "Local shard 0 only; no training, no API calls, no full-dataset validation",
        "source_sha256": digest, "sessions": len(sessions), "spans": len(frame),
        "benchmark_sessions": dict(Counter(s["benchmark"] for s in sessions)),
        "harness_sessions": dict(Counter(s["harness"] for s in sessions)),
        "model_spans": dict(models), "status_codes": dict(statuses),
        "counts": dict(counts), "attribute_keys": dict(attr_keys),
        "roles": dict(roles), "part_types": dict(part_types),
        "finish_reasons": dict(finish_reasons), "common_output_tools": tool_counts.most_common(12),
        "eligible_labels": len(labeled),
        "zero_usage_error_spans": sum(
            s["status"]["code"] == 2 and s["attributes"]["gen_ai.usage.input_tokens"] == 0
            and s["attributes"]["gen_ai.usage.output_tokens"] == 0
            for row in sessions for s in row["spans"]
        ),
        "eligible_with_history": int(labeled.has_prev_usage.sum()),
        "unique_initial_inputs": len(initial_hashes),
        "duplicate_initial_input_sessions": sum(v - 1 for v in initial_hashes.values()),
        "session_max_equals_observed_max_call_tokens": sum(
            row["max_tokens"] == max(
                s["attributes"]["gen_ai.usage.input_tokens"] + s["attributes"]["gen_ai.usage.output_tokens"]
                for s in row["spans"]
            ) for row in sessions
        ),
        "output_token_summary": labeled.target_output_tokens.describe(
            percentiles=[.5, .9, .95, .99]
        ).to_dict(),
        "feature_columns": INPUT_FEATURES,
        "feature_missing_counts": {key: int(frame[key].isna().sum()) for key in INPUT_FEATURES},
        "input_tool_response_length_nonzero": int((frame.input_tool_response_chars > 0).sum()),
        "input_thinking_length_nonzero": int((frame.input_thinking_chars > 0).sum()),
        "checks": {
            "source_hash_verified": True,
            "unique_span_ids": bool(frame.span_id.is_unique),
            "past_completion_strictly_before_prediction": True,
            "current_output_usage_duration_and_finish_not_input_features": True,
            "raw_messages_not_exported": True,
        },
        "limits": [
            "Naive timestamps are assumed UTC for ordering, following the existing local audit; their timezone is not independently verified.",
            "Recorded input messages/model/tools are assumed to reconstruct the prepared request; no independent pre-send snapshot.",
            "Current exact input usage is post-response telemetry; use request character lengths or a validated tokenizer estimate.",
            "No request output cap, temperature, reasoning-effort or cache/reasoning usage breakdown in this shard's attribute keys.",
            "Output target is recorded API usage, not guaranteed visible-answer-only tokens.",
            "Many adjacent spans overlap. Do not infer a serial conversation or usage availability from array adjacency alone.",
            "Session max_tokens matches the observed maximum input+output tokens; it is not a request output cap.",
            "Initial input hashes detect exact repeated prompts only, not semantic task overlap.",
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    frame.to_parquet(RESULTS / "next_output_feature_audit.parquet", index=False)
    (RESULTS / "next_output_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in {
        "attribute_keys", "roles", "part_types", "feature_missing_counts", "limits",
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
