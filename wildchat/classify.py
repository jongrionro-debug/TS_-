#!/usr/bin/env python3
"""Issue 19: zero-shot local prompt-category labels for train/validation rows only.

A multilingual sentence-embedding model (ONNX via fastembed, CPU) embeds each
current user prompt and a fixed set of category descriptions; each prompt gets
the category whose description prototypes are most similar (max cosine).
Only train.jsonl and validation.jsonl identities are read; test is never opened.
Output is an ID->category JSONL outside the repository; no text is written.
"""
import sys
from pathlib import Path

if __name__ == "__main__":  # wildchat/inspect.py would shadow stdlib inspect
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_CHARS = 600
SPLIT_FILES = {"train": "train.jsonl", "validation": "validation.jsonl"}  # never test
TAXONOMY = {
    "coding": [
        "Write or fix a program in Python, JavaScript, Java, C++ or SQL.",
        "Here is my code and the error message, please debug it.",
        "How do I implement this function, class or API? Show code.",
    ],
    "creative_writing": [
        "Write a story, a poem, song lyrics or a fan fiction.",
        "Continue this role-play scene and write the dialogue between characters.",
        "Write an episode script with characters and a plot.",
    ],
    "editing_summarization": [
        "Rewrite, paraphrase or polish this text to sound more professional.",
        "Summarize the following article in a few sentences.",
        "Correct the grammar of this paragraph.",
    ],
    "translation": [
        "Translate this text into English.",
        "Translate the following sentence into Chinese.",
        "How do you say this in Japanese? Translate it.",
    ],
    "knowledge_qa": [
        "Explain what this concept means and how it works.",
        "What are the causes and effects of this historical event?",
        "Give me advice and a list of tips about this topic.",
    ],
    "math_reasoning": [
        "Solve this math problem step by step.",
        "Calculate the answer to this equation or physics problem.",
        "Solve this logic puzzle or probability question.",
    ],
    "chit_chat": [
        "Hi, how are you?",
        "Thank you! Who are you, are you ChatGPT?",
        "Let's just chat casually, tell me a joke.",
    ],
}
CATEGORIES = tuple(TAXONOMY)


def assign(prompt_vectors, prototype_vectors, prototype_labels):
    """Nearest category by max cosine to its prototypes; ties -> first category."""
    import numpy as np
    p = prompt_vectors / np.maximum(np.linalg.norm(prompt_vectors, axis=1, keepdims=True), 1e-12)
    q = prototype_vectors / np.maximum(np.linalg.norm(prototype_vectors, axis=1, keepdims=True), 1e-12)
    sims = p @ q.T
    labels = np.asarray(prototype_labels)
    per_cat = np.column_stack([sims[:, labels == c].max(axis=1) for c in CATEGORIES])
    best = per_cat.argmax(axis=1)
    return [CATEGORIES[i] for i in best], per_cat.max(axis=1)


def wanted_ids(prepared):
    ids = {}
    for split, name in SPLIT_FILES.items():
        with (prepared / name).open("rb") as handle:
            for raw in handle:
                meta = json.loads(raw)["metadata"]
                ids.setdefault(meta["source_row_id"], set()).add(meta["user_message_index"])
    return ids


def prompts(source, ids):
    """Yield (source_row_id, user_message_index, truncated text) for wanted rows only."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    offset = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512, columns=["conversation"]):
        idx = [i for i in range(len(batch)) if offset + i in ids]
        for i, row in zip(idx, batch.take(pa.array(idx, type=pa.int64())).column(0).to_pylist()):
            for m in sorted(ids[offset + i]):
                message = row[m]
                if message["role"] != "user":
                    raise ValueError("prepared user_message_index is not a user message")
                yield offset + i, m, (message["content"] or "")[:MAX_CHARS]
        offset += len(batch)


def run(args):
    import numpy as np
    from fastembed import TextEmbedding
    started = time.monotonic()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    if args.source.name != manifest["input_files"][0]["name"]:
        raise ValueError("source filename differs from preparation provenance")
    ids = wanted_ids(args.prepared)
    model = TextEmbedding(MODEL, threads=args.threads)
    proto_text = [t for c in CATEGORIES for t in TAXONOMY[c]]
    proto_labels = [c for c in CATEGORIES for _ in TAXONOMY[c]]
    proto = np.asarray(list(model.embed(proto_text)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts, n, buf = {c: 0 for c in CATEGORIES}, 0, []

    def flush(out):
        nonlocal n
        vecs = np.asarray(list(model.embed([t for _, _, t in buf], batch_size=64)))
        labels, scores = assign(vecs, proto, proto_labels)
        for (s, m, _), c, sc in zip(buf, labels, scores):
            out.write(json.dumps({"source_row_id": s, "user_message_index": m,
                                  "category": c, "similarity": round(float(sc), 4)}) + "\n")
            counts[c] += 1
        n += len(buf)
        buf.clear()
        print(f"classified {n}", flush=True)

    with args.output.open("x") as out:
        for item in prompts(args.source, ids):
            buf.append(item)
            if len(buf) >= 2048:
                flush(out)
        if buf:
            flush(out)
    expected = sum(len(v) for v in ids.values())
    if n != expected:
        raise ValueError("classified row count differs from prepared train/validation")
    summary = {"issue": 19, "model": MODEL, "max_chars": MAX_CHARS, "taxonomy": TAXONOMY,
               "method": "zero-shot nearest prototype (max cosine) over multilingual sentence embeddings",
               "splits_read": list(SPLIT_FILES), "rows": n, "category_counts_train_plus_validation": counts,
               "taxonomy_sha256": hashlib.sha256(json.dumps(TAXONOMY, sort_keys=True).encode()).hexdigest(),
               "elapsed_seconds": time.monotonic() - started}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["category_counts_train_plus_validation"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="labels JSONL outside the repo")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"classify: {error}\n")


if __name__ == "__main__":
    main()
