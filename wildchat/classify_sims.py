#!/usr/bin/env python3
"""Issue 19 follow-up: per-category similarity scores for the existing MiniLM labels.

Re-embeds the same train/validation prompts with the same model, taxonomy and
truncation as classify.py and writes all 7 max-cosine category similarities.
This is not a re-classification: every row's argmax must equal the stored
label in --labels or the run fails. The test split file is never read.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from wildchat.classify import CATEGORIES, MODEL, TAXONOMY, prompts, wanted_ids


def per_category(prompt_vectors, prototype_vectors, prototype_labels):
    import numpy as np
    p = prompt_vectors / np.maximum(np.linalg.norm(prompt_vectors, axis=1, keepdims=True), 1e-12)
    q = prototype_vectors / np.maximum(np.linalg.norm(prototype_vectors, axis=1, keepdims=True), 1e-12)
    sims, labels = p @ q.T, np.asarray(prototype_labels)
    return np.column_stack([sims[:, labels == c].max(axis=1) for c in CATEGORIES])


def run(args):
    import numpy as np
    from fastembed import TextEmbedding
    stored = {}
    with args.labels.open() as h:
        for line in h:
            r = json.loads(line)
            stored[(r["source_row_id"], r["user_message_index"])] = r["category"]
    model = TextEmbedding(MODEL, threads=args.threads)
    proto_labels = [c for c in CATEGORIES for _ in TAXONOMY[c]]
    proto = np.asarray(list(model.embed([t for c in CATEGORIES for t in TAXONOMY[c]])))
    n, mismatch, buf = 0, 0, []

    def flush(out):
        nonlocal n, mismatch
        sims = per_category(np.asarray(list(model.embed([t for *_, t in buf], batch_size=64))), proto, proto_labels)
        for (s, m, _), row in zip(buf, sims):
            mismatch += CATEGORIES[int(row.argmax())] != stored[(s, m)]
            out.write(json.dumps({"source_row_id": s, "user_message_index": m,
                                  "sims": [round(float(v), 5) for v in row]}) + "\n")
        n += len(buf); buf.clear()
        print(f"embedded {n} mismatches {mismatch}", flush=True)

    with args.output.open("x") as out:
        for item in prompts(args.source, wanted_ids(args.prepared)):
            buf.append(item)
            if len(buf) >= 2048:
                flush(out)
        if buf:
            flush(out)
    if n != len(stored):
        raise ValueError("row count differs from stored labels")
    print(json.dumps({"rows": n, "argmax_mismatches": mismatch, "categories": CATEGORIES}))
    if mismatch > args.max_mismatch_fraction * n:
        raise ValueError("similarities disagree with stored labels")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("prepared", "source", "labels", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--max-mismatch-fraction", type=float, default=0.001,
                   help="tolerate float ties/ONNX nondeterminism only")
    run(p.parse_args())


if __name__ == "__main__":
    main()
