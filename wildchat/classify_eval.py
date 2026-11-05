#!/usr/bin/env python3
"""Issue 19: accuracy of the zero-shot classifier on hand-labelled TRAIN rows.

The 200 rows were drawn with random.Random(42).sample from train.jsonl
identities only (no validation/test) and labelled by reading the prompt.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from wildchat import classify as cl


def confusion(gold, pred):
    cats = cl.CATEGORIES
    table = {g: {p: 0 for p in cats} for g in cats}
    for g, p in zip(gold, pred):
        table[g][p] += 1
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    per = {c: {"support": sum(table[c].values()),
               "recall": table[c][c] / max(sum(table[c].values()), 1),
               "precision": table[c][c] / max(sum(table[g][c] for g in cats), 1)} for c in cats}
    return {"n": len(gold), "accuracy": acc, "per_category": per, "confusion_gold_by_pred": table}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand-labels", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", default=cl.MODEL)
    a = p.parse_args()
    import numpy as np
    from fastembed import TextEmbedding
    rows = json.loads(a.hand_labels.read_text())["rows"]
    ids = {}
    for r in rows:
        ids.setdefault(r["source_row_id"], set()).add(r["user_message_index"])
    text = {(s, m): t for s, m, t in cl.prompts(a.source, ids)}
    model = TextEmbedding(a.model)
    proto = np.asarray(list(model.embed([t for c in cl.CATEGORIES for t in cl.TAXONOMY[c]])))
    labels = [c for c in cl.CATEGORIES for _ in cl.TAXONOMY[c]]
    vecs = np.asarray(list(model.embed([text[(r["source_row_id"], r["user_message_index"])] for r in rows])))
    pred, _ = cl.assign(vecs, proto, labels)
    result = {"model": a.model, **confusion([r["category"] for r in rows], pred)}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model": a.model, "accuracy": result["accuracy"]}))


if __name__ == "__main__":
    main()
