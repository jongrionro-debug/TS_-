#!/usr/bin/env python3
"""Issue 19: save the MiniLM prompt embeddings (384-dim, float16) for train/validation.

Same model, truncation and prompts as classify.py / classify_sims.py. Writes
an .npz with keys (source_row_id, user_message_index) and vectors, outside the
repository. Earlier runs did not persist vectors, so this recomputes them once.
The test split file is never read.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from wildchat.classify import MODEL, prompts, wanted_ids


def main():
    import numpy as np
    from fastembed import TextEmbedding
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("prepared", "source", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("output exists")
    model = TextEmbedding(MODEL, threads=a.threads)
    keys, vecs, buf = [], [], []

    def flush():
        vecs.append(np.asarray(list(model.embed([t for *_, t in buf], batch_size=64)), dtype=np.float16))
        keys.extend((s, m) for s, m, _ in buf); buf.clear()
        print(f"embedded {len(keys)}", flush=True)

    for item in prompts(a.source, wanted_ids(a.prepared)):
        buf.append(item)
        if len(buf) >= 2048:
            flush()
    if buf:
        flush()
    np.savez(a.output, keys=np.asarray(keys, dtype=np.int64), vectors=np.concatenate(vecs))
    print("saved", len(keys))


if __name__ == "__main__":
    main()
