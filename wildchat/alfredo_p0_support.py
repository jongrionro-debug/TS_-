"""PR24 allowlist/cache checks, vendored unchanged for main-based P0.

Source commit d1a45fa, wildchat/temporal.py. H1 source is copied byte-for-byte
in alfredo_h1_features.py; original hashes are in alfredo_p0_reference.json.
"""
import sys
from pathlib import Path
from contextlib import contextmanager
import json
import numpy as np
from wildchat import revalidate as rv
from wildchat.classify import MODEL, MAX_CHARS, SPLIT_FILES

@contextmanager
def allowed_jsonl_reads(prepared):
    """Audit actual Python file opens; allow only the two declared JSONL inputs."""
    allowed = {(prepared / name).resolve() for name in SPLIT_FILES.values()}
    opened, active = set(), [True]

    def audit(event, args):
        if active[0] and event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
            if path.suffix == ".jsonl":
                if path.resolve() not in allowed:
                    raise PermissionError("JSONL access outside train/validation allowlist")
                opened.add(path.name)

    sys.addaudithook(audit)
    try:
        yield opened
    finally:
        active[0] = False

def load_embeddings(prepared, path, expected_hash):
    """Check prior checksum, format, unique keys and exact allowed-row coverage."""
    digest = rv.fingerprint(path)
    if digest != expected_hash:
        raise ValueError("embedding cache checksum differs from recorded provenance")
    with np.load(path, allow_pickle=False) as data:
        keys, vectors = data["keys"], data["vectors"]
        if (keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype.kind not in "iu" or
            vectors.shape != (len(keys), 384) or vectors.dtype.kind != "f" or
            not np.isfinite(vectors).all() or np.any(keys < 0)):
            raise ValueError("invalid embedding cache format or values")
        index = {tuple(map(int, key)): i for i, key in enumerate(keys)}
        if len(index) != len(keys):
            raise ValueError("duplicate embedding cache identities")
        out, seen = {}, set()
        for split, name in SPLIT_FILES.items():
            indices = []
            with (prepared / name).open("rb") as handle:
                for raw in handle:
                    meta = json.loads(raw)["metadata"]
                    key = (meta["source_row_id"], meta["user_message_index"])
                    if key not in index or key in seen:
                        raise ValueError("embedding cache missing or repeated prepared identity")
                    seen.add(key)
                    indices.append(index[key])
            out[split] = vectors[indices].astype(np.float32)
        if seen != set(index):
            raise ValueError("embedding cache has identities outside train/validation")
    return out, {"sha256": digest, "rows": len(index), "dimensions": 384,
                 "unique_keys": True, "exact_train_validation_coverage": True,
                 "model": MODEL, "prompt_prefix_characters": MAX_CHARS,
                 "regenerated": False, "reuse_authority": "issue 23 operator context"}
