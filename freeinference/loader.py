"""Load fixed-revision processed tables while guarding the held-out period."""
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REVISION = "bb26cf72c9c20f2b05f765d88548156a74d3c253"
TEST_START = datetime(2026, 8, 16, tzinfo=timezone.utc)
TEST_START_MS = int(TEST_START.timestamp() * 1000)
FIRST_WEEK = date(2026, 5, 17)
WEEK_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$")
TABLES = {"requests", "sessions", "tool_calls"}


def data_dir():
    return Path(os.environ.get("FREEINFERENCE_DATA_DIR", "data/freeinference"))


def weeks(root=None, *, allow_test=False):
    root = Path(root) if root is not None else data_dir()
    marker = root / "REVISION"
    if not marker.is_file() or REVISION not in marker.read_text(encoding="utf-8"):
        raise ValueError("Data REVISION does not match the pinned FreeInference revision")
    release = root / "release"
    result = []
    for path in sorted(release.iterdir()):
        if not path.is_dir():
            continue
        match = WEEK_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected release directory: {path.name}")
        start, end = map(date.fromisoformat, match.groups())
        offset = (start - FIRST_WEEK).days
        if offset < 0 or offset % 7 or end != start + timedelta(days=6) or offset >= 16 * 7:
            raise ValueError(f"Invalid release week: {path.name}")
        if start >= TEST_START.date() and not allow_test:
            raise ValueError(f"Held-out test week refused: {path.name}")
        result.append(path)
    return result


def load_table(week, table, columns=None, *, allow_test=False):
    """Read only processed tables; requests are checked even for session/tool reads.

    The test opt-in applies to both folder and timestamps. block_ids is never read.
    """
    week = Path(week)
    root = week.parent.parent
    if week not in weeks(root, allow_test=allow_test):
        raise ValueError("Week is not in the validated release directory")
    if table not in TABLES:
        raise ValueError(f"Unknown processed table: {table}")
    if columns is not None and any(name == "block_ids" or name.startswith("block_ids.") for name in columns):
        raise ValueError("block_ids must not be loaded")
    request_path = week / "processed" / "requests.parquet"
    # Read only the timestamp column to guard all tables, including requests
    # whose caller did not ask for start_ms.
    if not allow_test:
        starts = pq.read_table(request_path, columns=["start_ms"]).column("start_ms")
        if starts.null_count or any(value.as_py() >= TEST_START_MS for value in starts):
            raise ValueError("Missing start_ms or held-out test timestamp refused")
    path = week / "processed" / f"{table}.parquet"
    if columns is None:
        columns = [name for name in pq.read_schema(path).names if name != "block_ids"]
    return pd.read_parquet(path, columns=columns)
