"""Fixed, label-only chronological masks; never remove feature-history rows."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .loader import FIRST_WEEK, TEST_START_MS

TRAIN_WEEKS = tuple(range(1, 11))
VALIDATION_WEEKS = (11, 12, 13)
HOLDOUT_WEEK = 10
FIT_WEEKS = tuple(week for week in TRAIN_WEEKS if week < HOLDOUT_WEEK)
FIRST_MS = int(datetime.combine(FIRST_WEEK, datetime.min.time(), timezone.utc).timestamp() * 1000)
WEEK_MS = int(timedelta(weeks=1).total_seconds() * 1000)
VALIDATION_START_MS = FIRST_MS + 10 * WEEK_MS
HOLDOUT_START_MS = FIRST_MS + (HOLDOUT_WEEK - 1) * WEEK_MS
SESSION_COLUMNS = ("week", "session_id")


def split_masks(requests):
    """Assign by first start in each public (week, session_id), preserve index.

    ``train``/``fit`` include all session rows. Only ``*_target`` masks apply
    strict end-time cutoffs, including exclusion of missing end timestamps.
    Public week is an integer 1–13; it remains part of identity even when a
    session's first start belongs to a different calendar week.
    """
    if requests[list(SESSION_COLUMNS)].isna().any().any():
        raise ValueError("Session keys must be non-null")
    if not requests.week.isin((*TRAIN_WEEKS, *VALIDATION_WEEKS)).all():
        raise ValueError("Only public weeks 1–13 are allowed")
    starts = requests.start_ms
    if (starts.isna().any() or not np.isfinite(starts).all()
            or starts.lt(FIRST_MS).any() or starts.ge(TEST_START_MS).any()):
        raise ValueError("Missing or outside train/validation start timestamp refused")
    first = requests.groupby(list(SESSION_COLUMNS), sort=False).start_ms.transform("min")
    assigned = ((first - FIRST_MS) // WEEK_MS + 1).astype(int)
    result = pd.DataFrame({"session_week": assigned}, index=requests.index)
    for name, selected, cutoff in (
        ("train", assigned.isin(TRAIN_WEEKS), VALIDATION_START_MS),
        ("validation", assigned.isin(VALIDATION_WEEKS), None),
        ("fit", assigned.isin(FIT_WEEKS), HOLDOUT_START_MS),
        ("holdout", assigned.eq(HOLDOUT_WEEK), VALIDATION_START_MS),
    ):
        result[name] = selected
        result[f"{name}_target"] = selected if cutoff is None else (
            selected & requests.end_ms.notna() & requests.end_ms.lt(cutoff))
    return result
