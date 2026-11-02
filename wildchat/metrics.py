"""Baseline quantiles and evaluation for non-negative token counts."""
import math

QUANTILES = (0.1, 0.5, 0.9)


def empirical_quantile(values, q):
    """Inverse empirical CDF: sorted value at ceil(n*q), one-based."""
    values = sorted(values)
    if not values or not 0 < q <= 1:
        raise ValueError("nonempty values and 0 < q <= 1 required")
    return values[math.ceil(len(values) * q) - 1]


def baseline(train_targets):
    values = list(train_targets)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("training targets must be finite, nonnegative, and nonempty")
    return [empirical_quantile(values, q) for q in QUANTILES]


def evaluate(targets, predictions):
    """Evaluate per-row P10/P50/P90; reject crossing rather than repair it."""
    totals = [0.0] * 7
    n = 0
    for y, pred in zip_strict(targets, predictions):
        if len(pred) != 3 or any(not math.isfinite(v) or v < 0 for v in (y, *pred)):
            raise ValueError("targets and three predictions must be finite and nonnegative")
        lo, mid, hi = pred
        if not lo <= mid <= hi:
            raise ValueError("quantile crossing: expected P10 <= P50 <= P90")
        totals[0] += abs(y - mid)
        totals[1] += abs(math.log1p(y) - math.log1p(mid))
        for i, (q, p) in enumerate(zip(QUANTILES, pred)):
            error = y - p
            totals[2 + i] += max(q * error, (q - 1) * error)
        totals[5] += lo <= y <= hi
        totals[6] += hi - lo
        n += 1
    if not n:
        raise ValueError("evaluation requires at least one row")
    vals = [x / n for x in totals]
    return {"n": n, "mae_p50": vals[0], "mean_absolute_log1p_error_p50": vals[1],
            "pinball_loss": dict(zip(("p10", "p50", "p90"), vals[2:5])),
            "mean_pinball_loss": sum(vals[2:5]) / 3,
            "p10_p90_coverage": vals[5], "mean_interval_width": vals[6]}


def zip_strict(left, right):
    from itertools import zip_longest
    sentinel = object()
    for a, b in zip_longest(left, right, fillvalue=sentinel):
        if a is sentinel or b is sentinel:
            raise ValueError("target and prediction lengths differ")
        yield a, b
