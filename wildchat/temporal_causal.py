"""Exact reusable causal/PCA/evaluation helpers from the issue-21 remote ref.

Extracted definitions only; never imports or calls the raw-shard metadata loader.
Upstream: origin/agent/issue-21-cross-conversation-val, cross_history.py and
category_pca.py. MODELS/COMPARISONS restrict evaluation to this experiment.
"""
from dataclasses import dataclass
import numpy as np
from wildchat import revalidate as rv
from wildchat.prepare import timestamp_us

MODELS = ("baseline", "H1_sequence_lengths", "H2_reply_cadence", "H3_calendar_drift")
COMPARISONS = tuple((m, "baseline") for m in MODELS[1:])
MISSING_TIME = np.iinfo(np.int64).min

@dataclass
class HistoryRows(rv.Rows):
    indices: np.ndarray
    anchors: np.ndarray
    completions: np.ndarray
    embeddings: np.ndarray

def temporal_metadata(messages, recorded):
    """Prefix-only anchors; future quality checks affect publication only."""
    stamps, anchors, previous, valid_prefix = [], {}, None, True
    for j, message in enumerate(messages):
        stamp = timestamp_us(message.get("timestamp"))
        if message.get("role") == "user":
            anchors[j] = previous if valid_prefix else None
        elif message.get("role") == "assistant":
            valid_prefix = valid_prefix and stamp is not None and (previous is None or previous <= stamp)
            stamps.append(stamp)
            previous = stamp
    complete = (bool(stamps) and valid_prefix and recorded is not None and max(stamps) == recorded)
    return anchors, recorded if complete else None

def evaluate(rows, predictions, available, seed, replicates):
    """All metrics and paired user bootstrap, same draws for every subgroup."""
    users, codes = np.unique(rows.users, return_inverse=True)
    g = len(users)
    history_users = set(rows.users[available])
    history_mask = np.isin(rows.users, list(history_users))
    populations = {"all_users": np.ones(len(rows.y), dtype=bool),
                   "users_with_history": history_mask, "users_without_history": ~history_mask}
    masks = {(p, c): pm & cm for p, pm in populations.items() for c, cm in rv.cohorts(rows).items()}
    values = np.stack([rv.contributions(rows.y, predictions[m]) for m in MODELS], axis=1)
    metrics, aggregates = {}, {}
    for (population, cohort), mask in masks.items():
        counts = np.bincount(codes[mask], minlength=g)
        sums = np.zeros((g, len(MODELS), len(rv.METRICS)))
        np.add.at(sums, codes[mask], values[mask])
        means = np.divide(sums, counts[:, None, None], out=np.zeros_like(sums), where=counts[:, None, None] > 0)
        aggregates[(population, cohort)] = counts, sums[:, :, 0], means[:, :, 0]
        entry = metrics.setdefault(population, {}).setdefault(cohort, {})
        for weight, scores, denominator in (("row", sums, counts.sum()), ("user", means, (counts > 0).sum())):
            entry[weight] = {m: {"n_rows": int(mask.sum()), "n_users": int((counts > 0).sum()),
                                 **dict(zip(rv.METRICS, map(float, scores[:, j].sum(axis=0) / denominator)))}
                             if denominator else None for j, m in enumerate(MODELS)}
    draws = {(p, c, w, a, b): [] for p, c in masks for w in ("row", "user") for a, b in COMPARISONS}
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(replicates):
        weights = np.bincount(rng.integers(0, g, size=g), minlength=g)
        for (p, c), (counts, sums, means) in aggregates.items():
            for w, data, den in (("row", sums, weights @ counts), ("user", means, weights @ (counts > 0))):
                if not den:
                    continue
                for a, b in COMPARISONS:
                    delta = weights @ (data[:, MODELS.index(a)] - data[:, MODELS.index(b)]) / den
                    draws[(p, c, w, a, b)].append(float(delta))
    boot = {}
    for (p, c, w, a, b), sample in draws.items():
        ms = metrics[p][c][w]
        boot.setdefault(p, {}).setdefault(c, {}).setdefault(w, {})[f"{a}-{b}"] = {
            "point_difference": ms[a]["mean_pinball_loss"] - ms[b]["mean_pinball_loss"] if ms[a] else None,
            "effective_replicates": len(sample), "empty_replicates": replicates - len(sample),
            "ci95": np.quantile(sample, [.025, .975], method="linear").tolist() if sample else None}
    return metrics, boot

class PCA:
    """Plain SVD PCA; fit() sees only the rows it is given."""

    def fit(self, x, k):
        self.mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[:k]
        self.explained = (s[:k] ** 2 / (s ** 2).sum()).tolist()
        return self

    def transform(self, x):
        return (x - self.mean) @ self.components.T
