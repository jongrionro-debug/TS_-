"""Validation request-start token demand nowcast; no request-level output."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import lightgbm as lgb

from .baselines import Baselines, PREDICTION_COLUMNS, build_history
from .eda import aggregate_10min, BIN_MS
from .evaluate import BOOTSTRAP_B, BOOTSTRAP_SEED, cluster_draws, evaluate_predictions
from .features import FEATURE_SETS, REQUEST_COLUMNS, SESSION_COLUMNS, build_features, build_cross_session_features
from .loader import FIRST_WEEK, REVISION, TEST_START_MS, data_dir, load_table, weeks
from .models import QuantileModels
from .split import VALIDATION_START_MS, WEEK_MS, split_masks

OUTPUT = Path(__file__).parent / 'outputs' / 'demand'
EXAMPLE_DAY = '2026-07-26'


def aggregate_predictions(rows, predictions, *, start_ms=None, end_ms=None):
    """Aggregate aligned predictions using the EDA UTC bin and gap rules."""
    source = rows[['start_ms', 'completion_tokens']].copy()
    source['prompt_tokens'] = 0
    actual = aggregate_10min(source, start_ms, end_ms)
    result = pd.DataFrame({'actual': actual.completion_tokens, 'count': 0}, index=actual.index)
    bins = pd.to_datetime((rows.start_ms // BIN_MS) * BIN_MS, unit='ms', utc=True)
    result['count'] = bins.value_counts().reindex(result.index, fill_value=0).astype(int)
    for name, frame in predictions.items():
        if not rows.index.equals(frame.index):
            raise ValueError('Predictions must align with rows')
        values = frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
            raise ValueError('Predictions must be finite and ordered')
        grouped = pd.DataFrame(values, index=bins, columns=PREDICTION_COLUMNS).groupby(level=0).sum()
        for col in PREDICTION_COLUMNS:
            result[f'{name}_sum_{col}'] = grouped[col].reindex(result.index, fill_value=0)
    return result


def metrics(frame, name):
    y = frame.actual.to_numpy(dtype=float)
    lo, mid, hi = (frame[f'{name}_sum_{q}'].to_numpy(dtype=float) for q in PREDICTION_COLUMNS)
    error = mid - y
    width = hi - lo
    ratios = width[y > 0] / y[y > 0]
    return {'bins': len(y), 'mae': float(np.abs(error).mean()),
            'wape': float(np.abs(error).sum() / y.sum()) if y.sum() else None,
            'bias': float(error.mean()), 'coverage': float(((lo <= y) & (y <= hi)).mean()),
            'mean_width': float(width.mean()),
            'median_width_actual': float(np.median(ratios)) if len(ratios) else None,
            'width_ratio_bins': int(len(ratios))}


def bootstrap(frame, name, reference, *, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Paired UTC day blocks; denominator recomputed in each draw."""
    days, codes = np.unique(frame.index.floor('D'), return_inverse=True)
    y = frame.actual.to_numpy(dtype=float)
    delta = (np.abs(y - frame[f'{name}_sum_P50'].to_numpy()) -
             np.abs(y - frame[f'{reference}_sum_P50'].to_numpy()))
    sums = np.bincount(codes, weights=delta, minlength=len(days))
    totals = np.bincount(codes, weights=y, minlength=len(days))
    counts = np.bincount(codes, minlength=len(days))
    samples = []
    for draws in cluster_draws(len(days), B=B, seed=seed):
        numerator = draws @ sums
        denominator = draws @ totals
        mae = numerator / (draws @ counts)
        wape = np.divide(numerator, denominator, out=np.full(len(draws), np.nan), where=denominator > 0)
        samples.extend(zip(mae, wape))
    samples = np.asarray(samples)
    return {'blocks': len(days), 'B': B, 'seed': seed,
            **{key: (np.percentile(samples[np.isfinite(samples[:, i]), i], [2.5, 97.5]).tolist()
                     if np.isfinite(samples[:, i]).any() else [None, None])
               for i, key in enumerate(('mae_ci95', 'wape_ci95'))}}


def plots(frame, output):
    def series(data, path):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(data.index, data.actual, label='Actual', linewidth=.7)
        ax.plot(data.index, data.B_sum_P50, label='B sum P50', linewidth=.7)
        ax.fill_between(data.index, data.B_sum_P10.to_numpy(), data.B_sum_P90.to_numpy(), alpha=.2, label='B sum P10–P90')
        ax.set_ylabel('Completion tokens / 10 min'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(output / path, dpi=130); plt.close(fig)
    series(frame, 'validation.png')
    series(frame.loc[EXAMPLE_DAY], 'first_day.png')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, ('B', 'L', 'model')):
        ax.scatter(frame.actual, frame[f'{name}_sum_P50'], s=3, alpha=.25)
        limit = max(frame.actual.max(), frame[f'{name}_sum_P50'].max())
        ax.plot([0, limit], [0, limit], color='black', linewidth=.7)
        ax.set(xlabel='Actual', ylabel='sum P50', title=name)
    fig.tight_layout(); fig.savefig(output / 'scatter.png', dpi=130); plt.close(fig)


def run(root=None, output=OUTPUT):
    started = perf_counter()
    paths = weeks(root)
    if len(paths) != 13 or any(p.name[:10] != (FIRST_WEEK + timedelta(weeks=i)).isoformat() for i, p in enumerate(paths)):
        raise ValueError('Demand requires consecutive weeks 1–13')
    weekly, histories, feature_frames, session_frames = [], [], [], []
    for number, path in enumerate(paths, 1):
        requests = load_table(path, 'requests', REQUEST_COLUMNS)
        sessions = load_table(path, 'sessions', SESSION_COLUMNS)
        session_frames.append(sessions.assign(week=number))
        calls = load_table(path, 'tool_calls', ['session_id', 'step', 'category', 'outcome', 'result_tokens'])
        features = build_features(requests, sessions, calls)
        feature_frames.append(features['B'])
        rows = requests[['session_id', 'step', 'start_ms', 'end_ms', 'link_reason', 'model', 'completion_tokens']].copy()
        rows['week'] = number
        rows['harness'] = features['A'].harness
        histories.append(build_history(rows)); weekly.append(rows)
        print(f'week {number}: {len(rows):,} rows', flush=True)
    rows = pd.concat(weekly, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    sessions = pd.concat(session_frames, ignore_index=True)
    cross = build_cross_session_features(rows, sessions); cross.attrs.clear()
    b = pd.concat([pd.concat(feature_frames, ignore_index=True), cross], axis=1)
    features = {name: b.loc[:, list(columns)] for name, columns in FEATURE_SETS.items()}
    del weekly, histories, feature_frames, session_frames, cross, b, requests, calls, sessions
    masks = split_masks(rows)
    model = QuantileModels().fit(rows, features)
    validation = rows.loc[masks.validation].copy()
    model_predictions, _ = model.predict({name: frame.loc[masks.validation] for name, frame in features.items()})
    baselines = Baselines().fit(rows, history, masks.train_target)
    baseline_predictions, _ = baselines.predict(validation, history.loc[masks.validation])
    predictions = {**baseline_predictions, **model_predictions}
    request_metrics = evaluate_predictions(validation, predictions)
    reference = json.loads((Path(__file__).parent / 'outputs/models/models.json').read_text())
    equal = request_metrics == reference['metrics']
    if len(validation) != 327952 or not equal:
        raise ValueError(f'Validation reproduction failed: rows={len(validation)}, metrics_equal={equal}')
    frame = aggregate_predictions(validation, predictions, start_ms=VALIDATION_START_MS, end_ms=TEST_START_MS)
    names = list(predictions)
    groups = {'overall': frame}
    groups.update({f'hour/{h:02}': frame[frame.index.hour == h] for h in range(24)})
    groups.update({f'week/{w}': frame[(frame.index >= pd.to_datetime(VALIDATION_START_MS + (w-11)*WEEK_MS, unit='ms', utc=True)) & (frame.index < pd.to_datetime(VALIDATION_START_MS + (w-10)*WEEK_MS, unit='ms', utc=True))] for w in (11, 12, 13)})
    scores = {group: {name: metrics(part, name) for name in names} for group, part in groups.items()}
    comparisons = {f'{a}-{b}': bootstrap(frame, a, b) for a, b in [('B', 'L'), ('C', 'B')]}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    plots(frame, output)
    result = {'revision': REVISION, 'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'code_state': 'working tree implementation; harness records final commit',
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ('loader.py', 'features.py', 'split.py', 'baselines.py', 'evaluate.py', 'models.py', 'eda.py', 'demand.py')},
              'command': 'FREEINFERENCE_DATA_DIR="$FREEINFERENCE_DATA_DIR" OPENBLAS_NUM_THREADS=1 python -m freeinference.demand',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__, 'lightgbm': lgb.__version__, 'platform': platform.platform()},
              'validation_requests': len(validation), 'request_metrics_match_models_json': equal,
              'bins': len(frame), 'empty_bins': int(frame['count'].eq(0).sum()),
              'metrics': scores, 'bootstrap': comparisons,
              'series': [{'start_utc': index.isoformat(), **{k: float(v) if k != 'count' else int(v) for k, v in row.items()}} for index, row in frame.iterrows()],
              'elapsed_seconds': perf_counter() - started}
    (output / 'demand.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# Validation 10분 출력 토큰 수요', '', '요청 시작 UTC 10분 구간의 현재 추정(nowcast)이다. 구간의 모든 요청이 도착해야 합이 완성되며 미래 도착을 예측하지 않는다.', 'P10·P90의 합은 완전 양의 상관을 가정한 보수적 범위이며 합의 보정된 80% 구간이 아니다. P50의 합은 합의 중앙값이 아니므로 bias를 함께 표시한다.', '폭/실측 중앙값은 실측이 0인 구간을 제외한다. UTC 일 블록은 21개이며 재학습 불확실성은 포함하지 않는다. 운영 지표이며 주 가설 판정을 대신하지 않는다.', '', f"요청 {len(validation):,}건, 구간 {len(frame):,}개, 빈 구간 {result['empty_bins']:,}개. 요청 지표의 models.json 일치: {equal}.", '', '[전체 그림](validation.png) · [첫날](first_day.png) · [산점도](scatter.png) · [전체 지표와 실행 정보](demand.json)', '', '| 구간 | 예측기 | MAE | WAPE | bias | 포함률 | 평균 폭 | 폭/실측 중앙값 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for group, values in scores.items():
        for name, v in values.items():
            fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
            lines.append('| ' + ' | '.join([group, name] + [fmt(v[k]) for k in ('mae', 'wape', 'bias', 'coverage', 'mean_width', 'median_width_actual')]) + ' |')
    lines += ['', '## UTC 일 블록 bootstrap 95% CI', '', '| 비교 | MAE 차이 | WAPE 차이 |', '|---|---|---|']
    for name, value in comparisons.items():
        lines.append(f"| {name} | {value['mae_ci95']} | {value['wape_ci95']} |")
    (output / 'demand.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    run(data_dir())
