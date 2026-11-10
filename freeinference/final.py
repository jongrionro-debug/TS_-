"""Building blocks for the approved final evaluation; no test data read on import."""
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import PREDICTION_COLUMNS, QUANTILES
from .models import (CATEGORICAL, PARAMETERS, FeatureEncoder, QuantileModels,
                     _features, _targets)

REFERENCE = Path(__file__).parent / "outputs" / "models" / "models.json"
DEMAND_REFERENCE = Path(__file__).parent / 'outputs' / 'demand' / 'demand.json'


def fit_fixed(rows, features, train_target, reference=REFERENCE):
    """Refit the recorded selected models, without running selection or early stopping."""
    _features(features, rows.index)
    mask = np.asarray(train_target, dtype=bool)
    if mask.shape != (len(rows),):
        raise ValueError("Training mask must align with rows")
    selected = json.loads(Path(reference).read_text(encoding="utf-8"))["selection"]
    y = _targets(rows, mask)
    fitted = QuantileModels()
    fitted.models, fitted.encoders, fitted.selection = {}, {}, selected
    for condition, frame in features.items():
        encoder = FeatureEncoder().fit(frame.loc[mask])
        x = encoder.transform(frame.loc[mask])
        fitted.encoders[condition] = encoder
        fitted.models[condition] = {}
        names = [f"f{i}" for i in range(len(frame.columns))]
        categorical = [i for i, name in enumerate(frame.columns) if name in CATEGORICAL]
        for q, label in zip(QUANTILES, PREDICTION_COLUMNS):
            selection = selected[condition][label]
            setting = selection["settings"]
            transform = setting["target_transform"]
            if transform not in ("identity", "log1p"):
                raise ValueError("Unknown target transform")
            target = np.log1p(y) if transform == "log1p" else y
            params = {**PARAMETERS, **{k: v for k, v in setting.items()
                                      if k != "target_transform"}, "alpha": q}
            fitted.models[condition][label] = lgb.train(
                params, lgb.Dataset(x, label=target, feature_name=names,
                                    categorical_feature=categorical),
                num_boost_round=selection["iterations"])
    return fitted

from . import demand
from .baselines import Baselines, build_history
from .evaluate import evaluate_predictions, paired_bootstrap, BOOTSTRAP_B, BOOTSTRAP_SEED
from .features import (FEATURE_SETS, REQUEST_COLUMNS, SESSION_COLUMNS as FEATURE_SESSION_COLUMNS,
                       build_features, build_cross_session_features)
from .loader import FIRST_WEEK, REVISION, TEST_START_MS, load_table, weeks
from .models import comparisons
from .split import SESSION_COLUMNS, WEEK_MS, split_masks

OUTPUT = Path(__file__).parent / 'outputs' / 'final'
TABLE_NAMES = ('requests', 'sessions', 'tool_calls')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_data(root):
    root = Path(root)
    if (root / 'REVISION').read_text(encoding='utf-8').strip() != REVISION:
        raise ValueError('Data revision mismatch')
    paths = weeks(root, allow_test=True)
    if len(paths) != 16 or any(p.name[:10] != (FIRST_WEEK + timedelta(weeks=i)).isoformat()
                               for i, p in enumerate(paths)):
        raise ValueError('Final evaluation requires consecutive weeks 1–16')
    expected = json.loads((root / 'expected_sha256.json').read_text(encoding='utf-8'))
    checked, manifests = {}, {}
    for path in paths[13:]:
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            raise ValueError('Invalid manifest')
        if manifest.get('week') != path.name:
            raise ValueError('Manifest week mismatch')
        traces = trace_hashes(manifest)
        if not traces:
            raise ValueError('Manifest has no trace hashes')
        manifests[path.name] = {'sha256': sha256(path / 'manifest.json'), 'trace_hashes': traces}
        for table in TABLE_NAMES:
            relative = f'release/{path.name}/processed/{table}.parquet'
            value = expected.get(relative)
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
                raise ValueError('Missing or invalid expected processed hash')
            actual = sha256(root / relative)
            if actual != value:
                raise ValueError('Processed hash mismatch')
            checked[relative] = actual
    return paths, checked, manifests


def trace_hashes(manifest):
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Manifest files missing')
    found = {}
    for name, entry in files.items():
        if (not isinstance(name, str) or not re.fullmatch(r'traces/\d{4}-\d{2}-\d{2}\.jsonl', name)
                or not isinstance(entry, dict) or not isinstance(entry.get('sha256'), str)
                or not re.fullmatch(r'[0-9a-fA-F]{64}', entry['sha256'])):
            raise ValueError('Invalid manifest trace hash')
        found[name] = entry['sha256']
    return found


def final_masks(rows):
    earlier = rows.week.le(13)
    old = split_masks(rows.loc[earlier])
    result = pd.DataFrame(False, index=rows.index, columns=['train_target', 'validation', 'test'])
    result.loc[earlier, ['train_target', 'validation']] = old[['train_target', 'validation']]
    if rows[list(SESSION_COLUMNS)].isna().any().any() or rows.start_ms.isna().any():
        raise ValueError('Missing session keys or start time')
    first = rows.groupby(list(SESSION_COLUMNS), sort=False).start_ms.transform('min')
    assigned = (first - int(datetime.combine(FIRST_WEEK, datetime.min.time(), timezone.utc).timestamp() * 1000)) // WEEK_MS + 1
    if ((assigned < 1) | (assigned > 16) | (rows.start_ms >= TEST_START_MS + 3 * WEEK_MS)).any():
        raise ValueError('Request outside release period')
    if ((rows.week >= 14) & (assigned <= 13)).any():
        raise ValueError('Test-folder session starts before test period')
    result['test'] = assigned.between(14, 16)
    return result


def validate_test_rows(rows, mask):
    test = rows.loc[mask]
    if test.empty:
        raise ValueError('No test rows')
    if test[list(SESSION_COLUMNS)].isna().any().any():
        raise ValueError('Missing test session keys')
    if test.account.isna().any():
        raise ValueError('Missing test account')
    values = pd.to_numeric(test.completion_tokens, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid test completion tokens')


def run(root=None, output=OUTPUT):
    output = Path(output)
    if any((output / name).exists() for name in ('final.json', 'final.md', 'demand.png')):
        raise FileExistsError('Final evaluation already exists')
    if root is None:
        root = os.environ.get('FREEINFERENCE_FINAL_DATA_DIR')
        if not root:
            raise ValueError('FREEINFERENCE_FINAL_DATA_DIR is required')
    # Reserve the one run before opening any data, including for concurrent calls.
    # A failed attempt leaves the marker in place for explicit human review.
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'final.json').open('x', encoding='utf-8') as stream:
        stream.write('{"status": "incomplete"}\n')
    paths, checked, manifests = validate_data(root)
    weekly, histories, feature_frames, session_frames = [], [], [], []
    for number, path in enumerate(paths, 1):
        # The final directory contains held-out weeks, so its loader opt-in is
        # required even for earlier folders; split_masks validates their times.
        requests = load_table(path, 'requests', REQUEST_COLUMNS, allow_test=True)
        sessions = load_table(path, 'sessions', FEATURE_SESSION_COLUMNS, allow_test=True)
        calls = load_table(path, 'tool_calls', ['session_id', 'step', 'category', 'outcome', 'result_tokens'], allow_test=True)
        built = build_features(requests, sessions, calls)
        rows = requests[['session_id', 'step', 'start_ms', 'end_ms', 'link_reason', 'model', 'completion_tokens']].copy()
        rows['week'] = number
        rows['harness'] = built['A'].harness
        histories.append(build_history(rows))
        weekly.append(rows)
        feature_frames.append(built['B'])
        session_frames.append(sessions.assign(week=number))
    rows = pd.concat(weekly, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    sessions = pd.concat(session_frames, ignore_index=True)
    rows['account'] = rows[['week', 'session_id']].merge(sessions[['week', 'session_id', 'user_id']],
        on=['week', 'session_id'], how='left', sort=False, validate='many_to_one').user_id.to_numpy()
    cross = build_cross_session_features(rows, sessions)
    cross.attrs.clear()
    b = pd.concat([pd.concat(feature_frames, ignore_index=True), cross], axis=1)
    features = {name: b.loc[:, list(columns)] for name, columns in FEATURE_SETS.items()}
    masks = final_masks(rows)
    validate_test_rows(rows, masks.test)
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    model = fit_fixed(rows, features, masks.train_target, REFERENCE)
    baselines = Baselines().fit(rows, history, masks.train_target)
    def predict(mask):
        frame = rows.loc[mask].copy()
        model_predictions, _ = model.predict({name: values.loc[mask] for name, values in features.items()})
        base, _ = baselines.predict(frame, history.loc[mask])
        return frame, {**base, **model_predictions}
    validation, vp = predict(masks.validation)
    validation_metrics = evaluate_predictions(validation, vp)
    # Selection comes from the same reference; reproduction is checked by metrics.
    if validation_metrics != reference['metrics']:
        raise ValueError('Fixed model validation reproduction failed')
    test, predictions = predict(masks.test)
    metrics = evaluate_predictions(test, predictions)
    bootstrap = comparisons(test, predictions)
    scores = metrics['overall']['metrics']
    ci = bootstrap['B_minus_L']['groups']['overall']['differences']['B']['request']['mean_pinball']
    if None in ci:
        raise ValueError('Primary confidence interval unavailable')
    conclusion = ('test에서 B가 L보다 개선' if ci[1] < 0 else
                  'test에서 악화' if ci[0] > 0 else '차이 미확인')
    blocks = {}
    test['utc_day'] = pd.to_datetime(test.start_ms, unit='ms', utc=True).dt.floor('D')
    for cluster in ('account', 'utc_day'):
        blocks[cluster] = {name: paired_bootstrap(test, {ref: predictions[ref], target: predictions[target]},
            reference=ref, cluster_columns=(cluster,), B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
            for name, target, ref in (('B_minus_L', 'B', 'L'), ('L_minus_A', 'L', 'A'),
                                       ('C_minus_B', 'C', 'B'), ('C_minus_L', 'C', 'L'))}
    frame = demand.aggregate_predictions(test, predictions, start_ms=TEST_START_MS, end_ms=TEST_START_MS + 3 * WEEK_MS)
    demand_result = {'metrics': {name: demand.metrics(frame, name) for name in predictions},
                     'bootstrap': {f'{a}_minus_{b}': demand.bootstrap(frame, a, b) for a, b in (('B', 'L'), ('C', 'B'))}}
    validation_demand = json.loads(DEMAND_REFERENCE.read_text(encoding='utf-8'))
    result = {'executed_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {name: sha256(Path(__file__).parent / name) for name in
                  ('final.py', 'loader.py', 'features.py', 'split.py', 'baselines.py', 'evaluate.py', 'models.py', 'demand.py', 'eda.py')},
              'revision': REVISION, 'processed_sha256': checked, 'manifests': manifests,
              'validation': {'metrics': reference['metrics'], 'bootstrap': reference['bootstrap'],
                             'block_bootstrap': reference['robustness'],
                             'demand': {'metrics': validation_demand['metrics']['overall'],
                                        'bootstrap': validation_demand['bootstrap']},
                             'reproduced': True},
              'test': {'metrics': metrics, 'bootstrap': bootstrap, 'block_bootstrap': blocks,
                       'primary': {'difference': scores['B']['request']['mean_pinball'] - scores['L']['request']['mean_pinball'],
                                   'ci95': ci, 'conclusion': conclusion}, 'demand': demand_result}}
    result_json = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    report = f"# 최종 test 평가\n\n{conclusion}. B−L: {result['test']['primary']['difference']:.6f}, 95% CI {ci}.\n\n세션 bootstrap, 요청 가중 mean pinball을 주 판정으로 사용한다. 보조 비교와 validation 대비 값은 [final.json](final.json), 수요 뷰는 [그림](demand.png)에 있다.\n"
    # A single aggregate plot does not expose request-level values.
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(frame.index, frame.actual, label='Actual', linewidth=.7)
    ax.plot(frame.index, frame.B_sum_P50, label='B sum P50', linewidth=.7)
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    try:
        fig.savefig(output / 'demand.png', dpi=130)
    finally:
        plt.close(fig)
    (output / 'final.md').write_text(report, encoding='utf-8')
    (output / 'final.json').write_text(result_json, encoding='utf-8')
    return result


if __name__ == '__main__':
    run()
