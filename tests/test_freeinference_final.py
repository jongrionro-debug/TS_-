"""Synthetic checks for fixed selection; no release data is opened."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_DEPS = all(importlib.util.find_spec(n) for n in ("numpy", "pandas", "pyarrow", "lightgbm", "matplotlib"))
if HAS_DEPS:
    import numpy as np
    import pandas as pd
    from freeinference import final, models
    from freeinference.baselines import PREDICTION_COLUMNS
    from freeinference.features import FEATURE_SETS


@unittest.skipUnless(HAS_DEPS, "model dependencies are required")
class FixedSelectionTests(unittest.TestCase):
    def test_fit_uses_only_mask_and_recorded_settings(self):
        rows = pd.DataFrame({"completion_tokens": [1., 2., 900., 1000.]})
        features = {name: pd.DataFrame({column: [1, 2, 3, 4] for column in columns})
                    for name, columns in FEATURE_SETS.items()}
        setting = {"num_leaves": 5, "min_data_in_leaf": 1,
                   "learning_rate": .05, "target_transform": "identity"}
        selection = {name: {q: {"settings": setting, "iterations": 7}
                            for q in PREDICTION_COLUMNS} for name in FEATURE_SETS}
        seen = []
        def train(params, dataset, **kwargs):
            seen.append((dataset.label.copy(), kwargs))
            return object()
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "models.json"
            reference.write_text(json.dumps({"selection": selection}))
            with patch.object(final.lgb, "train", side_effect=train), \
                    patch.object(models.QuantileModels, "fit", side_effect=AssertionError("selection called")):
                fitted = final.fit_fixed(rows, features, np.array([True, True, False, False]), reference)
        self.assertEqual(fitted.selection, selection)
        self.assertEqual(len(seen), 12)
        for labels, kwargs in seen:
            np.testing.assert_array_equal(labels, [1, 2])
            self.assertEqual(kwargs, {"num_boost_round": 7})

@unittest.skipUnless(HAS_DEPS, "model dependencies are required")
class FinalBoundaryTests(unittest.TestCase):
    def synthetic_run_fixture(self, tmp):
        from freeinference.split import FIRST_MS, WEEK_MS
        from tests.test_freeinference_features import HistoryFeatureTests
        fixture = HistoryFeatureTests()
        root = Path(tmp) / 'UNIQUE-PRIVATE-DATA-ROOT'
        paths = [root / str(i) for i in range(1, 17)]
        tables, weekly, histories, built_frames, session_frames = {}, [], [], [], []
        for number, path in enumerate(paths, 1):
            requests = fixture.rows()
            requests['session_id'] = 'SECRET-SESSION'
            requests['start_ms'] += FIRST_MS + (number - 1) * WEEK_MS + 1000
            requests['end_ms'] += FIRST_MS + (number - 1) * WEEK_MS + 1000
            sessions = pd.DataFrame({'session_id': ['SECRET-SESSION'], 'harness': ['h'],
                                     'parent_session_id': [None], 'user_id': ['SECRET-USER']})
            calls = fixture.calls().assign(session_id='SECRET-SESSION')
            tables[path] = {'requests': requests, 'sessions': sessions, 'tool_calls': calls}
            built = final.build_features(requests, sessions, calls)
            rows = requests[['session_id', 'step', 'start_ms', 'end_ms', 'link_reason', 'model', 'completion_tokens']].copy()
            rows['week'] = number
            rows['harness'] = built['A'].harness
            weekly.append(rows)
            histories.append(final.build_history(rows))
            built_frames.append(built['B'])
            session_frames.append(sessions.assign(week=number))
        rows = pd.concat(weekly, ignore_index=True)
        history = pd.concat(histories, ignore_index=True)
        sessions = pd.concat(session_frames, ignore_index=True)
        rows['account'] = 'SECRET-USER'
        cross = final.build_cross_session_features(rows, sessions)
        cross.attrs.clear()
        b = pd.concat([pd.concat(built_frames, ignore_index=True), cross], axis=1)
        features = {name: b.loc[:, list(columns)] for name, columns in FEATURE_SETS.items()}
        masks = final.final_masks(rows)
        setting = {'num_leaves': 5, 'min_data_in_leaf': 1, 'learning_rate': .05,
                   'target_transform': 'identity'}
        selection = {name: {q: {'settings': setting, 'iterations': 2}
                            for q in PREDICTION_COLUMNS} for name in FEATURE_SETS}
        reference = Path(tmp) / 'models.json'
        reference.write_text(json.dumps({'selection': selection}))
        model = final.fit_fixed(rows, features, masks.train_target, reference)
        baseline = final.Baselines().fit(rows, history, masks.train_target)
        mask = masks.validation
        predictions, _ = model.predict({name: frame.loc[mask] for name, frame in features.items()})
        base, _ = baseline.predict(rows.loc[mask], history.loc[mask])
        reference_data = {'selection': selection,
                          'metrics': final.evaluate_predictions(rows.loc[mask], {**base, **predictions}),
                          'bootstrap': {}, 'robustness': {}}
        reference.write_text(json.dumps(reference_data))
        demand_reference = Path(tmp) / 'demand.json'
        demand_reference.write_text(json.dumps({'metrics': {'overall': {}}, 'bootstrap': {}}))
        checked = {'release/fixture/processed/requests.parquet': 'a' * 64}
        manifests = {'fixture': {'sha256': 'b' * 64, 'trace_hashes': {}}}
        def load(path, table, columns, allow_test=False):
            self.assertTrue(allow_test)
            return tables[path][table].copy()
        return root, paths, reference, demand_reference, checked, manifests, load, reference_data

    def test_run_writes_private_safe_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, paths, ref, demand_ref, checked, manifests, load, _ = self.synthetic_run_fixture(tmp)
            output = Path(tmp) / 'output'
            with patch.object(final, 'REFERENCE', ref), patch.object(final, 'DEMAND_REFERENCE', demand_ref), \
                 patch.object(final, 'validate_data', return_value=(paths, checked, manifests)), \
                 patch.object(final, 'load_table', side_effect=load):
                result = final.run(root=root, output=output)
            for name in ('final.json', 'final.md', 'demand.png'):
                self.assertTrue((output / name).is_file())
            for name in ('final.json', 'final.md'):
                content = (output / name).read_text()
                for secret in ('SECRET-SESSION', 'SECRET-USER', str(root), 'UNIQUE-PRIVATE-DATA-ROOT'):
                    self.assertNotIn(secret, content)
            self.assertEqual(json.loads((output / 'final.json').read_text()), result)
            primary = result['test']['primary']
            self.assertIn(primary['conclusion'], ('test에서 B가 L보다 개선', '차이 미확인', 'test에서 악화'))
            self.assertEqual(primary['ci95'], result['test']['bootstrap']['B_minus_L']['groups']['overall']['differences']['B']['request']['mean_pinball'])
            for key in ('processed_sha256', 'manifests', 'code_revision', 'source_sha256', 'revision', 'executed_at_utc'):
                self.assertTrue(result[key])
            self.assertEqual(result['processed_sha256'], checked)
            self.assertEqual(result['manifests'], manifests)

    def test_validation_mismatch_stops_before_test_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, paths, ref, demand_ref, checked, manifests, load, reference = self.synthetic_run_fixture(tmp)
            reference['metrics']['overall']['metrics']['B']['request']['mean_pinball'] += 1
            ref.write_text(json.dumps(reference))
            original = final.QuantileModels.predict
            with patch.object(final, 'REFERENCE', ref), patch.object(final, 'DEMAND_REFERENCE', demand_ref), \
                 patch.object(final, 'validate_data', return_value=(paths, checked, manifests)), \
                 patch.object(final, 'load_table', side_effect=load), \
                 patch.object(final, 'comparisons') as comparisons, \
                 patch.object(final.QuantileModels, 'predict', autospec=True, side_effect=original) as predict:
                with self.assertRaisesRegex(ValueError, 'validation reproduction failed'):
                    final.run(root=root, output=Path(tmp) / 'output')
                self.assertEqual(predict.call_count, 1)
                comparisons.assert_not_called()

    def test_missing_test_account_stops_before_fit_or_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, paths, ref, demand_ref, checked, manifests, load, _ = self.synthetic_run_fixture(tmp)
            def load_missing_account(path, table, columns, allow_test=False):
                result = load(path, table, columns, allow_test=allow_test)
                if path == paths[13] and table == 'sessions':
                    result['user_id'] = None
                return result
            with patch.object(final, 'REFERENCE', ref), \
                 patch.object(final, 'validate_data', return_value=(paths, checked, manifests)), \
                 patch.object(final, 'load_table', side_effect=load_missing_account), \
                 patch.object(final, 'fit_fixed') as fit, \
                 patch.object(final.QuantileModels, 'predict') as predict, \
                 patch.object(final, 'evaluate_predictions') as evaluate:
                with self.assertRaisesRegex(ValueError, 'Missing test account'):
                    final.run(root=root, output=Path(tmp) / 'output')
                fit.assert_not_called()
                predict.assert_not_called()
                evaluate.assert_not_called()

    def test_invalid_test_rows_are_rejected(self):
        rows = pd.DataFrame({'week': [14], 'session_id': ['s'], 'account': ['a'], 'completion_tokens': [1.]})
        final.validate_test_rows(rows, [True])
        for changed, mask in ((rows, [False]), (rows.assign(session_id=None), [True]),
                              (rows.assign(completion_tokens=-1), [True]),
                              (rows.assign(completion_tokens=float('inf')), [True]),
                              (rows.assign(completion_tokens=None), [True])):
            with self.subTest(changed=changed, mask=mask), self.assertRaises(ValueError):
                final.validate_test_rows(changed, mask)

    def test_plot_failure_leaves_incomplete_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, paths, ref, demand_ref, checked, manifests, load, _ = self.synthetic_run_fixture(tmp)
            output = Path(tmp) / 'output'
            with patch.object(final, 'REFERENCE', ref), patch.object(final, 'DEMAND_REFERENCE', demand_ref), \
                 patch.object(final, 'validate_data', return_value=(paths, checked, manifests)), \
                 patch.object(final, 'load_table', side_effect=load), \
                 patch('matplotlib.figure.Figure.savefig', side_effect=OSError('plot failure')):
                with self.assertRaisesRegex(OSError, 'plot failure'):
                    final.run(root=root, output=output)
            self.assertEqual(json.loads((output / 'final.json').read_text()), {'status': 'incomplete'})

    def test_existing_output_refuses_before_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('final.json', 'final.md', 'demand.png'):
                artifact = Path(tmp) / name
                artifact.write_text('fixture')
                with self.assertRaises(FileExistsError):
                    final.run(root=Path(tmp) / 'missing', output=tmp)
                artifact.unlink()

    def test_processed_hash_validation_and_missing_file(self):
        import hashlib
        from datetime import timedelta
        from freeinference.loader import FIRST_WEEK, REVISION
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'REVISION').write_text(REVISION)
            expected = {}
            for i in range(16):
                start = FIRST_WEEK + timedelta(weeks=i)
                week = root / 'release' / f'{start}_{start + timedelta(days=6)}'
                (week / 'processed').mkdir(parents=True)
                if i >= 13:
                    (week / 'manifest.json').write_text(json.dumps({
                        'format_version': 1, 'week': week.name, 'tokenizer': 'fixture',
                        'block_tokens': 1, 'salt_series': 'fixture', 'excluded_sessions': {},
                        'distinct_block_ids': 0,
                        'files': {f'traces/{start}.jsonl': {'bytes': 1, 'sha256': 'a' * 64, 'lines': 1}}}))
                    for name in final.TABLE_NAMES:
                        path = week / 'processed' / f'{name}.parquet'
                        path.write_bytes(b'fixture')
                        expected[path.relative_to(root).as_posix()] = hashlib.sha256(b'fixture').hexdigest()
            (root / 'expected_sha256.json').write_text(json.dumps(expected))
            manifest = week / 'manifest.json'
            contents = manifest.read_text()
            manifest.write_text('{}')
            with self.assertRaises(ValueError):
                final.validate_data(root)
            manifest.write_text(contents)
            _, checked, _ = final.validate_data(root)
            self.assertEqual(checked, expected)
            malformed = json.loads(contents)
            malformed['week'] = 'wrong'
            manifest.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                final.validate_data(root)
            malformed['week'] = week.name
            del next(iter(malformed['files'].values()))['sha256']
            manifest.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                final.validate_data(root)
            manifest.write_text(contents)
            missing = dict(expected)
            missing.pop(next(iter(missing)))
            (root / 'expected_sha256.json').write_text(json.dumps(missing))
            with self.assertRaises(ValueError):
                final.validate_data(root)
            (root / 'expected_sha256.json').write_text(json.dumps(expected))
            target = root / next(iter(expected))
            target.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                final.validate_data(root)
            target.unlink()
            with self.assertRaises(FileNotFoundError):
                final.validate_data(root)

    def test_failed_attempt_reserves_output_without_path_or_identifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out'
            with patch.object(final, 'validate_data', side_effect=ValueError('failure')):
                with self.assertRaises(ValueError):
                    final.run(root=Path(tmp) / 'private-session-id', output=output)
            self.assertEqual(json.loads((output / 'final.json').read_text()), {'status': 'incomplete'})
            with self.assertRaises(FileExistsError):
                final.run(root=tmp, output=output)

    def test_test_folder_session_before_boundary_is_rejected(self):
        rows = pd.DataFrame({'week': [14], 'session_id': ['s'],
                             'start_ms': [final.TEST_START_MS - 1],
                             'end_ms': [final.TEST_START_MS + 1]})
        with self.assertRaises(ValueError):
            final.final_masks(rows)

    def test_test_opt_in_only_in_final_module(self):
        import ast
        for path in Path(final.__file__).parent.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg == 'allow_test' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            self.assertEqual(path.name, 'final.py')

    def test_test_sessions_do_not_enter_training(self):
        from freeinference.split import FIRST_MS, WEEK_MS
        rows = pd.DataFrame({'week': [1, 11, 14], 'session_id': ['a', 'b', 'c'],
                             'start_ms': [FIRST_MS + i * WEEK_MS for i in (0, 10, 13)],
                             'end_ms': [FIRST_MS + i * WEEK_MS + 1 for i in (0, 10, 13)]})
        masks = final.final_masks(rows)
        self.assertEqual(masks.train_target.tolist(), [True, False, False])
        self.assertEqual(masks.validation.tolist(), [False, True, False])
        self.assertEqual(masks.test.tolist(), [False, False, True])

    def test_test_response_only_affects_strictly_later_history_and_predictions(self):
        from tests.test_freeinference_features import HistoryFeatureTests
        from pandas.testing import assert_frame_equal
        fixture = HistoryFeatureTests()
        rows = fixture.rows()
        rows['start_ms'] += final.TEST_START_MS
        rows['end_ms'] += final.TEST_START_MS
        rows.loc[1, 'end_ms'] = final.TEST_START_MS + 350
        rows['week'] = 14
        # A second session sees account history; its first request is in flight
        # when the changed response finishes, and its second starts afterwards.
        extra = rows.iloc[[2, 4]].copy()
        extra['session_id'] = 'other'
        extra['step'] = [0, 1]
        rows = pd.concat([rows, extra], ignore_index=True)
        sessions = pd.DataFrame({'session_id': ['parent', 'other'], 'harness': ['h', 'h'],
                                 'parent_session_id': [None, None], 'user_id': ['u', 'u'],
                                 'week': [14, 14]})
        self.assertTrue(final.final_masks(rows).test.all())
        def features(data):
            b = final.build_features(data, sessions, fixture.calls().assign(week=14))['B']
            cross = final.build_cross_session_features(data, sessions)
            cross.attrs.clear()
            combined = pd.concat([b, cross], axis=1)
            return {name: combined.loc[:, list(columns)] for name, columns in FEATURE_SETS.items()}
        original = features(rows)
        altered = rows.copy()
        altered.loc[1, 'completion_tokens'] = 9999
        changed = features(altered)
        unchanged = [0, 1, 2, 3, 5]
        for name in FEATURE_SETS:
            assert_frame_equal(original[name].loc[unchanged], changed[name].loc[unchanged])
        self.assertFalse(original['B'].loc[4].equals(changed['B'].loc[4]))
        self.assertFalse(original['C'].loc[6].equals(changed['C'].loc[6]))
        setting = {'num_leaves': 5, 'min_data_in_leaf': 1, 'learning_rate': .05,
                   'target_transform': 'identity'}
        selection = {name: {q: {'settings': setting, 'iterations': 2}
                            for q in PREDICTION_COLUMNS} for name in FEATURE_SETS}
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'models.json'
            reference.write_text(json.dumps({'selection': selection}))
            fitted = final.fit_fixed(rows, original, [True, False, True, True, True, True, True], reference)
            before, _ = fitted.predict(original)
            after, _ = fitted.predict(changed)
        for name in FEATURE_SETS:
            assert_frame_equal(before[name].loc[unchanged], after[name].loc[unchanged])
