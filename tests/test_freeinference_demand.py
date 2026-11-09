"""Synthetic request-start demand aggregation and day bootstrap contracts."""
import importlib.util
import unittest

HAS_DEPS = all(importlib.util.find_spec(n) for n in ('numpy', 'pandas', 'pyarrow', 'lightgbm', 'matplotlib'))
if HAS_DEPS:
    import numpy as np
    import pandas as pd
    from freeinference import demand, eda
    from freeinference.evaluate import cluster_draws


@unittest.skipUnless(HAS_DEPS, 'analysis dependencies required')
class DemandTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({'start_ms': [0, 600000, 1800000], 'completion_tokens': [10, 20, 30], 'prompt_tokens': [1, 2, 3]})
        self.pred = pd.DataFrame({'P10': [5, 10, 20], 'P50': [8, 22, 25], 'P90': [12, 25, 35]})

    def test_bins_and_metrics(self):
        result = demand.aggregate_predictions(self.rows, {'B': self.pred})
        expected = eda.aggregate_10min(self.rows)
        np.testing.assert_array_equal(result.actual, expected.completion_tokens)
        self.assertEqual(result['count'].tolist(), [1, 1, 0, 1])
        self.assertEqual(result.B_sum_P10.tolist(), [5, 10, 0, 20])
        self.assertEqual(result.B_sum_P50.tolist(), [8, 22, 0, 25])
        self.assertEqual(result.B_sum_P90.tolist(), [12, 25, 0, 35])
        score = demand.metrics(result, 'B')
        self.assertEqual(score['mae'], 2.25)
        self.assertEqual(score['wape'], 9 / 60)
        self.assertEqual(score['bias'], -1.25)
        self.assertEqual(score['coverage'], 1)
        self.assertEqual(score['mean_width'], 9.25)
        changed = self.rows.copy(); changed.completion_tokens += 100
        other = demand.aggregate_predictions(changed, {'B': self.pred})
        np.testing.assert_array_equal(result.filter(like='_sum_'), other.filter(like='_sum_'))
        self.assertFalse(result.actual.equals(other.actual))

    def test_day_bootstrap(self):
        rows = pd.concat([self.rows.assign(start_ms=self.rows.start_ms + day * 86400000) for day in range(3)], ignore_index=True)
        pred = pd.concat([self.pred] * 3, ignore_index=True)
        frame = demand.aggregate_predictions(rows, {'B': pred, 'L': pred})
        first = demand.bootstrap(frame, 'B', 'L', B=50, seed=3)
        self.assertEqual(first, demand.bootstrap(frame, 'B', 'L', B=50, seed=3))
        self.assertEqual(first['mae_ci95'], [0, 0]); self.assertEqual(first['wape_ci95'], [0, 0])
        self.assertEqual(first['blocks'], 3)
        draws = np.concatenate(list(cluster_draws(3, B=50, seed=3)))
        codes = pd.factorize(frame.index.floor('D'))[0]
        for draw in draws:
            weights = draw[codes]
            for day in range(3):
                self.assertEqual(len(set(weights[codes == day])), 1)


if __name__ == '__main__':
    unittest.main()
