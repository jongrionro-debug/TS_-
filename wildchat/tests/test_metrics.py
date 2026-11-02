import math
import unittest

from wildchat.metrics import baseline, empirical_quantile, evaluate


class MetricsTest(unittest.TestCase):
    def test_inverse_empirical_cdf_and_training_baseline(self):
        self.assertEqual(baseline([4, 1, 3, 2]), [1, 2, 4])
        self.assertEqual(empirical_quantile([1, 2, 3, 4], .5), 2)
        with self.assertRaises(ValueError):
            baseline([])

    def test_hand_calculated_metrics(self):
        result = evaluate([0, 4], [[1, 2, 3], [1, 2, 3]])
        self.assertEqual(result['mae_p50'], 2)
        self.assertAlmostEqual(result['mean_absolute_log1p_error_p50'], (math.log(3) + math.log(5/3)) / 2)
        for key, expected in zip(('p10', 'p50', 'p90'), (.6, 1, .6)):
            self.assertAlmostEqual(result['pinball_loss'][key], expected)
        self.assertAlmostEqual(result['mean_pinball_loss'], 2.2 / 3)
        self.assertEqual(result['p10_p90_coverage'], 0)
        self.assertEqual(result['mean_interval_width'], 2)
        self.assertEqual(evaluate([1, 3], [[1, 2, 3]] * 2)['p10_p90_coverage'], 1)

    def test_invalid_inputs_and_crossing_rejected(self):
        for targets, predictions in [([], []), ([1], []), ([-1], [[0, 1, 2]]),
                                     ([1], [[2, 1, 3]]), ([1], [[0, 1, float('nan')]])]:
            with self.subTest(targets=targets, predictions=predictions), self.assertRaises(ValueError):
                evaluate(targets, predictions)


if __name__ == '__main__':
    unittest.main()
