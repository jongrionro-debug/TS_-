"""P0 causal feature and paired cluster-bootstrap checks, synthetic data only."""
import math
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from wildchat.alfredo_h1_features import h1_features

class CausalityTests(unittest.TestCase):
    def same(self, a, b):
        self.assertEqual(len(a), len(b))
        for av, bv in zip(a, b):
            self.assertTrue(math.isnan(av) and math.isnan(bv) or av == bv)

    def test_known_values_ewma_std_difference_slope_ratios(self):
        actual = h1_features([1] * 4, [0, 1, 3, 4], [10, 20, 10, 40], [20, 60, 40, 100])
        self.assertTrue(all(math.isnan(v) for v in actual[0][:-1]))
        self.assertEqual(actual[0][-1], 0)
        self.same(actual[1], [20, 20, 0, math.nan, math.nan, 2, 2, 2, 1])
        self.same(actual[2], [32, 40, 20, 40, 40, 3, 2.3, .5, 2])
        self.assertAlmostEqual(actual[3][0], 34.4)
        self.assertAlmostEqual(actual[3][1], 40)
        self.assertAlmostEqual(actual[3][2], math.sqrt(800 / 3))
        self.same(actual[3][3:], [-20, -10, 4, .3 * 4 + .7 * 2.3, 4, 3])

    def test_current_and_all_future_responses_cannot_change_prefix(self):
        sources, turns, inputs, outputs = [1] * 5, list(range(5)), [2, 3, 5, 7, 11], [10, 20, 30, 40, 50]
        original = h1_features(sources, turns, inputs, outputs)
        for current in range(5):
            changed_y = outputs[:current] + [9999] * (5 - current)
            changed_x = inputs[:current + 1] + [8888] * (4 - current)
            changed = h1_features(sources, turns, changed_x, changed_y)
            for i in range(current + 1):
                self.same(original[i], changed[i])
            truncated = h1_features(sources[:current + 1], turns[:current + 1],
                                    inputs[:current + 1], outputs[:current + 1])
            self.same(original[current], truncated[-1])
        self.assertNotEqual(original[1][0], h1_features(sources, turns, inputs, [9999] + outputs[1:])[1][0])

    def test_interleaved_conversations_do_not_share_history(self):
        sources, turns, inputs, outputs = [1, 2, 1, 2, 1], [0, 0, 2, 1, 3], [10] * 5, [10, 9999, 30, 8888, 50]
        together = h1_features(sources, turns, inputs, outputs)
        alone = h1_features([1] * 3, [0, 2, 3], [10] * 3, [10, 30, 50])
        for a, b in zip([together[i] for i in (0, 2, 4)], alone):
            self.same(a, b)

    def test_current_prompt_is_available_only_input_multiplier_changes(self):
        first = h1_features([1, 1], [0, 1], [10, 20], [20, 50])[1]
        changed = h1_features([1, 1], [0, 1], [10, 100], [20, 50])[1]
        self.same(first[:7] + first[8:], changed[:7] + changed[8:])
        self.assertEqual((first[7], changed[7]), (2, 10))

    def test_zero_denominators_and_zero_output_are_well_defined(self):
        actual = h1_features([1] * 4, range(4), [10, 0, 20, 10], [0, 60, 40, 50])
        self.assertEqual(actual[1][5:8], [0, 0, 0])
        self.assertTrue(math.isnan(actual[2][5]))
        self.assertTrue(math.isnan(actual[2][7]))
        self.assertEqual(actual[2][6], 0)
        self.assertAlmostEqual(actual[3][6], .6)
        self.assertEqual(actual[3][-1], 3)

    def test_invalid_prior_pairs_are_skipped_with_original_turn_gap(self):
        from wildchat import prepare
        from wildchat.tests.test_prepare import CharacterEncoding, message
        source = {"conversation": [message("user", "aa"), message("assistant", "bbbb"),
                    message("user", "unanswered"), message("user", "cc"), message("assistant", "dddddddd"),
                    message("user", "ee"), message("assistant", "ff")]}
        samples = list(prepare.iter_samples(source, CharacterEncoding()))
        turns = [s["history_features"]["previous_user_turns"] for s in samples]
        self.assertEqual(turns, [0, 2, 3])
        actual = h1_features([1] * 3, turns,
                             [s["prompt_features"]["input_tokens"] for s in samples],
                             [s["target"]["output_tokens"] for s in samples])
        self.assertEqual(actual[1][-1], 1)
        self.assertEqual(actual[2][3:5], [4, 2])

    def test_out_of_order_or_invalid_numeric_inputs_fail(self):
        for turns in ([1, 1], [1, 0], [-1, 0]):
            with self.assertRaises(ValueError):
                h1_features([1, 1], turns, [10, 10], [20, 20])
        with self.assertRaises(ValueError):
            h1_features([1], [0], [10], [-1])
        with self.assertRaises(ValueError):
            h1_features([1], [0], [10], [math.inf])
        with self.assertRaises(ValueError):
            h1_features([1, 1], [0, 1], [10], [20, 20])


@unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None and
                     importlib.util.find_spec("numpy") is not None,
                     "requires WildChat numerical environment")
class P0Tests(unittest.TestCase):
    def test_lag_only_skips_invalid_turns_and_is_causal_per_source(self):
        import numpy as np
        from wildchat.alfredo_p0 import lag1_features
        source, turns, y = [1, 2, 1, 1, 2], [0, 0, 3, 4, 2], [20, 9999, 40, 50, 8888]
        a = lag1_features(source, turns, y).ravel()
        np.testing.assert_allclose(a, [np.nan, np.nan, 20, 40, 9999], equal_nan=True)
        np.testing.assert_allclose(lag1_features(source, turns, [20, 9999, 400, 500, 800]).ravel()[:3],
                                   a[:3], equal_nan=True)
        for bad_turns in ([0, 0], [1, 0]):
            with self.assertRaises(ValueError):
                lag1_features([1, 1], bad_turns, [2, 3])

    def test_paired_bootstrap_against_explicit_cluster_resampling(self):
        import numpy as np
        from wildchat import alfredo_p0 as p0
        rows = SimpleNamespace(users=np.array(["a", "a", "a", "b", "b", "c", "c", "c"]),
                               turns=np.array([0, 1, 2, 0, 1, 0, 1, 2]),
                               y=np.array([10., 30., 80., 50., 10., 100., 20., 30.]))
        base = np.column_stack([rows.y*.5, rows.y*.8, rows.y*1.2])
        predictions = {m: base + i for i, m in enumerate(p0.MODELS)}
        points, ci = p0.score_bootstrap(rows, predictions, replicates=50)
        users = np.unique(rows.users)
        rng = np.random.default_rng(42)
        raw = {(c, w): [] for c in ("first_turn", "followup_turn", "all") for w in ("row", "user")}
        for _ in range(50):
            sampled = rng.integers(0, 3, size=3)
            for c, mask in p0.rv.cohorts(rows).items():
                for w in ("row", "user"):
                    model_losses = []
                    for m in p0.MODELS:
                        loss = np.abs(rows.y - predictions[m][:, 1])
                        clusters = [loss[(rows.users == users[u]) & mask] for u in sampled]
                        score = (np.concatenate(clusters).mean() if w == "row" else
                                 np.mean([v.mean() for v in clusters if len(v)]))
                        model_losses.append(score)
                    raw[c, w].append(model_losses)
        for (c, w), draws in raw.items():
            v = np.asarray(draws)
            for a, b in p0.PAIRS:
                ia, ib = p0.MODELS.index(a), p0.MODELS.index(b)
                expected = np.quantile(v[:, ia]-v[:, ib], [.025, .975])
                np.testing.assert_allclose(ci[c][w]["differences"][a+"-"+b]["mae_p50"]["ci95"], expected)
            explicit = np.mean([np.abs(rows.y[rows.users == u] - predictions["B"][rows.users == u, 1]).mean()
                                for u in users])
            self.assertAlmostEqual(points["all"]["user"]["B"]["mae_p50"], explicit)

    def test_jsonl_guard_and_embedding_alignment_fail_closed(self):
        import numpy as np
        from wildchat.alfredo_p0_support import allowed_jsonl_reads, load_embeddings, rv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, key in (("train", (1, 0)), ("validation", (2, 0))):
                (root / (split+".jsonl")).write_text(json.dumps({"metadata":
                    {"source_row_id": key[0], "user_message_index": key[1]}})+"\n")
            cache = root / "cache.npz"
            np.savez(cache, keys=np.array([[2,0],[1,0]]), vectors=np.array([[2.]*384,[1.]*384]))
            with allowed_jsonl_reads(root) as opened:
                values, audit = load_embeddings(root, cache, rv.fingerprint(cache))
                self.assertEqual(values["train"][0,0], 1)
                self.assertTrue(audit["exact_train_validation_coverage"])
                with self.assertRaises(PermissionError):
                    (root / "forbidden.jsonl").read_text()
                with self.assertRaises(ValueError):
                    load_embeddings(root, cache, "bad")
            self.assertEqual(opened, {"train.jsonl", "validation.jsonl"})
            np.savez(cache, keys=np.array([[1,0],[1,0]]), vectors=np.ones((2,384)))
            with self.assertRaises(ValueError):
                load_embeddings(root, cache, rv.fingerprint(cache))

    def test_reference_dependency_change_rejected_before_data_access(self):
        from wildchat import alfredo_p0 as p0
        reference = json.loads((p0.ROOT / "wildchat/alfredo_p0_reference.json").read_text())
        args = SimpleNamespace(**reference["settings"])
        self.assertEqual(p0.check_reference(reference, args), reference["versions"])
        with mock.patch.object(p0.rv, "fingerprint", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "dependency mismatch"):
                p0.check_reference(reference, args)

    def test_metadata_projection_and_kst_rollover(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datetime import datetime, timezone
        from wildchat import alfredo_p0 as p0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.parquet"
            pq.write_table(pa.table({"timestamp": [datetime(2023,1,1,18,tzinfo=timezone.utc)],
                "hashed_ip": ["synthetic"], "model": ["toy"], "turn": [2],
                "conversation": ["DO NOT READ PAYLOAD"]}), path)
            original = pq.ParquetFile
            projected = []
            class Guard:
                def __init__(self, path):
                    self.p = original(path)
                    self.metadata = self.p.metadata
                def iter_batches(self, *, columns, **kwargs):
                    projected.append(columns)
                    self_outer.assertEqual(columns, ["timestamp", "hashed_ip", "model", "turn"])
                    return self.p.iter_batches(columns=columns, **kwargs)
            self_outer = self
            with mock.patch.object(pq, "ParquetFile", Guard):
                r = p0.metadata_summary([path])
            self.assertEqual(r["recorded_at_min_kst"], "2023-01-02T03:00:00+09:00")
            self.assertEqual(r["recorded_hour_kst_by_conversation_rows"]["3"], 1)
            self.assertTrue(projected)
            self.assertNotIn("synthetic", json.dumps(r))

    def test_pca_never_sees_validation_fit_and_current_target_is_not_design_input(self):
        import numpy as np
        from wildchat import alfredo_p0 as p0
        pca = p0.PCA().fit(np.array([[0.,0.],[1.,2.],[2.,1.]]), 1)
        before = pca.mean.copy()
        rows = SimpleNamespace(x=np.ones((2, 3+9+1+2)), y=np.array([1,2]))
        a = p0.designs(rows, 3, pca)
        rows.y[:] = 9999
        b = p0.designs(rows, 3, pca)
        for model in a:
            np.testing.assert_equal(a[model], b[model])
        np.testing.assert_equal(pca.mean, before)

    def test_bin_boundary_and_totals(self):
        from wildchat.alfredo_p0 import toy_bins
        self.assertEqual(toy_bins([(0,1,2),(9.999,3,4),(10,5,6),(20,7,8)]),
                         {0:[4,6],10:[5,6],20:[7,8]})

    def test_output_creation_never_overwrites_existing_file(self):
        from wildchat.alfredo_p0 import save_json, save_text
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text("original")
            for function, value in ((save_json, {}), (save_text, "new")):
                with self.assertRaises(FileExistsError):
                    function(path, value)
            self.assertEqual(path.read_text(), "original")
