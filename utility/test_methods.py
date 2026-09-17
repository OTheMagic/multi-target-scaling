"""Small, in-memory extraction checks. Run: python -B -m utility.test_methods.

Legacy imports below were inspected: they contain definitions/imports only.
No notebook or experiment driver is imported. Agreement tests establish
preservation; separate hand examples check the cited papers and project definitions.
Documented tie-policy changes are checked explicitly rather than called legacy parity.
"""
import importlib
import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from utility import methods
from utility.conformal_utils import conformal_quantile, conformal_rank
from utility.signed_geometry import interval_sup, link, score_at
from utility.tie_breaking import (
    conformal_quantile_pair, coordinate_quantile_pairs, lex_le, lex_order,
)


class MethodChecks(unittest.TestCase):
    def test_imports(self):
        for name in ("unscaled", "ec", "point_chr", "cqhr", "gwc", "envelope", "additional"):
            importlib.import_module("utility.methods." + name)

    def test_legacy_agreement_rectangles(self):
        from utility.data_splitting import data_spliting_CHR_prediction
        pairs = [
            (methods.data_spliting_CHR_prediction, data_spliting_CHR_prediction),
        ]
        scores = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.],
                           [2., 2.], [5., 6.], [6., 5.]])
        for new, old in pairs:
            for alpha in (.5, .2):
                actual = new(scores, alpha=alpha)
                np.testing.assert_array_equal(actual.info(), old(scores, alpha=alpha).info())
                self.assertEqual(actual.upper.shape, (2,))
                self.assertEqual(actual.contain_points(scores).shape, (7,))

    def test_legacy_agreement_cqhr(self):
        from utility.cqhr import cqhr_adjustments
        scores = np.array([[1., 1.], [-1., -3.], [3., 2.]])
        lengths = np.array([[2., 4.], [4., 2.], [2., 2.]])
        test_lengths = np.array([[2., 6.], [4., 2.]])
        for reference in (0, 1):
            actual = methods.cqhr_adjustments(scores, lengths, test_lengths, .5, reference)
            expected = cqhr_adjustments(scores, lengths, test_lengths, .5, reference)
            for a, b in zip((actual.adjustments, actual.quantile_adjustment,
                             actual.calibration_scores), expected):
                np.testing.assert_array_equal(a, b)

    def test_legacy_agreement_envelope_and_signed_gwc(self):
        from utility.envelope import EnvelopeCalibration
        cases = [
            (np.array([[0., 2.], [1., 0.], [2., 4.]]), .5, 0.),
            (np.array([[-2., 1.], [-1., -2.], [0., 3.]]), .5, [-2., -1.]),
            (np.array([[-1.], [0.], [1.]]), .5, -np.inf),
            (np.array([[0.], [0.], [2.]]), .5, 0.),
            (np.array([[0.], [1.], [2.]]), .5, 10.),
            (np.ones((3, 2)), .5, 0.),
            (np.array([[1., 2.]]), .1, 0.),
        ]
        for scores, alpha, lower in cases:
            old_cal = EnvelopeCalibration(scores, alpha)
            keys = np.linspace(.1, .9, len(scores))
            new_cal = methods.EnvelopeCalibration(scores, alpha, scalar_keys=keys)
            for search in ("backward", "exhaustive", "rank"):
                old, old_cells = old_cal.predict(lower, search=search, return_cells=True)
                new, new_cells = new_cal.predict(lower, search=search, return_cells=True, test_key=0.)
                for attr in ("lower", "upper", "gwc_upper"):
                    np.testing.assert_array_equal(getattr(new, attr), getattr(old, attr))
                for attr in ("q_gwc", "empty", "evaluations", "fallback"):
                    self.assertEqual(getattr(new, attr), getattr(old, attr))
                # Explicit smallest candidate key recovers closed legacy boundaries.
                for actual, expected in zip(new_cells, old_cells):
                    self.assertEqual([{k: c[k] for k in expected[i]} for i, c in enumerate(actual)],
                                     expected)
            gwc = methods.gwc_prediction(scores, alpha, lower, scalar_keys=keys, test_key=0.)
            np.testing.assert_array_equal(gwc.upper, old.gwc_upper)
            self.assertEqual(gwc.q_gwc, old.q_gwc)
            self.assertEqual(gwc.fallback, old.fallback)
            self.assertEqual(gwc.empty, bool(np.any(gwc.upper < gwc.lower)
                                             or np.any(np.isneginf(gwc.upper))))

    def test_conformal_rank_definition(self):
        # Project quantile: ceil((n+1)(1-alpha)), with +inf above n.
        self.assertEqual(conformal_rank(3, .5), 2)
        self.assertEqual(conformal_quantile([9., 1., 4.], .5), 4.)
        self.assertTrue(np.isposinf(conformal_quantile([1.], .1)))

    def test_conformal_rank_decimal_boundaries(self):
        # Exact decimal positions: 7, 7.00025, and 6.99975. Two-decimal
        # rounding would incorrectly lower the middle rank from eight to seven.
        for alpha, expected in ((.72, 7), (.71999, 8), (.72001, 7),
                                (np.nextafter(.72, 0.), 8),
                                (np.nextafter(.72, 1.), 7)):
            with self.subTest(alpha=alpha):
                self.assertEqual(conformal_rank(24, alpha), expected)
        self.assertEqual(conformal_rank(9, .7), 3)
        self.assertEqual(conformal_rank(9, .1), 9)
        # The same convention controls the finite/infinite cutoff boundary.
        self.assertEqual(conformal_quantile([1., 2., 3., 4.], .2), 4.)
        self.assertTrue(np.isposinf(conformal_quantile(
            [1., 2., 3., 4.], np.nextafter(.2, 0.))))

    def test_conformal_rank_exact_fraction(self):
        self.assertEqual(conformal_rank(2, Fraction(1, 3)), 2)
        # A rounded float cannot recover the intended exact rational 1/3.
        self.assertEqual(conformal_rank(2, 1 / 3), 3)
        self.assertEqual(conformal_rank(np.int64(24), np.float64(.72)), 7)

    def test_decimal_rank_quantile_integration(self):
        scores = np.arange(1., 25.)
        keys = np.linspace(.1, .9, len(scores))
        self.assertEqual(conformal_quantile(scores, .72), 7.)
        self.assertEqual(conformal_quantile_pair(scores, keys, .72), (7., keys[6]))
        values = scores[:, None] * [1., 2.]
        upper, upper_keys = coordinate_quantile_pairs(
            values, np.column_stack([keys, keys]), .72)
        np.testing.assert_array_equal(upper, [7., 14.])
        np.testing.assert_array_equal(upper_keys, [keys[6], keys[6]])
        np.testing.assert_array_equal(
            methods.unscaled_prediction(scores[:, None], .72).upper, [7.])
        result = methods.cqhr_adjustments(scores[:, None], np.ones((24, 1)),
                                          np.ones((1, 1)), .72)
        np.testing.assert_array_equal(result.adjustments, [[7.]])
        self.assertEqual(methods.EnvelopeCalibration(scores[:, None], .72).rank, 7)

    def test_unscaled_manuscript_example(self):
        # supplementary.tex alg:unscaled: raw rank-three row maximum, no jitter.
        scores = np.array([[1., 2.], [3., 1.], [4., 2.], [5., 1.]])
        np.testing.assert_array_equal(methods.unscaled_prediction(scores, .5).upper,
                                      [4., 4.])

    def test_unscaled_documented_change_from_legacy(self):
        from utility.unscaled import unscaled_prediction
        scores = np.array([[1., 2.], [3., 1.], [4., 2.], [5., 1.]])
        delta = unscaled_prediction(scores, .5).upper - methods.unscaled_prediction(scores, .5).upper
        self.assertTrue(np.all((delta > 0) & (delta <= 1e-10)))

    def test_ec_paper_empirical_cdf(self):
        # Messoudi et al. equations (25)--(27): closed empirical CDFs, denominator n.
        from utility.copula import empirical_copula_prediction
        scores = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.]])
        copula = methods.EmpiricalCopula()
        copula.fit(scores)
        np.testing.assert_array_equal(copula.U, [[.25, 1.], [.5, .75], [.75, .5], [1., .25]])
        self.assertEqual(copula.cdf([.75, .75]), .5)
        np.testing.assert_array_equal(copula.quantile_box(.5), [.75, .75])
        region = methods.empirical_copula_prediction(scores, .5)
        np.testing.assert_array_equal(region.upper, [3., 3.])
        # Untied control agrees despite the different normalization of U.
        np.testing.assert_array_equal(region.info(), empirical_copula_prediction(scores, .5).info())

    def test_ec_tied_common_level_and_permutation(self):
        from utility.copula import empirical_copula_prediction
        scores = np.array([[0., 0.], [0., 1.], [0., 2.], [1., 3.]])
        copula = methods.EmpiricalCopula()
        keys = np.array([[.1, .2], [.3, .8], [.2, .1], [.4, .3]])
        copula.fit(scores, coordinate_keys=keys)
        np.testing.assert_array_equal(copula.U, [[.25, .25], [.75, .5], [.5, .75], [1., 1.]])
        self.assertEqual(copula.cdf([.5, .5]), .25)
        self.assertEqual(copula.cdf([.75, .75]), .75)
        # No diagonal level attains exactly .5: take the first reaching it.
        np.testing.assert_array_equal(copula.quantile_box(.5), [.75, .75])
        region = methods.empirical_copula_prediction(scores, .5, coordinate_keys=keys,
                                                    test_keys=[.2, .05])
        np.testing.assert_array_equal(region.upper, [0., 2.])
        self.assertTrue(region.contains([0., 2.]))
        for test_keys in ([.4, .05], [.2, .2]):
            rejected = methods.empirical_copula_prediction(scores, .5, coordinate_keys=keys,
                                                          test_keys=test_keys)
            self.assertFalse(rejected.contains([0., 2.]))
        np.testing.assert_array_equal(empirical_copula_prediction(scores, .5).upper, [0., 1.])
        for order in ([1, 0, 3, 2], [3, 2, 1, 0]):
            region = methods.empirical_copula_prediction(scores[order], .5,
                                                        coordinate_keys=keys[order], test_keys=[.2, .05])
            np.testing.assert_array_equal(region.upper, [0., 2.])
            self.assertTrue(region.contains([0., 2.]))

    def test_ec_inverse_grid_and_legacy_roundoff_fix(self):
        from utility.copula import empirical_copula_prediction
        scores = np.arange(1., 25.)[:, None]
        self.assertEqual(empirical_copula_prediction(scores, .72).upper[0], 8.)
        self.assertEqual(methods.empirical_copula_prediction(scores, .72).upper[0], 7.)
        scores = np.arange(1., 26.)[:, None]
        self.assertEqual(methods.inverse_ecdf_transform([7 / 25], scores)[0], 7.)
        self.assertEqual(methods.inverse_ecdf_transform([1.], scores)[0], 25.)
        for threshold in (0., -1., 1.1, np.nan, np.inf, -np.inf):
            self.assertTrue(np.isposinf(methods.inverse_ecdf_transform([threshold], scores)[0]))

    def test_point_chr_paper_algorithm_1(self):
        # Sampson--Chan Algorithm 1 uses each half's own rank. Seed42 splits 1..7 into
        # shape half [5,4,7], recalibration half [1,2,6,3]. Ranks are 3 and 4.
        # Base [7,14], excess quantile -1, final [6,12].
        scores = np.arange(1., 8.)[:, None] * [1., 2.]
        np.testing.assert_array_equal(methods.data_spliting_CHR_prediction(scores, .3).upper,
                                      [6., 12.])

    def test_cqhr_paper_algorithm_2(self):
        # Sampson--Chan Algorithm 2, starting with feasible outcomes/intervals.
        lengths = np.array([[2., 4.], [4., 2.], [2., 2.]])
        lower, upper = np.zeros((3, 2)), lengths
        outcomes = np.array([[3., 5.], [2., 1.], [5., 4.]])
        scores = np.maximum(lower - outcomes, outcomes - upper)
        np.testing.assert_array_equal(scores, [[1., 1.], [-2., -1.], [3., 2.]])
        result = methods.cqhr_adjustments(
            scores, lengths, [[2., 6.], [4., 2.]], alpha=.5)
        np.testing.assert_array_equal(result.calibration_scores, [1., -2., 3.])
        self.assertEqual(result.quantile_adjustment, 1.)
        np.testing.assert_array_equal(result.adjustments, [[1., 3.], [1., .5]])
        # Midpoints of width-two intervals have feasible signed score -1.
        negative = methods.cqhr_adjustments(-np.ones((3, 2)), 2 * np.ones((3, 2)),
                                             2 * np.ones((1, 2)), alpha=.5)
        self.assertEqual(negative.quantile_adjustment, -1.)
        np.testing.assert_array_equal(negative.adjustments, [[-1., -1.]])

    def test_randomized_ties_across_methods(self):
        self.assertEqual(conformal_quantile([1., 2., 2., 3.], .5), 2.)
        for test_key, accepted in ((.15, True), (.25, False)):
            unscaled = methods.unscaled_prediction(np.ones((3, 2)), .5,
                scalar_keys=[.1, .3, .2], test_key=test_key)
            np.testing.assert_array_equal(unscaled.upper, [1., 1.])
            self.assertEqual(unscaled.contains([1., 1.]), accepted)
            cqhr = methods.cqhr_adjustments(np.ones((3, 2)), np.ones((3, 2)),
                np.ones((1, 2)), .5, scalar_keys=[.1, .3, .2], test_keys=[test_key])
            self.assertEqual(bool(cqhr.contains([[1., 1.]])[0]), accepted)
            self.assertEqual(bool(cqhr.upper_closed[0]), accepted)
        # Seed42 recalibration indices for six rows: [0,1,5], cutoff key .2.
        for test_key, accepted in ((.15, True), (.25, False)):
            chr_region = methods.data_spliting_CHR_prediction(np.tile([1., 2.], (6, 1)), .5,
                scalar_keys=[.1, .3, .4, .5, .6, .2], test_key=test_key)
            self.assertEqual(chr_region.contains([1., 2.]), accepted)
        tied = np.array([[0.], [0.], [1.]])
        for test_key, accepted in ((.5, True), (.8, False)):
            gwc = methods.gwc_prediction(tied, .5, scalar_keys=[.2, .7, .4], test_key=test_key)
            self.assertEqual(gwc.contains([0.]), accepted)
            self.assertEqual(gwc.empty, not accepted)
            for search in ('backward', 'exhaustive', 'rank'):
                env = methods.envelope_prediction(tied, .5, search=search,
                    scalar_keys=[.2, .7, .4], test_key=test_key)
                self.assertEqual(env.contains([0.]), accepted)
                self.assertEqual(env.empty, not accepted)

    def test_random_keys_order_and_rank_calculation(self):
        near = np.array([1., np.nextafter(1., np.inf)])
        np.testing.assert_array_equal(lex_order(near, [.99, .01]), [0, 1])
        # Enumerate four possible held-out keys, not a simulation/coverage claim.
        keys = np.array([.1, .3, .6, .9])
        accepted = 0
        for i in range(4):
            q, key = conformal_quantile_pair(np.ones(3), np.delete(keys, i), .5)
            accepted += bool(lex_le(1., keys[i], q, key))
        self.assertEqual(accepted, 2)
        with self.assertRaises(ValueError):
            lex_order([1., 1.], [.2, .2])
        with self.assertRaises(ValueError):
            lex_le(1., .2, 1., .2)

    def test_envelope_local_keys_and_searches(self):
        # Hand-sorted local score/key pairs select keys .2 and .7 on the
        # rightmost coordinate cells, while global calibration selects .7.
        scores = np.array([[0., 2.], [1., 0.], [2., 4.]])
        calibration = methods.EnvelopeCalibration(scores, .5, scalar_keys=[.2, .7, .4])
        np.testing.assert_array_equal(calibration.mean, scores.mean(axis=0))
        np.testing.assert_array_equal(calibration.scale, scores.std(axis=0))
        for test_key, second_closed in ((.5, True), (.8, False)):
            expected, cells = calibration.predict(search='exhaustive', return_cells=True,
                                                  test_key=test_key)
            self.assertEqual(expected.q_gwc_key, .7)
            self.assertEqual(cells[0][-1]['q_key'], .2)
            self.assertEqual(cells[1][-1]['q_key'], .7)
            np.testing.assert_array_equal(expected.upper_closed, [False, second_closed])
            self.assertEqual(expected.contains([0., expected.upper[1]]), second_closed)
            for search in ('backward', 'rank'):
                actual = calibration.predict(search=search, test_key=test_key)
                np.testing.assert_array_equal(actual.upper, expected.upper)
                np.testing.assert_array_equal(actual.upper_closed, expected.upper_closed)

    def test_seed_reproducibility_and_boundaries(self):
        from utility.rectangle import Rectangle
        a, b = Rectangle([1.], upper_closed=False), Rectangle([2.], [1.])
        self.assertFalse(a.contains([1.]))
        self.assertFalse(a.intersects(b))
        self.assertTrue(Rectangle([0.], upper_closed=False).empty)
        scores = np.array([[0., 2.], [1., 0.], [2., 4.]])
        for predict in (methods.unscaled_prediction, methods.empirical_copula_prediction,
                        methods.gwc_prediction, methods.envelope_prediction):
            first, second = predict(scores, .5), predict(scores, .5)
            np.testing.assert_array_equal(first.info(), second.info())
            np.testing.assert_array_equal(first.upper_closed, second.upper_closed)

    def test_signed_geometry_and_regions_definition(self):
        # signed_envelope.tex eq:psiclosed/eq:link + the surface-cell definition.
        np.testing.assert_allclose(interval_sup([3.], 0., 2., 3, -2., 0.), [np.sqrt(2.5)])
        np.testing.assert_array_equal(score_at(3., 0., 2., 3, [-np.inf, np.inf]), [.5, -.5])
        self.assertTrue(np.isposinf(link(1.5, 0., 2., 3)))
        self.assertTrue(np.isneginf(link(-1.5, 0., 2., 3)))
        for shift in (0., -2.):
            scores = (np.array([0., 1., 2.]) + shift)[:, None]
            gwc = methods.gwc_prediction(scores, .5, shift)
            self.assertAlmostEqual(gwc.q_gwc, np.sqrt(3. / 44.))
            np.testing.assert_allclose(gwc.upper, [1. + shift + 1. / np.sqrt(12.)])
            for search in ("backward", "exhaustive", "rank"):
                env = methods.envelope_prediction(scores, .5, shift, search=search,
                                                 scalar_keys=[.1, .2, .3], test_key=0.)
                np.testing.assert_array_equal(env.upper, [1. + shift])
                np.testing.assert_array_equal(env.contain_points([[shift], [1. + shift],
                                                                 [1.01 + shift]]), [True, True, False])

    def test_preserved_edge_behavior(self):
        self.assertTrue(np.isposinf(methods.unscaled_prediction(np.ones((1, 2))).upper).all())
        self.assertTrue(np.isposinf(methods.empirical_copula_prediction(np.empty((0, 2))).upper).all())
        self.assertTrue(np.isposinf(methods.data_spliting_CHR_prediction(np.ones((2, 2)), .1).upper).all())
        with np.errstate(divide="ignore", invalid="ignore"), self.assertRaises(ValueError):
            methods.data_spliting_CHR_prediction(np.zeros((6, 2)), .5)
        result = methods.cqhr_adjustments(np.ones((3, 2)), np.zeros((3, 2)),
                                                    np.zeros((1, 2)), .5)
        np.testing.assert_array_equal(result.adjustments, [[1., 1.]])
        for predict in (methods.gwc_prediction, methods.envelope_prediction):
            self.assertEqual(predict(np.ones((3, 2)), .5).fallback, "zero_calibration_scale")
            self.assertEqual(predict(np.ones((1, 2)), .1).fallback, "infinite_rank")
            empty = predict(np.array([[0.], [1.], [2.]]), .5, 10.)
            self.assertTrue(empty.empty)
            self.assertEqual(empty.volume(), 0.)
            self.assertFalse(empty.contains([10.]))

    def test_input_errors(self):
        for predict in (methods.unscaled_prediction, methods.empirical_copula_prediction):
            with self.assertRaises(ValueError):
                predict(np.ones(3))
        with self.assertRaises(ValueError):
            methods.unscaled_prediction(np.empty((0, 2)))
        for predict in (methods.gwc_prediction, methods.envelope_prediction):
            for scores in (np.empty((0, 2)), np.ones(3), [[np.nan]], [[np.inf]]):
                with self.assertRaises(ValueError):
                    predict(scores)
            with self.assertRaises(ValueError):
                predict([[0.], [1.], [2.]], .5, np.inf)
        scores, lengths = np.ones((3, 2)), np.ones((3, 2))
        for kwargs in ({"reference_dim": 2}, {"min_length": 0}):
            with self.assertRaises(ValueError):
                methods.cqhr_adjustments(scores, lengths, [[1., 1.]], **kwargs)
        for test_lengths in ([[1.]], [[-1., 1.]]):
            with self.assertRaises(ValueError):
                methods.cqhr_adjustments(scores, lengths, test_lengths)

    def test_additional_manuscript_examples(self):
        # supplementary.tex alg:bonferroni, alg:std-plug-in, alg:std-ds;
        # body.tex eq:residual-transformation and eq:set-valued-map.
        scores = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.]])
        np.testing.assert_array_equal(methods.bonferroni_prediction(scores, .8).upper,
                                      [3., 3.])  # alpha/d=.4, marginal rank 3
        proportional = np.array([[1., 2.], [2., 4.], [3., 6.]])
        np.testing.assert_array_equal(methods.naive_prediction(proportional, .5).upper,
                                      [2., 4.])  # mean [2,4], sample std [1,2], q=0
        oracle_scores = np.array([[1., 3.], [3., 1.], [5., 5.]])
        np.testing.assert_array_equal(methods.data_splitting_oracle_prediction(
            oracle_scores, [1., 1.], [2., 2.], .5).upper, [3., 3.])
        # Split seed42: fit rows [5,4,7], calibrate [1,2,6,3]. Rank 4
        # selects row (6,12); this is the population-oracle formula with fitted scales.
        split_scores = np.arange(1., 8.)[:, None] * [1., 2.]
        np.testing.assert_allclose(methods.data_splitting_standardized_prediction(
            split_scores, .3).upper, [6., 12.], rtol=0, atol=2e-15)

    def test_additional_legacy_formulas(self):
        # These legacy modules contain definitions/imports only. Compare actual
        # jittered results approximately, then their formulas with noise disabled
        # exactly; the latter is not a claim of unmodified legacy output parity.
        from utility import unscaled as old_unscaled, data_splitting as old_split
        scores = np.arange(1., 8.)[:, None] * [1., 2.]
        cases = [
            (methods.bonferroni_prediction, old_unscaled, 'bonferroni_prediction', (), .8),
            (methods.naive_prediction, old_split, 'naive_prediction', (), .5),
            (methods.data_splitting_oracle_prediction, old_split,
             'data_splitting_oracle_prediction', (np.array([1., 2.]), np.array([2., 3.])), .5),
            (methods.data_splitting_standardized_prediction, old_split,
             'data_splitting_standardized_prediction', (), .3),
        ]
        for new, module, name, args, alpha in cases:
            with self.subTest(method=name):
                actual = new(scores, *args, alpha=alpha)
                expected = getattr(module, name)(scores, *args, alpha=alpha)
                np.testing.assert_allclose(actual.upper, expected.upper, rtol=0, atol=1e-9)
                with patch.object(module, 'add_jitter', lambda values, **kwargs: np.asarray(values)):
                    no_noise = getattr(module, name)(scores, *args, alpha=alpha)
                np.testing.assert_array_equal(actual.upper, no_noise.upper)
                self.assertEqual(actual.upper.shape, (2,))
                self.assertEqual(actual.contain_points(scores).shape, (7,))

    def test_additional_randomized_boundaries(self):
        coordinate_keys = np.array([[.1, .2], [.5, .6], [.3, .4]])
        for test_keys, accepted in (([.4, .5], True), ([.7, .5], False)):
            region = methods.bonferroni_prediction(np.ones((3, 2)), .75,
                coordinate_keys=coordinate_keys, test_keys=test_keys)
            np.testing.assert_array_equal(region.upper, [1., 1.])
            self.assertEqual(region.contains([1., 1.]), accepted)
        scores = np.array([[1., 3.], [3., 1.], [3., 3.]])
        split_scores = np.array([[1., 3.], [3., 1.], [1., 1.],
                                 [2., 2.], [3., 3.], [3., 3.]])
        for test_key, accepted in ((.2, True), (.8, False)):
            for predict, args, data, keys in (
                (methods.naive_prediction, (), scores, [.1, .7, .4]),
                (methods.data_splitting_oracle_prediction, (0., 1.), scores, [.1, .7, .4]),
                (methods.data_splitting_standardized_prediction, (), split_scores,
                 [.1, .7, .2, .3, .6, .4]),
            ):
                region = predict(data, *args, alpha=.5, scalar_keys=keys, test_key=test_key)
                np.testing.assert_array_equal(region.upper, [3., 3.])
                self.assertEqual(region.contains([3., 3.]), accepted)
                self.assertTrue(region.contains([2., 2.]))

    def test_additional_exact_allocation_and_edges(self):
        # Converting alpha to an exact fraction BEFORE division matters: the
        # float .3/3 is below .1 and would spuriously produce an infinite rank.
        scores = np.arange(1., 10.)[:, None] * [1., 2., 3.]
        np.testing.assert_array_equal(methods.bonferroni_prediction(scores, .3).upper,
                                      [9., 18., 27.])
        small = np.array([[1., 2.], [2., 4.]])
        for predict, args in ((methods.bonferroni_prediction, ()),
                              (methods.naive_prediction, ()),
                              (methods.data_splitting_oracle_prediction, (0., 1.))):
            self.assertTrue(np.isposinf(predict(small, *args, alpha=.1).upper).all())
        split = np.arange(1., 7.)[:, None] * [1., 2.]
        self.assertTrue(np.isposinf(methods.data_splitting_standardized_prediction(split, .1).upper).all())
        # Scalars advertised by the old oracle docstring now broadcast to (d,).
        scalar = methods.data_splitting_oracle_prediction(small, 0., 1., .5)
        vector = methods.data_splitting_oracle_prediction(small, [0., 0.], [1., 1.], .5)
        np.testing.assert_array_equal(scalar.info(), vector.info())
        for predict, args in ((methods.bonferroni_prediction, ()),
                              (methods.naive_prediction, ()),
                              (methods.data_splitting_oracle_prediction, (0., 1.)),
                              (methods.data_splitting_standardized_prediction, ())):
            for invalid in (np.empty((0, 2)), np.ones(3), [[np.nan]], [[np.inf]]):
                with self.subTest(method=predict.__name__, invalid=invalid):
                    with self.assertRaises(ValueError):
                        predict(invalid, *args)
        for predict in (methods.naive_prediction, methods.data_splitting_standardized_prediction):
            with self.assertRaises(ValueError):
                predict(np.ones((6, 2)), .5)
        for scale in (0., -1., np.inf):
            with self.assertRaises(ValueError):
                methods.data_splitting_oracle_prediction(small, 0., scale, .5)

    def test_signed_rectangle_methods(self):
        # In one dimension these rank-4 examples select -4. The preserved
        # seed-42 split at alpha=.3 selects -2 (rank 4 in its second half).
        from utility.rectangle import Rectangle
        scores = np.arange(-7., 0.)[:, None]
        cases = (
            (methods.unscaled_prediction, (), dict(alpha=.5, test_key=0.), -4.),
            (methods.empirical_copula_prediction, (), dict(alpha=.5, test_keys=[0.]), -4.),
            (methods.bonferroni_prediction, (), dict(alpha=.5, test_keys=[0.]), -4.),
            (methods.naive_prediction, (), dict(alpha=.5, test_key=0.), -4.),
            (methods.data_splitting_oracle_prediction, (0., 1.), dict(alpha=.5, test_key=0.), -4.),
            (methods.data_splitting_standardized_prediction, (), dict(alpha=.3, test_key=0.), -2.),
        )
        for predict, args, kwargs, expected in cases:
            with self.subTest(method=predict.__name__):
                # Some calibration scores lie below this test-specific lower
                # bound. They must still participate unchanged in calibration.
                region = predict(scores, *args, lower=expected-1, **kwargs)
                self.assertIsInstance(region, Rectangle)
                np.testing.assert_allclose(region.upper, [expected], rtol=0, atol=2e-15)
                np.testing.assert_array_equal(region.lower, [expected-1])
                np.testing.assert_array_equal(region.contain_points(
                    [[expected-2], [expected-1], [expected-.5], [expected+1]]),
                    [False, True, True, False])
                self.assertAlmostEqual(region.volume(), 1.)
                translated = predict(scores+10, *args, **kwargs)
                np.testing.assert_allclose(translated.upper, region.upper+10, rtol=0, atol=2e-15)
                np.testing.assert_array_equal(translated.upper_closed, region.upper_closed)
                for lower_kwargs in ({}, dict(lower=expected+1)):
                    empty = predict(scores, *args, **kwargs, **lower_kwargs)
                    np.testing.assert_array_equal(empty.upper, region.upper)
                    self.assertTrue(empty.empty)
                    self.assertFalse(empty.contains([expected]))
                    np.testing.assert_array_equal(empty.length_along_dimensions(), [0.])
                    self.assertEqual(empty.volume(), 0.)
                unbounded = predict(scores, *args, lower=-np.inf, **kwargs)
                self.assertTrue(unbounded.contains([-100.]))
                self.assertTrue(np.isposinf(unbounded.volume()))
                for lower in (np.nan, np.inf, [0., 0.], [[0.]]):
                    with self.assertRaises(ValueError):
                        predict(scores, *args, lower=lower, **kwargs)

    def test_rectangle_signed_empty_and_unbounded_geometry(self):
        from utility.rectangle import Rectangle
        region = Rectangle([-2., -1.], lower=-3.)
        np.testing.assert_array_equal(region.lower, [-3., -3.])
        self.assertTrue(region.contains([-2.5, -2.]))
        with self.assertRaises(ValueError):
            Rectangle([-2.], lower=-1.)  # legacy default still rejects inversion
        empty = Rectangle([-2., np.inf], lower=[-1., -np.inf], allow_empty=True)
        self.assertTrue(empty.empty)
        self.assertFalse(empty.contains([-1., 0.]))
        self.assertFalse(empty.intersects(region))
        np.testing.assert_array_equal(empty.length_along_dimensions(), [0., 0.])
        self.assertEqual(empty.volume(), 0.)
        self.assertFalse(Rectangle([-2.], lower=-2.).empty)
        self.assertTrue(Rectangle([-2.], lower=-2., upper_closed=False).empty)
        # A closed singleton side makes volume zero, even with another
        # unbounded side; 0 * infinity must not become NaN.
        singleton = Rectangle([0., np.inf], lower=[0., -np.inf])
        self.assertTrue(singleton.contains([0., 1.]))
        self.assertEqual(singleton.volume(), 0.)
        for lower in (np.nan, np.inf, [0., 0.], [[0.]]):
            with self.assertRaises(ValueError):
                Rectangle([1.], lower=lower)


if __name__ == "__main__":
    unittest.main()
