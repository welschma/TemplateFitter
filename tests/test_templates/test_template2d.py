import unittest

import numpy as np

from templatefitter.templates import Template2d
from templatefitter.histograms import Hist2d

from templatefitter.utility import get_systematic_cov_mat


class TestHist2d(unittest.TestCase):

    def setUp(self):
        self.x = np.array(
            [5.1, 4.9, 4.7, 4.6, 5., 5.4, 4.6, 5., 4.4, 4.9, 5.4, 4.8, 4.8,
             4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.,
             5., 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5., 5.5, 4.9, 4.4,
             5.1, 5., 4.5, 4.4, 5., 5.1, 4.8, 5.1, 4.6, 5.3, 5., 7., 6.4,
             6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5., 5.9, 6., 6.1, 5.6,
             6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7,
             6., 5.7, 5.5, 5.5, 5.8, 6., 5.4, 6., 6.7, 6.3, 5.6, 5.5, 5.5,
             6.1, 5.8, 5., 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3,
             6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5,
             7.7, 7.7, 6., 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
             7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6., 6.9, 6.7, 6.9, 5.8,
             6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9]
        )
        self.y = np.array(
            [3.5, 3., 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.,
             3., 4., 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.,
             3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.1, 3.,
             3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3., 3.8, 3.2, 3.7, 3.3, 3.2, 3.2,
             3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2., 3., 2.2, 2.9, 2.9,
             3.1, 3., 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3., 2.8, 3.,
             2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3., 3.4, 3.1, 2.3, 3., 2.5, 2.6,
             3., 2.6, 2.3, 2.7, 3., 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3., 2.9,
             3., 3., 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3., 2.5, 2.8, 3.2, 3.,
             3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3., 2.8, 3.,
             2.8, 3.8, 2.8, 2.8, 2.6, 3., 3.4, 3.1, 3., 3.1, 3.1, 3.1, 2.7,
             3.2, 3.3, 3., 2.5, 3., 3.4, 3.]
        )

        self.bins = (4, 4)
        self.num_bins = self.bins[0] * self.bins[1]
        self.range = ((4, 6), (2, 4))
        self.hist = Hist2d(bins=self.bins, range=self.range, data=(self.x, self.y))
        self.template = Template2d(
            "test", ("test_x", "test_y"), self.hist
        )

    def test_values(self):
        np.testing.assert_array_almost_equal(self.template.values, self.hist.bin_counts)
        self.assertEqual(self.template.values.shape, self.bins)

    def test_errors(self):
        expected = self.hist.bin_counts * np.divide(
            self.hist.bin_errors, self.hist.bin_counts,
            out=np.full(self.bins, 1e-7), where=self.hist.bin_counts != 0.
        )
        self.assertEqual(self.template.errors.shape, self.bins)
        np.testing.assert_almost_equal(self.template.errors, expected)

    def test_params(self):
        expected = np.zeros(self.num_bins + 1)
        expected[0] = np.sum(self.hist.bin_counts)
        np.testing.assert_array_equal(self.template.params, expected)
        np.testing.assert_array_equal(self.template.yield_param, expected[0])
        np.testing.assert_array_equal(self.template.nui_params, expected[1:])

    def test_cov_mat(self):
        errors_sq = self.hist.bin_errors_sq.flatten()
        errors_sq[errors_sq == 0] = 1e-14
        expected = np.diag(errors_sq)

        np.testing.assert_array_equal(self.template._cov, expected)
        self.assertEqual(self.template._cov.shape, (self.num_bins, self.num_bins))

    def test_add_variation(self):
        up = np.full(self.x.shape, 1.1)
        down = np.full(self.x.shape, 0.85)
        hup, _, _ = np.histogram2d(
            self.x, self.y, bins=self.bins, range=self.range, weights=up
        )
        hdown, _, _ = np.histogram2d(
            self.x, self.y, bins=self.bins, range=self.range, weights=down
        )

        errors_sq = self.hist.bin_errors_sq.flatten()
        errors_sq[errors_sq == 0] = 1e-14
        stat_cov = np.diag(errors_sq)
        sys_cov = get_systematic_cov_mat(
            self.template._hist.bin_counts.flatten(), hup.flatten(), hdown.flatten())

        self.template.add_variation((self.x, self.y), up, down)

        expected = stat_cov + sys_cov
        np.testing.assert_array_almost_equal(self.template._cov, expected)

        relative_errors = np.divide(
            np.sqrt(np.diag(expected)),
            self.hist.bin_counts.flatten(),
            out=np.full(self.num_bins, 1e-7),
            where=self.hist.bin_counts.flatten() != 0,
        )
        np.testing.assert_almost_equal(self.template._relative_errors, relative_errors)
