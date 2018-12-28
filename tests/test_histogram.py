import unittest

import numpy as np
import scipy.stats

from templatefitter import Histogram
from templatefitter.histogram import HistogramError

class TestHistogram(unittest.TestCase):
    
    def setUp(self):
        self.data = np.array([1, 1, 2, 5, 7])
        self.weights = np.array([1.0, 1.0, 2.0, 1.5, 3.0])

        self.hist = Histogram(2, (1., 7))

    def test_fill(self):
        """Tests if Histogram.fill works correctly if applied
        more than once.
        """
        self.hist.fill(self.data, self.weights)

        counts = self.hist.bin_counts
        entries = self.hist.bin_entries
        errors = self.hist.bin_errors

        true_counts = scipy.stats.binned_statistic(
            self.data, self.weights, 'sum', self.hist.bin_edges)[0]

        true_entries = scipy.stats.binned_statistic(
            self.data, self.weights, 'count', self.hist.bin_edges)[0]

        true_errors_sq = scipy.stats.binned_statistic(
            self.data, self.weights**2, 'sum', self.hist.bin_edges)[0]

        np.testing.assert_equal(counts, true_counts)
        np.testing.assert_equal(entries, true_entries)
        np.testing.assert_almost_equal(errors, np.sqrt(true_errors_sq))

        self.hist.fill(self.data, self.weights)

        counts = self.hist.bin_counts
        entries = self.hist.bin_entries
        errors = self.hist.bin_errors

        true_counts += scipy.stats.binned_statistic(
            self.data, self.weights, 'sum', self.hist.bin_edges)[0]
        true_entries += scipy.stats.binned_statistic(
            self.data, self.weights, 'count', self.hist.bin_edges)[0]
        true_errors_sq += scipy.stats.binned_statistic(
            self.data, self.weights**2, 'sum', self.hist.bin_edges)[0]
        
        np.testing.assert_equal(counts, true_counts)
        np.testing.assert_equal(entries, true_entries)
        np.testing.assert_almost_equal(errors, np.sqrt(true_errors_sq))

    def test_scale(self):
        """Test scale method.
        """
        self.hist.fill(self.data, self.weights)

        c = 2
        self.hist.scale(c)

        counts = self.hist.bin_counts
        entries = self.hist.bin_entries
        errors = self.hist.bin_errors

        true_counts = scipy.stats.binned_statistic(
            self.data, self.weights, 'sum', self.hist.bin_edges)[0]

        true_entries = scipy.stats.binned_statistic(
            self.data, self.weights, 'count', self.hist.bin_edges)[0]

        true_errors_sq = scipy.stats.binned_statistic(
            self.data, self.weights**2, 'sum', self.hist.bin_edges)[0]

        np.testing.assert_equal(counts, true_counts*c)
        np.testing.assert_equal(entries, true_entries)
        np.testing.assert_almost_equal(errors, np.sqrt(true_errors_sq*c**2))

    def test_print(self):
        """Test __str__ method.
        """

        self.hist.fill(self.data, self.weights)
        print(self.hist)
        self.assertTrue(True)

    def test_fill_error(self):
        """Test if fill raises HistogramError if data and weights
        have not the same length.
        """
        weights = self.weights[:-1]
        self.assertRaises(HistogramError, self.hist.fill, self.data, weights)

