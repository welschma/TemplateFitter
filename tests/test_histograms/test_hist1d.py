import unittest

import numpy as np

from templatefitter.histograms import Hist1d

class TestHist1d(unittest.TestCase):

    def setUp(self):

        self.iris_sepal_length = np.array(
            [5.1, 4.9, 4.7, 4.6, 5. , 5.4, 4.6, 5. , 4.4, 4.9, 5.4, 4.8, 4.8,
             4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5. ,
             5. , 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5. , 5.5, 4.9, 4.4,
             5.1, 5. , 4.5, 4.4, 5. , 5.1, 4.8, 5.1, 4.6, 5.3, 5. , 7. , 6.4,
             6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5. , 5.9, 6. , 6.1, 5.6,
             6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7,
             6. , 5.7, 5.5, 5.5, 5.8, 6. , 5.4, 6. , 6.7, 6.3, 5.6, 5.5, 5.5,
             6.1, 5.8, 5. , 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3,
             6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5,
             7.7, 7.7, 6. , 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
             7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6. , 6.9, 6.7, 6.9, 5.8,
             6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9]
        )

    def test_hist_from_int_bin(self):
        nbins = 10
        iris_bc, iris_be = np.histogram(self.iris_sepal_length, bins=nbins)
        iris_hist = Hist1d(self.iris_sepal_length,
                           bins=nbins)

        np.testing.assert_array_equal(iris_hist.bin_counts, iris_bc)
        np.testing.assert_array_equal(iris_hist.bin_edges, iris_be)

    def test_hist_from_array_bins(self):
        bins = np.linspace(0, 10, 11)
        iris_bc, iris_be = np.histogram(self.iris_sepal_length, bins=bins)
        iris_hist = Hist1d(self.iris_sepal_length,
                           bins=bins)

        np.testing.assert_array_equal(iris_hist.bin_counts, iris_bc)
        np.testing.assert_array_equal(iris_hist.bin_edges, iris_be)

    def test_hist_with_range_option(self):
        nbins = 10
        range=(4, 6)
        iris_bc, iris_be = np.histogram(
            self.iris_sepal_length, bins=nbins, range=range
        )
        iris_hist = Hist1d(
            self.iris_sepal_length, bins=nbins, range=range
        )

        np.testing.assert_array_equal(iris_hist.bin_counts, iris_bc)
        np.testing.assert_array_equal(iris_hist.bin_edges, iris_be)
