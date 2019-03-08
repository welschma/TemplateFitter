import unittest

import numpy as np

from templatefitter.histograms import Hist2d
from templatefitter.histograms import bin_widths, bin_mids


class TestHist2d(unittest.TestCase):
    def setUp(self):

        self.iris_sepal_length = np.array(
            [
                5.1,
                4.9,
                4.7,
                4.6,
                5.0,
                5.4,
                4.6,
                5.0,
                4.4,
                4.9,
                5.4,
                4.8,
                4.8,
                4.3,
                5.8,
                5.7,
                5.4,
                5.1,
                5.7,
                5.1,
                5.4,
                5.1,
                4.6,
                5.1,
                4.8,
                5.0,
                5.0,
                5.2,
                5.2,
                4.7,
                4.8,
                5.4,
                5.2,
                5.5,
                4.9,
                5.0,
                5.5,
                4.9,
                4.4,
                5.1,
                5.0,
                4.5,
                4.4,
                5.0,
                5.1,
                4.8,
                5.1,
                4.6,
                5.3,
                5.0,
                7.0,
                6.4,
                6.9,
                5.5,
                6.5,
                5.7,
                6.3,
                4.9,
                6.6,
                5.2,
                5.0,
                5.9,
                6.0,
                6.1,
                5.6,
                6.7,
                5.6,
                5.8,
                6.2,
                5.6,
                5.9,
                6.1,
                6.3,
                6.1,
                6.4,
                6.6,
                6.8,
                6.7,
                6.0,
                5.7,
                5.5,
                5.5,
                5.8,
                6.0,
                5.4,
                6.0,
                6.7,
                6.3,
                5.6,
                5.5,
                5.5,
                6.1,
                5.8,
                5.0,
                5.6,
                5.7,
                5.7,
                6.2,
                5.1,
                5.7,
                6.3,
                5.8,
                7.1,
                6.3,
                6.5,
                7.6,
                4.9,
                7.3,
                6.7,
                7.2,
                6.5,
                6.4,
                6.8,
                5.7,
                5.8,
                6.4,
                6.5,
                7.7,
                7.7,
                6.0,
                6.9,
                5.6,
                7.7,
                6.3,
                6.7,
                7.2,
                6.2,
                6.1,
                6.4,
                7.2,
                7.4,
                7.9,
                6.4,
                6.3,
                6.1,
                7.7,
                6.3,
                6.4,
                6.0,
                6.9,
                6.7,
                6.9,
                5.8,
                6.8,
                6.7,
                6.7,
                6.3,
                6.5,
                6.2,
                5.9,
            ]
        )

        self.iris_sepal_width = np.array(
            [
                3.5,
                3.0,
                3.2,
                3.1,
                3.6,
                3.9,
                3.4,
                3.4,
                2.9,
                3.1,
                3.7,
                3.4,
                3.0,
                3.0,
                4.0,
                4.4,
                3.9,
                3.5,
                3.8,
                3.8,
                3.4,
                3.7,
                3.6,
                3.3,
                3.4,
                3.0,
                3.4,
                3.5,
                3.4,
                3.2,
                3.1,
                3.4,
                4.1,
                4.2,
                3.1,
                3.2,
                3.5,
                3.1,
                3.0,
                3.4,
                3.5,
                2.3,
                3.2,
                3.5,
                3.8,
                3.0,
                3.8,
                3.2,
                3.7,
                3.3,
                3.2,
                3.2,
                3.1,
                2.3,
                2.8,
                2.8,
                3.3,
                2.4,
                2.9,
                2.7,
                2.0,
                3.0,
                2.2,
                2.9,
                2.9,
                3.1,
                3.0,
                2.7,
                2.2,
                2.5,
                3.2,
                2.8,
                2.5,
                2.8,
                2.9,
                3.0,
                2.8,
                3.0,
                2.9,
                2.6,
                2.4,
                2.4,
                2.7,
                2.7,
                3.0,
                3.4,
                3.1,
                2.3,
                3.0,
                2.5,
                2.6,
                3.0,
                2.6,
                2.3,
                2.7,
                3.0,
                2.9,
                2.9,
                2.5,
                2.8,
                3.3,
                2.7,
                3.0,
                2.9,
                3.0,
                3.0,
                2.5,
                2.9,
                2.5,
                3.6,
                3.2,
                2.7,
                3.0,
                2.5,
                2.8,
                3.2,
                3.0,
                3.8,
                2.6,
                2.2,
                3.2,
                2.8,
                2.8,
                2.7,
                3.3,
                3.2,
                2.8,
                3.0,
                2.8,
                3.0,
                2.8,
                3.8,
                2.8,
                2.8,
                2.6,
                3.0,
                3.4,
                3.1,
                3.0,
                3.1,
                3.1,
                3.1,
                2.7,
                3.2,
                3.3,
                3.0,
                2.5,
                3.0,
                3.4,
                3.0,
            ]
        )

    def test_init_with_int_bins(self):
        bins = (3, 3)
        range = ((0, 10), (0, 10))
        bc, be_x, be_y = np.histogram2d(
            self.iris_sepal_length, self.iris_sepal_width, bins=bins, range=range
        )

        hiris = Hist2d(
            bins=bins, data=(self.iris_sepal_length, self.iris_sepal_width), range=range
        )

        np.testing.assert_array_equal(hiris.bin_counts, bc)
        np.testing.assert_array_equal(hiris.x_edges, be_x)
        np.testing.assert_array_equal(hiris.y_edges, be_y)

        for obs, exp in zip(hiris.bin_edges, [be_x, be_y]):
            np.testing.assert_array_equal(obs, exp)

        for obs, exp in zip(hiris.bin_mids, [be_x, be_y]):
            mids = bin_mids(exp)
            np.testing.assert_array_equal(obs, mids)

        for obs, exp in zip(hiris.bin_widths, [be_x, be_y]):
            widths = bin_widths(exp)
            np.testing.assert_array_equal(obs, widths)

    def test_init_with_array_bins(self):
        bins = (np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        bc, be_x, be_y = np.histogram2d(
            self.iris_sepal_length, self.iris_sepal_width, bins=bins
        )

        hiris = Hist2d(
            bins=bins, data=(self.iris_sepal_length, self.iris_sepal_width), range=range
        )

        np.testing.assert_array_equal(hiris.bin_counts, bc)
        np.testing.assert_array_equal(hiris.x_edges, be_x)
        np.testing.assert_array_equal(hiris.y_edges, be_y)

        self.assertEqual(hiris.shape, bc.shape)

        for obs, exp in zip(hiris.bin_edges, [be_x, be_y]):
            np.testing.assert_array_equal(obs, exp)

        for obs, exp in zip(hiris.bin_mids, [be_x, be_y]):
            mids = bin_mids(exp)
            np.testing.assert_array_equal(obs, mids)

        for obs, exp in zip(hiris.bin_widths, [be_x, be_y]):
            widths = bin_widths(exp)
            np.testing.assert_array_equal(obs, widths)

    def test_from_binned_data(self):
        bins = (np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        bc, be_x, be_y = np.histogram2d(
            self.iris_sepal_length, self.iris_sepal_width, bins=bins
        )

        hiris = Hist2d.from_binned_data(bc, bins)

        np.testing.assert_array_equal(hiris.bin_counts, bc)
        np.testing.assert_array_equal(hiris.x_edges, be_x)
        np.testing.assert_array_equal(hiris.y_edges, be_y)

        for obs, exp in zip(hiris.bin_edges, [be_x, be_y]):
            np.testing.assert_array_equal(obs, exp)

        for obs, exp in zip(hiris.bin_mids, [be_x, be_y]):
            mids = bin_mids(exp)
            np.testing.assert_array_equal(obs, mids)

        for obs, exp in zip(hiris.bin_widths, [be_x, be_y]):
            widths = bin_widths(exp)
            np.testing.assert_array_equal(obs, widths)


