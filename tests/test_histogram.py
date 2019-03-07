# import unittest
#
# import numpy as np
# import scipy.stats
#
# from templatefitter import Hist1d, Hist2d, Histogram
#
#
# class TestHistogram(unittest.TestCase):
#     def setUp(self):
#         self.data = np.array([1, 1, 2, 5, 7])
#         self.weights = np.array([1.0, 1.0, 2.0, 1.5, 3.0])
#
#         self.hist = Histogram(2, (1.0, 7))
#
#     def test_fill(self):
#         """Tests if Histogram.fill works correctly if applied
#         more than once.
#         """
#         self.hist.fill(self.data, self.weights)
#
#         counts = self.hist.bin_counts
#         entries = self.hist.bin_entries
#         errors = self.hist.bin_errors
#
#         true_counts = scipy.stats.binned_statistic(
#             self.data, self.weights, "sum", self.hist.bin_edges
#         )[0]
#
#         true_entries = scipy.stats.binned_statistic(
#             self.data, self.weights, "count", self.hist.bin_edges
#         )[0]
#
#         true_errors_sq = scipy.stats.binned_statistic(
#             self.data, self.weights ** 2, "sum", self.hist.bin_edges
#         )[0]
#
#         np.testing.assert_equal(counts, true_counts)
#         np.testing.assert_equal(entries, true_entries)
#         np.testing.assert_almost_equal(errors, np.sqrt(true_errors_sq))
#
#         self.hist.fill(self.data, self.weights)
#
#         counts = self.hist.bin_counts
#         entries = self.hist.bin_entries
#         errors = self.hist.bin_errors
#
#         true_counts += scipy.stats.binned_statistic(
#             self.data, self.weights, "sum", self.hist.bin_edges
#         )[0]
#         true_entries += scipy.stats.binned_statistic(
#             self.data, self.weights, "count", self.hist.bin_edges
#         )[0]
#         true_errors_sq += scipy.stats.binned_statistic(
#             self.data, self.weights ** 2, "sum", self.hist.bin_edges
#         )[0]
#
#         np.testing.assert_equal(counts, true_counts)
#         np.testing.assert_equal(entries, true_entries)
#         np.testing.assert_almost_equal(errors, np.sqrt(true_errors_sq))
#
#     def test_scale(self):
#         """Test scale method.
#         """
#         self.hist.fill(self.data, self.weights)
#
#         c = 2
#         self.hist.scale(c)
#
#         counts = self.hist.bin_counts
#         entries = self.hist.bin_entries
#         errors = self.hist.bin_errors
#
#         true_counts = scipy.stats.binned_statistic(
#             self.data, self.weights, "sum", self.hist.bin_edges
#         )[0]
#
#         true_entries = scipy.stats.binned_statistic(
#             self.data, self.weights, "count", self.hist.bin_edges
#         )[0]
#
#         true_errors_sq = scipy.stats.binned_statistic(
#             self.data, self.weights ** 2, "sum", self.hist.bin_edges
#         )[0]
#
#         np.testing.assert_equal(counts, true_counts * c)
#         np.testing.assert_equal(entries, true_entries)
#         np.testing.assert_almost_equal(errors, np.sqrt(true_errors_sq * c ** 2))
#
#     def test_print(self):
#         """Test __str__ method.
#         """
#
#         self.hist.fill(self.data, self.weights)
#         print(self.hist)
#         self.assertTrue(True)
#
#     def test_fill_error(self):
#         """Test if fill raises HistogramError if data and weights
#         have not the same length.
#         """
#         weights = self.weights[:-1]
#         self.assertRaises(HistogramError, self.hist.fill, self.data, weights)
#
#
# class TestHist1d(unittest.TestCase):
#     """Unittests for Hist1d"""
#
#     def setUp(self):
#         self.test_data = np.array(
#             [0.2, 0.3, 0.2, 0.7, 1.2, 1.5, 1.3, 1.3, 2.4, 2.6, 2.2]
#         )
#         self.test_weights = np.array([1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2])
#         self.test_bin_counts = np.array([5.0, 7.0, 5.0])
#         self.test_bin_errors = np.sqrt(np.array([7.0, 13.0, 9.0]))
#
#     def test_init_with_data_weights_and_bin_edges(self):
#         """Test Hist1d for initialization with data, weights and bin edges"""
#         bin_edges = np.array([0, 1, 2, 3])
#
#         hist = Hist1d(
#             bins=bin_edges,
#             limits=(0.0, 3.0),
#             data=self.test_data,
#             weights=self.test_weights,
#         )
#
#         np.testing.assert_array_equal(hist.bin_counts, self.test_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, self.test_bin_errors)
#         np.testing.assert_array_equal(hist.bin_edges, bin_edges)
#         np.testing.assert_equal(hist.num_bins, len(bin_edges) - 1)
#
#         hist.fill(np.array([0.4, 2.5]), np.array([3, 1]))
#
#         new_bin_counts = self.test_bin_counts + np.array([3, 0, 1])
#         new_bin_errors = np.sqrt(self.test_bin_errors ** 2 + np.array([3 ** 2, 0, 1]))
#
#         np.testing.assert_array_equal(hist.bin_counts, new_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, new_bin_errors)
#
#     def test_init_with_data_and_weights(self):
#         """Test Hist1d for initialization with data and bin edges"""
#
#         hist = Hist1d(
#             bins=3, limits=(0.0, 3.0), data=self.test_data, weights=self.test_weights
#         )
#
#         np.testing.assert_array_equal(hist.bin_counts, self.test_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, self.test_bin_errors)
#         np.testing.assert_array_equal(hist.bin_edges, np.array([0.0, 1.0, 2.0, 3.0]))
#         np.testing.assert_equal(hist.num_bins, 3)
#
#         hist.fill(np.array([0.4, 2.5]), np.array([3, 1]))
#
#         new_bin_counts = self.test_bin_counts + np.array([3, 0, 1])
#         new_bin_errors = np.sqrt(self.test_bin_errors ** 2 + np.array([3 ** 2, 0, 1]))
#
#         np.testing.assert_array_equal(hist.bin_counts, new_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, new_bin_errors)
#
#     def test_init_with_data(self):
#         """Test Hist1d for initialization with data and bin numbers"""
#
#         hist = Hist1d(bins=3, limits=(0.0, 3.0), data=self.test_data)
#         test_bin_counts = np.array([4, 4, 3])
#         test_bin_errors = np.sqrt(test_bin_counts)
#
#         np.testing.assert_array_equal(hist.bin_counts, test_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, test_bin_errors)
#         np.testing.assert_array_equal(hist.bin_edges, np.array([0.0, 1.0, 2.0, 3.0]))
#         np.testing.assert_equal(hist.num_bins, 3)
#
#         hist.fill(np.array([0.4, 2.5]))
#
#         new_bin_counts = test_bin_counts + np.array([1, 0, 1])
#         new_bin_errors = np.sqrt(test_bin_errors ** 2 + np.array([1, 0, 1]))
#         np.testing.assert_array_equal(hist.bin_counts, new_bin_counts)
#         np.testing.assert_array_almost_equal(hist.bin_errors, new_bin_errors)
#
#
# class TestHist2d(unittest.TestCase):
#     def setUp(self):
#         self.test_data = np.array(
#             [[0.3, 3.3], [0.5, 3.3], [0.5, 4.7], [1.3, 3.2], [1.5, 4.6]]
#         )
#         self.test_weights = np.array([3, 1, 2, 4, 1])
#         self.test_bin_counts = np.array([[4, 2], [4, 1]])
#         self.test_bin_errors = np.sqrt(np.array([[10, 4], [16, 1]]))
#
#     def test_init_with_data_weights_and_bin_edges(self):
#         bin_edges = [np.array([0, 1, 2]), np.array([3, 4, 5])]
#
#         hist = Hist2d(bins=bin_edges, data=self.test_data, weights=self.test_weights)
#
#         np.testing.assert_array_equal(hist.bin_counts, self.test_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, self.test_bin_errors)
#         np.testing.assert_array_equal(hist.bin_edges, bin_edges)
#         self.assertListEqual(hist.num_bins, [len(edges) - 1 for edges in bin_edges])
#
#         np.testing.assert_array_equal(hist.x_projection()[0], np.array([6, 5]))
#         np.testing.assert_array_equal(hist.y_projection()[0], np.array([8, 3]))
#         np.testing.assert_array_equal(
#             hist.x_projection()[1], np.sqrt(np.array([14, 17]))
#         )
#         np.testing.assert_array_equal(
#             hist.y_projection()[1], np.sqrt(np.array([26, 5]))
#         )
#
#         hist.fill(np.array([[0.4, 4.5]]), np.array([3]))
#
#         new_bin_counts = self.test_bin_counts + np.array([[0, 3], [0, 0]])
#         new_bin_errors = np.sqrt(
#             self.test_bin_errors ** 2 + np.array([[0, 3], [0, 0]]) ** 2
#         )
#
#         np.testing.assert_array_equal(hist.bin_counts, new_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, new_bin_errors)
#
#         np.testing.assert_array_equal(hist.x_projection()[0], np.array([9, 5]))
#         np.testing.assert_array_equal(hist.y_projection()[0], np.array([8, 6]))
#         np.testing.assert_array_equal(
#             hist.x_projection()[1], np.sqrt(np.array([23, 17]))
#         )
#         np.testing.assert_array_equal(
#             hist.y_projection()[1], np.sqrt(np.array([26, 14]))
#         )
#
#     def test_init_with_data_and_weights(self):
#         bin_edges = [np.array([0, 1, 2]), np.array([3, 4, 5])]
#
#         hist = Hist2d(
#             bins=[2, 2],
#             limits=[(0.0, 2.0), (3.0, 5.0)],
#             data=self.test_data,
#             weights=self.test_weights,
#         )
#
#         np.testing.assert_array_equal(hist.bin_edges, bin_edges)
#         np.testing.assert_array_equal(hist.bin_counts, self.test_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, self.test_bin_errors)
#         self.assertListEqual(hist.num_bins, [len(edges) - 1 for edges in bin_edges])
#
#         hist.fill(np.array([[0.4, 4.5]]), np.array([3]))
#
#         new_bin_counts = self.test_bin_counts + np.array([[0, 3], [0, 0]])
#         new_bin_errors = np.sqrt(
#             self.test_bin_errors ** 2 + np.array([[0, 3], [0, 0]]) ** 2
#         )
#
#         np.testing.assert_array_equal(hist.bin_counts, new_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, new_bin_errors)
#
#     def test_init_with_data(self):
#         bin_edges = [np.array([0, 1, 2]), np.array([3, 4, 5])]
#
#         hist = Hist2d(bins=[2, 2], limits=[(0.0, 2.0), (3.0, 5.0)], data=self.test_data)
#
#         bin_counts = np.array([[2, 1], [1, 1]])
#         bin_errors = np.sqrt(np.array([[2, 1], [1, 1]]))
#
#         np.testing.assert_array_equal(hist.bin_edges, bin_edges)
#         np.testing.assert_array_equal(hist.bin_counts, bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, bin_errors)
#         self.assertListEqual(hist.num_bins, [len(edges) - 1 for edges in bin_edges])
#
#         hist.fill(np.array([[0.4, 4.5]]))
#
#         new_bin_counts = bin_counts + np.array([[0, 1], [0, 0]])
#         new_bin_errors = np.sqrt(bin_errors ** 2 + np.array([[0, 1], [0, 0]]) ** 2)
#
#         np.testing.assert_array_equal(hist.bin_counts, new_bin_counts)
#         np.testing.assert_array_equal(hist.bin_errors, new_bin_errors)
