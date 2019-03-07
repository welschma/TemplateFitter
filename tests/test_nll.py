# import unittest
# from typing import Tuple
#
# import numpy as np
# import pandas as pd
# import scipy.stats
#
# import templatefitter as tf
#
# class TestStackedTemplateNegLogLikelihood(unittest.TestCase):
#
#     def setUp(self):
#         self.bkg = pd.DataFrame({
#             "x": np.array([0.5, 0.2, 0.4,
#                            1.2, 1.3, 1.4,
#                            2.4, 2.7, 2.8]),
#             "weight": np.array([2, 2, 1,
#                                 1, 1, 2,
#                                 2, 2, 2])
#         })
#
#         self.sig = pd.DataFrame({
#             "x": np.array([1.2, 1.3, 1.4,
#                            2.4, 2.7, 2.8]),
#             "weight": np.array([2, 2, 1,
#                                 2, 2, 2])
#         })
#
#         # bin0 -> 6 data points, bin 1 -> 9 data points, bin 2 -> 13 data points
#         self.data = np.array([0.5, 0.2, 0.4, 0.5, 0.2, 0.4,
#                                 1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.2, 1.3, 1.4,
#                                 2.4, 2.7, 2.8, 2.4, 2.7, 2.8,
#                                 2.4, 2.7, 2.8, 2.4, 2.7, 2.8,
#                                 2.4])
#
#         self.limits = (0., 3.)
#         self.num_bins = 3
#         self.var = "x"
#         self.num_templates = 2
#
#         self.hdata = tf.histogram.Hist1d(self.num_bins, self.limits, self.data)
#
#         self.sig_hist = tf.histogram.Hist1d(self.num_bins, self.limits, self.sig.x, self.sig.weight)
#         self.bkg_hist = tf.histogram.Hist1d(self.num_bins, self.limits, self.bkg.x, self.bkg.weight)
#
#         self.st = tf.templates.StackedTemplate("test", "x", self.num_bins, self.limits)
#         self.sig_temp = tf.templates.Template("sig", "x", self.num_bins, self.limits, self.sig)
#         self.bkg_temp = tf.templates.Template("bkg", "x", self.num_bins, self.limits, self.bkg)
#         self.st.add_template("sig", self.sig_temp)
#         self.st.add_template("bkg", self.bkg_temp)
#
#         self.nll = self.st.create_nll(self.hdata)
#
#     def test_x0(self):
#         """Test the starting values for the minimizer provided by the likelihood."""
#         exp_x0 = np.array([self.sig_temp.yield_param_values, self.bkg_temp.yield_param_values])
#         exp_x0 = np.append(exp_x0, self.sig_temp.nui_param_values)
#         exp_x0 = np.append(exp_x0, self.bkg_temp.nui_param_values)
#         np.testing.assert_array_equal(self.nll.x0,
#                                       exp_x0)
#
#     def test_call(self):
#         """Test if return of the __call__ function is ok."""
#         pass
