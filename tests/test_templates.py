# import unittest
#
# import pandas as pd
# import scipy.stats
# import numpy as np
#
# from templatefitter.templates import Template, StackedTemplate
# from templatefitter.utility import cov2corr
# from templatefitter.histogram import Hist1d
#
#
# class TestTemplate(unittest.TestCase):
#     """Test suite fore Template class.
#     """
#
#     def setUp(self):
#
#         self.df = pd.DataFrame({
#             "x": np.array([0.5, 0.2, 0.4,
#                            1.2, 1.3, 1.4,
#                            2.4, 2.7, 2.8]),
#             "weight": np.array([2, 2, 1,
#                                 1, 1, 2,
#                                 2, 2, 2])
#         })
#
#         self.limits = (0., 3.)
#         self.num_bins = 3
#         self.var = "x"
#
#         self.x_hist = scipy.stats.binned_statistic(
#             self.df.x, self.df.weight, "sum",
#             bins=self.num_bins, range=self.limits
#         )[0]
#         self.x_errors = np.sqrt(scipy.stats.binned_statistic(
#             self.df.x, self.df.weight.values**2, "sum",
#             bins=self.num_bins, range=self.limits
#         )[0])
#
#     def test_template_init_state(self):
#         template = Template("test", self.var, self.num_bins, self.limits, self.df)
#
#         np.testing.assert_equal(
#             template.yield_param.value,
#             np.sum(self.df.weight)
#         )
#         np.testing.assert_equal(
#             template.yield_param.error,
#             np.sqrt(np.sum(self.df.weight.values**2))
#         )
#         np.testing.assert_array_equal(
#             template.nui_params.value,
#             np.zeros(self.num_bins)
#         )
#         np.testing.assert_array_equal(
#             template.nui_params.error,
#             np.ones(self.num_bins)
#         )
#         np.testing.assert_array_equal(
#             template.values(),
#             self.x_hist
#         )
#         np.testing.assert_array_equal(
#             template._relative_errors,
#             self.x_errors/self.x_hist
#         )
#         np.testing.assert_array_equal(
#             template.errors(),
#             self.x_errors
#         )
#         np.testing.assert_array_almost_equal(
#             template._cov,
#             np.diag(self.x_errors**2)
#         )
#         np.testing.assert_array_equal(
#             template._corr,
#             np.diag(np.ones(self.num_bins))
#         )
#         np.testing.assert_array_equal(
#             template._inv_corr,
#             np.diag(np.ones(self.num_bins))
#         )
#
#     def test_fractions(self):
#         template = Template("test", self.var, self.num_bins, self.limits, self.df)
#
#         nui_params = np.array([0, 0, 0])
#         fractions_wo_nui_params = self.x_hist/np.sum(self.x_hist)
#         np.testing.assert_array_equal(
#             template.fractions(nui_params),
#                                fractions_wo_nui_params
#         )
#
#         nui_params = np.array([0.5, 0.5, 0.5])
#         rel_errors = self.x_errors/self.x_hist
#         fractions_w_nui_params = self.x_hist*(1 + nui_params*rel_errors)/np.sum(self.x_hist*(1 + nui_params*rel_errors))
#         np.testing.assert_array_equal(
#             template.fractions(nui_params),
#             fractions_w_nui_params
#         )
#
#     def test_template_init_state_with_empty_bins(self):
#         self.df = pd.DataFrame({
#             "x": np.array([0.5, 0.2, 0.4,
#                            2.4, 2.7, 2.8]),
#             "weight": np.array([2, 2, 1,
#                                 2, 2, 2])
#         })
#
#         self.x_hist = scipy.stats.binned_statistic(
#             self.df.x, self.df.weight, "sum",
#             bins=self.num_bins, range=self.limits
#         )[0]
#         self.x_errors = np.sqrt(scipy.stats.binned_statistic(
#             self.df.x, self.df.weight.values**2, "sum",
#             bins=self.num_bins, range=self.limits
#         )[0])
#
#         expected_rel_errors = np.divide(
#             self.x_errors,
#             self.x_hist,
#             out=np.full(self.num_bins, 1e-7),
#             where=self.x_hist!=0
#         )
#
#         template = Template("test", self.var, self.num_bins, self.limits, self.df)
#
#         np.testing.assert_equal(
#             template.yield_param.value,
#             np.sum(self.df.weight)
#         )
#         np.testing.assert_equal(
#             template.yield_param.error,
#             np.sqrt(np.sum(self.df.weight.values**2))
#         )
#         np.testing.assert_array_equal(
#             template.nui_params.value,
#             np.zeros(self.num_bins)
#         )
#         np.testing.assert_array_equal(
#             template.nui_params.error,
#             np.ones(self.num_bins)
#         )
#         np.testing.assert_array_equal(
#             template.values(),
#             self.x_hist
#         )
#         np.testing.assert_array_equal(
#             template._relative_errors,
#             expected_rel_errors
#         )
#         np.testing.assert_array_equal(
#             template.errors(),
#             expected_rel_errors*self.x_hist
#         )
#
#     def test_template_add_cov_mat(self):
#         template = Template("test", self.var, self.num_bins, self.limits, self.df)
#
#         stat_cov_mat = np.diag(self.x_errors**2)
#         cov_mat = np.array([[2, 1, 0], [1, 3, 0], [0, 0, 1]])
#         expected_cov_mat = stat_cov_mat+cov_mat
#
#         template.add_covariance_matrix(cov_mat)
#
#         new_x_errors = np.sqrt(np.diag(expected_cov_mat))
#         np.testing.assert_array_equal(
#             template._relative_errors,
#             new_x_errors/self.x_hist
#         )
#         np.testing.assert_array_almost_equal(
#             template._cov,
#             expected_cov_mat
#         )
#         np.testing.assert_array_almost_equal(
#             template._corr,
#             cov2corr(expected_cov_mat)
#         )
#         np.testing.assert_array_almost_equal(
#             template._inv_corr,
#             np.linalg.inv(cov2corr(expected_cov_mat))
#         )
#
#     def test_template_update_params(self):
#         template = Template("test", self.var, self.num_bins, self.limits, self.df)
#
#         new_yield_param = 20
#         template.yield_param.value = new_yield_param
#         new_nui_params = np.random.randn(self.num_bins)
#         template.nui_params.value = new_nui_params
#
#         np.testing.assert_equal(template.yield_param.value, new_yield_param)
#         np.testing.assert_array_equal(
#             template.values(),
#             new_yield_param*template.fractions(new_nui_params)
#         )
#         np.testing.assert_array_equal(
#             template.nui_params.value,
#             new_nui_params
#         )
#
#
# class TestStackedTemplate(unittest.TestCase):
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
#         self.limits = (0., 3.)
#         self.num_bins = 3
#         self.var = "x"
#         self.num_templates = 2
#
#         self.sig_hist = Hist1d(self.num_bins, self.limits, self.sig.x, self.sig.weight)
#         self.bkg_hist = Hist1d(self.num_bins, self.limits, self.bkg.x, self.bkg.weight)
#
#         self.st = StackedTemplate("test", "x", self.num_bins, self.limits)
#         self.sig_temp = Template("sig", "x", self.num_bins, self.limits, self.sig)
#         self.bkg_temp = Template("bkg", "x", self.num_bins, self.limits, self.bkg)
#         self.st.add_template("sig", self.sig_temp)
#         self.st.add_template("bkg", self.bkg_temp)
#
#     def test_add_template(self):
#         sig_temp = Template("sig", "x", self.num_bins, self.limits, self.sig)
#         bkg_temp = Template("bkg", "x", self.num_bins, self.limits, self.bkg)
#
#         stacked_temp = StackedTemplate("test", "x", self.num_bins, self.limits)
#         stacked_temp.add_template("sig", sig_temp)
#         stacked_temp.add_template("bkg", bkg_temp)
#
#         self.assertEqual(stacked_temp.num_templates, 2)
#         self.assertListEqual(stacked_temp.template_names, ["sig", "bkg"])
#
#     def test_add_not_valid_template(self):
#         stacked_temp = StackedTemplate("test", "x", self.num_bins, self.limits)
#
#         sig_temp = Template("sig", "x", 2, self.limits, self.sig)
#         self.assertRaises(ValueError, stacked_temp.add_template, "sig", sig_temp)
#
#         sig_temp = Template("sig", "x", self.num_bins, (-4, 6), self.sig)
#         self.assertRaises(ValueError, stacked_temp.add_template, "sig", sig_temp)
#
#     def test_create_template(self):
#         stacked_temp = StackedTemplate("test", "x", self.num_bins, self.limits)
#         stacked_temp.create_template("sig", self.sig)
#         stacked_temp.create_template("bkg", self.bkg)
#
#         self.assertEqual(stacked_temp.num_templates, 2)
#         self.assertListEqual(stacked_temp.template_names, ["sig", "bkg"])
#
#     def test_fractions(self):
#
#         nui_params = np.zeros(2*self.st.num_bins)
#         exp_fractions_wo_nui_params = np.array(
#             [
#                 self.sig_temp.fractions(nui_params[:self.num_bins]),
#                 self.bkg_temp.fractions(nui_params[self.num_bins:])
#             ]
#         )
#         np.testing.assert_array_equal(self.st.fractions(nui_params), exp_fractions_wo_nui_params)
#
#         nui_params = np.random.randn(2*self.num_bins)
#         exp_fractions_w_nui_params = np.array(
#             [
#                 self.sig_temp.fractions(nui_params[:self.num_bins]),
#                 self.bkg_temp.fractions(nui_params[self.num_bins:])
#             ]
#         )
#         np.testing.assert_array_equal(self.st.fractions(nui_params), exp_fractions_w_nui_params)
#
#     def test_values(self):
#         exp_values = self.sig_hist.bin_counts + self.bkg_hist.bin_counts
#         np.testing.assert_array_equal(self.st.values(), exp_values)
#
#     def test_errors(self):
#         exp_errors = np.sqrt(self.sig_hist.bin_errors_sq + self.bkg_hist.bin_errors_sq)
#         np.testing.assert_array_equal(self.st.errors(), exp_errors)
#
#     def test_param_values(self):
#         exp_yields = np.array([np.sum(self.sig_hist.bin_counts),
#                                np.sum(self.bkg_hist.bin_counts)])
#         np.testing.assert_array_equal(self.st.yield_param_values, exp_yields)
#         np.testing.assert_array_equal(self.st.nui_param_values,
#                                       np.zeros((self.num_templates, self.num_bins)))
#
#     def test_getting_parameters(self):
#         np.testing.assert_array_equal(
#             self.st.yield_param_values,
#             np.array([self.sig_temp.yield_param_values, self.bkg_temp.yield_param_values])
#         )
#         np.testing.assert_array_equal(
#             self.st.yield_param_errors,
#             np.array([self.sig_temp.yield_param_errors, self.bkg_temp.yield_param_errors])
#         )
#         np.testing.assert_array_equal(
#             self.st.nui_param_values,
#             np.vstack((self.sig_temp.nui_param_values, self.bkg_temp.nui_param_values))
#         )
#         np.testing.assert_array_equal(
#             self.st.nui_param_errors,
#             np.vstack((self.sig_temp.nui_params_errors, self.bkg_temp.nui_params_errors))
#         )
#
#     def test_setting_parameters(self):
#         new_yield_values = np.array([10, 200])
#         new_yield_errors = np.sqrt(np.array([10, 200]))
#         self.st.yield_param_values = new_yield_values
#         self.st.yield_param_errors = new_yield_errors
#         np.testing.assert_array_equal(
#             self.st.yield_param_values,
#             new_yield_values
#         )
#         np.testing.assert_array_equal(
#             self.st.yield_param_errors,
#             new_yield_errors
#         )
#
#         new_nui_values = np.random.randn(2*self.num_bins)
#         new_nui_errors = np.random.randn(2*self.num_bins) + 1
#         self.st.nui_param_values = new_nui_values
#         self.st.nui_param_errors = new_nui_errors
#
#         np.testing.assert_array_equal(
#             self.st.nui_param_values,
#             new_nui_values.reshape((self.num_templates, self.num_bins))
#         )
#         np.testing.assert_array_equal(
#             self.st.nui_param_errors,
#             new_nui_errors.reshape((self.num_templates, self.num_bins))
#         )
#
#     def test_update_parameters(self):
#         new_values = np.random.randn(self.num_templates + self.num_templates*self.num_bins)
#         new_errors = np.random.randn(self.num_templates + self.num_templates*self.num_bins)
#
#         self.st.update_parameters(new_values, new_errors)
#
#         exp_yield_values= new_values[:self.num_templates]
#         exp_nui_params = new_values[self.num_templates:].reshape((self.num_templates, self.num_bins))
#         exp_yield_errors = new_errors[:self.num_templates]
#         exp_nui_params_errors = new_errors[self.num_templates:].reshape((self.num_templates, self.num_bins))
#
#         np.testing.assert_array_equal(
#             self.st.yield_param_values,
#             exp_yield_values
#         )
#         np.testing.assert_array_equal(
#             self.st.yield_param_errors,
#             exp_yield_errors
#         )
#         np.testing.assert_array_equal(
#             self.st.nui_param_values,
#             exp_nui_params
#         )
#         np.testing.assert_array_equal(
#             self.st.nui_param_errors,
#             exp_nui_params_errors
#         )
#
#
# class TestSimultaneousTemplate(unittest.TestCase):
#     pass
