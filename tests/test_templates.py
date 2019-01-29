import unittest

import pandas as pd
import scipy.stats
import numpy as np

from templatefitter.templates import Template, StackedTemplate, SimultaneousTemplate
from templatefitter.utility import cov2corr

class TestTemplate(unittest.TestCase):
    """Test suite fore Template class.
    """

    def setUp(self):

        self.df = pd.DataFrame({
            "x": np.array([0.5, 0.2, 0.4,
                           1.2, 1.3, 1.4,
                           2.4, 2.7, 2.8]),
            "weight": np.array([2, 2, 1,
                                1, 1, 2,
                                2, 2, 2])
        })

        self.limits = (0., 3.)
        self.num_bins = 3
        self.var = "x"

        self.x_hist = scipy.stats.binned_statistic(
            self.df.x, self.df.weight, "sum",
            bins=self.num_bins, range=self.limits
        )[0]
        self.x_errors = np.sqrt(scipy.stats.binned_statistic(
            self.df.x, self.df.weight.values**2, "sum",
            bins=self.num_bins, range=self.limits
        )[0])

    def test_template_init_state(self):
        template = Template("test", self.var, self.num_bins, self.limits, self.df)

        np.testing.assert_equal(
            template.yield_param.value,
            np.sum(self.df.weight)
        )
        np.testing.assert_equal(
            template.yield_param.error,
            np.sqrt(np.sum(self.df.weight.values**2))
        )
        np.testing.assert_array_equal(
            template.nui_params.value,
            np.zeros(self.num_bins)
        )
        np.testing.assert_array_equal(
            template.nui_params.error,
            np.ones(self.num_bins)
        )
        np.testing.assert_array_equal(
            template.values(),
            self.x_hist
        )
        np.testing.assert_array_equal(
            template._relative_errors,
            self.x_errors/self.x_hist
        )
        np.testing.assert_array_equal(
            template.errors(),
            self.x_errors
        )
        np.testing.assert_array_almost_equal(
            template._cov,
            np.diag(self.x_errors**2)
        )
        np.testing.assert_array_equal(
            template._corr,
            np.diag(np.ones(self.num_bins))
        )
        np.testing.assert_array_equal(
            template._inv_corr,
            np.diag(np.ones(self.num_bins))
        )

    def test_template_init_state_with_empty_bins(self):
        self.df = pd.DataFrame({
            "x": np.array([0.5, 0.2, 0.4,
                           2.4, 2.7, 2.8]),
            "weight": np.array([2, 2, 1,
                                2, 2, 2])
        })

        self.x_hist = scipy.stats.binned_statistic(
            self.df.x, self.df.weight, "sum",
            bins=self.num_bins, range=self.limits
        )[0]
        self.x_errors = np.sqrt(scipy.stats.binned_statistic(
            self.df.x, self.df.weight.values**2, "sum",
            bins=self.num_bins, range=self.limits
        )[0])

        expected_rel_errors = np.divide(
            self.x_errors,
            self.x_hist,
            out=np.full(self.num_bins, 1e-7),
            where=self.x_hist!=0
        )

        template = Template("test", self.var, self.num_bins, self.limits, self.df)

        np.testing.assert_equal(
            template.yield_param.value,
            np.sum(self.df.weight)
        )
        np.testing.assert_equal(
            template.yield_param.error,
            np.sqrt(np.sum(self.df.weight.values**2))
        )
        np.testing.assert_array_equal(
            template.nui_params.value,
            np.zeros(self.num_bins)
        )
        np.testing.assert_array_equal(
            template.nui_params.error,
            np.ones(self.num_bins)
        )
        np.testing.assert_array_equal(
            template.values(),
            self.x_hist
        )
        np.testing.assert_array_equal(
            template._relative_errors,
            expected_rel_errors
        )
        np.testing.assert_array_equal(
            template.errors(),
            expected_rel_errors*self.x_hist
        )

    def test_template_add_cov_mat(self):
        template = Template("test", self.var, self.num_bins, self.limits, self.df)

        stat_cov_mat = np.diag(self.x_errors**2)
        cov_mat = np.array([[2, 1, 0], [1, 3, 0], [0, 0, 1]])
        expected_cov_mat = stat_cov_mat+cov_mat

        template.add_covariance_matrix(cov_mat)

        new_x_errors = np.sqrt(np.diag(expected_cov_mat))
        np.testing.assert_array_equal(
            template._relative_errors,
            new_x_errors/self.x_hist
        )
        np.testing.assert_array_almost_equal(
            template._cov,
            expected_cov_mat
        )
        np.testing.assert_array_almost_equal(
            template._corr,
            cov2corr(expected_cov_mat)
        )
        np.testing.assert_array_almost_equal(
            template._inv_corr,
            np.linalg.inv(cov2corr(expected_cov_mat))
        )

    def test_template_update_params(self):
        template = Template("test", self.var, self.num_bins, self.limits, self.df)

        new_yield_param = 20
        template.yield_param.value = new_yield_param
        new_nui_params = np.random.randn(self.num_bins)
        template.nui_params.value = new_nui_params

        np.testing.assert_equal(template.yield_param.value, new_yield_param)
        np.testing.assert_array_equal(
            template.values(),
            new_yield_param*template.fractions(new_nui_params)
        )
        np.testing.assert_array_equal(
            template.nui_params.value,
            new_nui_params
        )





class TestStackedTemplate(unittest.TestCase):
    pass


class TestSimultaneousTemplate(unittest.TestCase):
    pass
