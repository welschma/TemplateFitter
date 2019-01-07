import unittest

import numpy as np
import pandas as pd
import scipy.stats

from templatefitter import TemplateModel, CompositeTemplateModel

class TestTemplate(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 1, 2, 5, 7],
            "weight": [1.0, 1.0, 2.0, 1.5, 3.0]
        })

    def test_default_constructor(self):
        template = TemplateModel("test", "x", 2, (1., 7.), self.df)

        true_counts = scipy.stats.binned_statistic(
            self.df["x"], self.df["weight"], 'sum', template.bin_edges
            )[0]

        true_errors_sq = scipy.stats.binned_statistic(
            self.df["x"], self.df["weight"].values**2, 'sum', template.bin_edges
            )[0]

        np.testing.assert_equal(
            template.values,
            true_counts
        )

        np.testing.assert_almost_equal(
            template.errors,
            np.sqrt(true_errors_sq)
        )

        np.testing.assert_almost_equal(
            template.rel_errors,
            np.sqrt(true_errors_sq)/true_counts
        )

    def test_set_yield(self):

        template = TemplateModel("test", "x", 2, (1., 7.), self.df)

        true_counts = scipy.stats.binned_statistic(
            self.df["x"], self.df["weight"], 'sum', template.bin_edges
            )[0]

        self.assertEqual(template.expected_yield, np.sum(self.df["weight"]))

        template.expected_yield = 500


        self.assertEqual(template.expected_yield, 500)
        np.testing.assert_array_equal(template.values, true_counts*500/np.sum(true_counts))


class TestTemplateCollection(unittest.TestCase):

    def setUp(self):
        self.sig_df = pd.DataFrame({
            "x": [1, 1, 2, 5, 7],
            "weight": [1.0, 1.0, 2.0, 1.5, 3.0]
        })
        
        self.bkg_df = pd.DataFrame({
            "x": [1, 1, 3, 3, 2, 4, 4, 5, 5, 7],
            "weight": [1.0, 1.0, 1.5, 2.0, 2.0, 1.0, 2.0, 2.5, 1.5, 3.0]
        })

        self.tc = CompositeTemplateModel("x", 2, (1., 7.))
        self.tc.add_template("sig", self.sig_df)
        self.tc.add_template("bkg", self.bkg_df)
    
    def test_set_yields(self):
        self.tc.set_yields(sig=1000, bkg=5000)
        np.testing.assert_array_equal(self.tc.yields, np.array([1000, 5000]))

    def test_yields(self):

        expected_yields = np.array([
            np.sum(self.sig_df["weight"]),
            np.sum(self.bkg_df["weight"])
            ])

        np.testing.assert_array_equal(self.tc.yields, expected_yields)

    @unittest.skip("Skip test_generate_toy_data (takes a lot of time).")
    def test_generate_toy_data(self):
        generator_yields = np.sum(self.tc.values, axis=0)
        toy_data_samples = np.array([self.tc.generate_toy_data() for _ in range(100000)])
        np.testing.assert_array_almost_equal(
            np.mean(toy_data_samples, axis=0),
            generator_yields,
            decimal=1
        )

    def test_value_matrix(self):

        expected_values = np.array(
            [[4, 4.5],
            [7.5, 10]])

        np.testing.assert_array_equal(self.tc.values, expected_values)
        self.assertEqual(self.tc.values.shape, expected_values.shape)

    def test_error_matrix(self):
        sig_counts = scipy.stats.binned_statistic(
            self.sig_df["x"], self.sig_df["weight"], "sum", np.linspace(1, 7, 3)
        )[0]

        bkg_counts = scipy.stats.binned_statistic(
            self.bkg_df["x"], self.bkg_df["weight"], "sum", np.linspace(1, 7, 3)
        )[0]

        sig_errors = scipy.stats.binned_statistic(
            self.sig_df["x"], self.sig_df["weight"].values**2, "sum", np.linspace(1, 7, 3)
        )[0]

        bkg_errors = scipy.stats.binned_statistic(
            self.bkg_df["x"], self.bkg_df["weight"].values**2, "sum", np.linspace(1, 7, 3)
        )[0]
        
        expected_values = np.array(
            [np.sqrt(sig_errors)/sig_counts,
            np.sqrt(bkg_errors)/bkg_counts])

        np.testing.assert_array_equal(self.tc.rel_errors, expected_values)
