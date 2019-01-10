
import unittest

import numpy as np
import pandas as pd
import scipy.stats

from templatefitter import AdvancedPoissonNegativeLogLikelihood, AdvancedCompositeTemplate, Histogram


class TestAdvancedPoissonNegativeLogLikelihood(unittest.TestCase):
    """Unittests for the AdvancePoisoonNegativeLogLikeLihood class."""
    def setUp(self):
        self.sig_df = pd.DataFrame({
            "x": [1, 1, 2, 5, 7],
            "weight": [1.0, 1.0, 2.0, 1.5, 3.0]
        })

        self.hsig = Histogram(2, (1., 7.), data=self.sig_df.x,
                              weights=self.sig_df.weight)

        self.bkg_df = pd.DataFrame({
            "x": [1, 1, 3, 3, 2, 4, 4, 5, 5, 7],
            "weight": [1.0, 1.0, 1.5, 2.0, 2.0, 1.0, 2.0, 2.5, 1.5, 3.0]
        })

        self.hbkg = Histogram(2, (1., 7.), data=self.bkg_df.x,
                              weights=self.bkg_df.weight)

        self.data = pd.DataFrame({
            "x": [1, 1,  5, 5, 7, 1, 1, 3, 3, 2, 4, 4, 7]
        })
        self.hdata = Histogram(2, (1., 7.), data=self.data.x.values)

        self.tc = AdvancedCompositeTemplate("x", 2, (1., 7.))
        self.tc.create_template("sig", self.sig_df)
        self.tc.create_template("bkg", self.bkg_df)

    def test_x0(self):
        """Test the starting values for the minimizer provided by the likelihood."""
        nll = AdvancedPoissonNegativeLogLikelihood(self.hdata, self.tc)

        expected_values = np.array([np.sum(self.hsig.bin_counts),
                                    np.sum(self.hbkg.bin_counts)])
        np.testing.assert_array_equal(nll.x0, expected_values)

        self.tc.set_yield("sig", 50)
        self.tc.set_yield("bkg", 100)
        expected_values = np.array([50, 100])
        np.testing.assert_array_equal(nll.x0, expected_values)

    def test_call(self):
        """Test if return of the __call__ function is ok."""

        nll = AdvancedPoissonNegativeLogLikelihood(self.hdata, self.tc)

        params = np.array([np.sum(self.hsig.bin_counts), np.sum(self.hsig.bin_counts), 0, 0, 0, 0])

        sig_expected = lambda x: x[0]*(self.hsig.bin_counts*(1 + x[[2, 3]]*self.hsig.bin_rel_errors)/
                                       np.sum(self.hsig.bin_counts*(1 + x[[2, 3]]*self.hsig.bin_rel_errors)))

        bkg_expected = lambda x: x[1]*(self.hbkg.bin_counts*(1 + x[[4, 5]]*self.hbkg.bin_rel_errors)/
                                       np.sum(self.hbkg.bin_counts*(1 + x[[4, 5]]*self.hbkg.bin_rel_errors)))

        total_expected = np.sum(np.array([sig_expected(params), bkg_expected(params)]), axis=0)
        poisson_term = np.sum(total_expected - self.hdata.bin_counts*np.log(total_expected))
        gauss_term = 0.5*np.sum(params[2:]**2)
        np.testing.assert_approx_equal(nll(params), poisson_term + gauss_term)

        params = np.array([14, 23, 0.1, 0.05, 0.03, 0.12])
        total_expected = np.sum(np.array([sig_expected(params), bkg_expected(params)]), axis=0)
        poisson_term = np.sum(total_expected - self.hdata.bin_counts*np.log(total_expected))
        gauss_term = 0.5*np.sum(params[2:]**2)
        np.testing.assert_approx_equal(nll(params), poisson_term + gauss_term)
