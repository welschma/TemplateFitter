
import unittest

import numpy as np
import pandas as pd
import scipy.stats

from templatefitter import TemplateModel, CompositeTemplateModel, PoissonNLL, Histogram

class TestPoissonNLL(unittest.TestCase):

    def setUp(self):
        self.sig_df = pd.DataFrame({
            "x": [1, 1, 2, 5, 7],
            "weight": [1.0, 1.0, 2.0, 1.5, 3.0]
        })
        
        self.bkg_df = pd.DataFrame({
            "x": [1, 1, 3, 3, 2, 4, 4, 5, 5, 7],
            "weight": [1.0, 1.0, 1.5, 2.0, 2.0, 1.0, 2.0, 2.5, 1.5, 3.0]
        })

        self.data = pd.DataFrame({
            "x": [1, 1,  5, 5, 7, 1, 1, 3, 3, 2, 4, 4, 7]
        })

        self.tc = CompositeTemplateModel("x", 2, (1., 7.))
        self.tc.add_template("sig", self.sig_df)
        self.tc.add_template("bkg", self.bkg_df)

        self.hdata = Histogram(2, (1., 7.), data=self.data.x.values)

        self.sig_temp = TemplateModel("sig", "x", 2, (1., 7.), self.sig_df)
        self.bkg_temp = TemplateModel("bkg", "x", 2, (1., 7.), self.bkg_df)
    
    def test_fraction_matrix(self):

        nll = PoissonNLL(self.hdata, self.tc)

        exp_fractions = np.array([
            self.sig_temp.values/np.sum(self.sig_temp.values),
            self.bkg_temp.values/np.sum(self.bkg_temp.values)
        ])

        np.testing.assert_array_equal(nll.fraction_matrix(), exp_fractions)

    def test_call(self):

        nll = PoissonNLL(self.hdata, self.tc)

        test_fractions = np.array([
            self.sig_temp.values/np.sum(self.sig_temp.values),
            self.bkg_temp.values/np.sum(self.bkg_temp.values)
        ])

        np.testing.assert_array_equal(nll.fraction_matrix(), test_fractions)
        
        test_params = np.array([4,5])
        test_evts_per_bin = test_params@test_fractions
        test_nll = np.sum(
            test_evts_per_bin - np.log(test_evts_per_bin)*self.hdata.bin_counts
        )
        
        np.testing.assert_array_equal(nll(test_params), test_nll)
