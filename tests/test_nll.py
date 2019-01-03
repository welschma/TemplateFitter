
import unittest

import numpy as np
import pandas as pd
import scipy.stats

from templatefitter import Template, TemplateCollection, PoissonNLL

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

        self.tc = TemplateCollection("x", 2, (1., 7.))
        self.tc.add_template("sig", self.sig_df)
        self.tc.add_template("bkg", self.bkg_df)

        self.sig_temp = Template("sig", "x", 2, (1., 7.), self.sig_df)
        self.bkg_temp = Template("bkg", "x", 2, (1., 7.), self.bkg_df)
    
    def test_fraction_matrix(self):

        nll = PoissonNLL(self.data, self.tc)

        exp_fractions = np.array([
            self.sig_temp.values/np.sum(self.sig_temp.values),
            self.bkg_temp.values/np.sum(self.bkg_temp.values)
        ])

        np.testing.assert_array_equal(nll.fraction_matrix(), exp_fractions)

    def test_call(self):

        nll = PoissonNLL(self.data, self.tc)

        hdata = np.histogram(self.data.x, bins=self.tc.bin_edges)[0]

        test_fractions = np.array([
            self.sig_temp.values/np.sum(self.sig_temp.values),
            self.bkg_temp.values/np.sum(self.bkg_temp.values)
        ])

        np.testing.assert_array_equal(nll.fraction_matrix(), test_fractions)
        
        test_params = np.array([4,5])
        test_evts_per_bin = np.matmul(test_params, test_fractions)
        test_nll = np.sum(test_evts_per_bin - np.matmul(
            np.log(test_evts_per_bin), hdata.reshape(-1,1))
        ) 
        
        np.testing.assert_array_equal(nll(test_params), test_nll)
