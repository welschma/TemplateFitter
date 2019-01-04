import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from templatefitter import Template, TemplateCollection, PoissonNLL, LikelihoodFitter

class TestPoissonNLL(unittest.TestCase):

    def setUp(self):
        nsig = 10000
        self.sig_df = pd.DataFrame({
            "x": np.random.randn(nsig)*4+125,
            "weight": np.random.randn(nsig)*1e-2 + 1 
        })
        
        nbkg=50000
        self.bkg_df = pd.DataFrame({
            "x": np.random.exponential(scale=10, size=nbkg) + 100,
            "weight": np.random.randn(nbkg)*1e-2 + 1.1 
        })


        self.tc = TemplateCollection("x", 100, (100., 140.))
        self.tc.add_template("sig", self.sig_df)
        self.tc.add_template("bkg", self.bkg_df)

        self.fake_data = self.tc.generate_toy_data()

        # plt.hist([self.bkg_df.x, self.sig_df.x], bins=np.linspace(100, 140, 101),
        #     weights=[self.bkg_df.weight, self.sig_df.weight],stacked=True)
        # plt.plot(self.tc.bin_mids, self.fake_data, ls='', marker='.')
        # plt.show()

        self.nll = PoissonNLL(self.fake_data, self.tc)


    def test_fit(self):
        lf = LikelihoodFitter(self.nll)
        result = lf.minimize()

        self.assertTrue(
            result.x[0] - 3*np.sqrt(result.covariance[0,0]) < self.tc.yields[0] < 
            result.x[0] + 3*np.sqrt(result.covariance[0,0])
        ) 

        self.assertTrue(
            result.x[1] - 3*np.sqrt(result.covariance[1,1]) < self.tc.yields[1] < 
            result.x[1] + 3*np.sqrt(result.covariance[1,1])
        ) 