# import unittest
# import numpy as np
# import pandas as pd
#
# import logging
#
# logging.basicConfig(level=logging.DEBUG)
#
# from templatefitter import (TemplateFitter, AdvancedCompositeTemplate, Histogram, AdvancedPoissonNegativeLogLikelihood)
#
#
# class TestTemplateFitter(unittest.TestCase):
#
#     def setUp(self):
#         nsig = 10000
#         self.sig_df = pd.DataFrame({
#             "x": np.random.randn(nsig)*4+125,
#             "weight": np.random.randn(nsig)*1e-2 + 1
#         })
#
#         nbkg=50000
#         self.bkg_df = pd.DataFrame({
#             "x": np.random.exponential(scale=10, size=nbkg) + 100,
#             "weight": np.random.randn(nbkg)*1e-2 + 1.1
#         })
#
#         self.tc = AdvancedCompositeTemplate("x", 40, (100., 140.))
#         self.tc.create_template("bkg", self.bkg_df)
#         self.tc.create_template("sig", self.sig_df)
#
#         self.hfake_data = Histogram(40, (100., 140.))
#         self.hfake_data.bin_counts = self.tc.generate_toy_dataset()
#
#     def test_do_fit(self):
#         logging.debug("start")
#         logging.debug(self.tc.yield_values)
#         fitter = TemplateFitter(self.hfake_data, self.tc, AdvancedPoissonNegativeLogLikelihood)
#         params = fitter.do_fit(get_hesse=False)
#
#         print(params.values)
