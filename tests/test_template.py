import unittest

import numpy as np
import pandas as pd
import scipy.stats

from templatefitter import Template

class TestTemplate(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 1, 2, 5, 7],
            "weight": [1.0, 1.0, 2.0, 1.5, 3.0]
        })

    def test_default_constructor(self):
        template = Template("test", "x", 2, (1., 7.), self.df)

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