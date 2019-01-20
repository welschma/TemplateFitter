import templatefitter
import unittest
import unittest.mock as mock

import numpy as np
import pandas as pd

from scipy.stats import binned_statistic

import logging

logging.basicConfig(level=logging.DEBUG)


class TestAbstractTemplate(unittest.TestCase):
    """Test properties of the AbstractTemplateClass."""

    def setUp(self):
        self.data = [1, 1, 2, 5, 7]
        self.weights = [1.0, 1.0, 2.0, 1.5, 3.0]
        self.n_bins = 2
        self.limits = (1.0, 7.0)
        self.bin_edges = [1.0, 4.0, 7.0]
        self.name = "test_template"
        self.var_id = "x"
        self.weight_id = "__weight__"
        self.df = pd.DataFrame({
            self.var_id: self.data,
            self.weight_id: self.weights
        })

    @mock.patch.multiple(
        templatefitter.AbstractTemplate, __abstractmethods__=set())
    def test_properties(self):
        template = templatefitter.AbstractTemplate(self.name, self.var_id,
                                                   self.n_bins, self.limits,
                                                   self.df, self.weight_id)
        self.assertEqual(template.name, self.name)
        self.assertEqual(template.variable, self.var_id)
        self.assertEqual(template.num_bins, self.n_bins)
        self.assertEqual(template.limits, self.limits)
        self.assertListEqual(template.bin_edges.tolist(), self.bin_edges)
        self.assertListEqual(template.bin_mids.tolist(), [2.5, 5.5])
        self.assertEqual(template.bin_width, 3.0)
        self.assertEqual(template.yield_value, np.sum(self.weights))
        self.assertEqual(template.yield_error,
                         np.sqrt(np.sum(np.array(self.weights)**2)))


class TestSimpleTemplate(unittest.TestCase):
    pass


class TestAdvancedTemplate(unittest.TestCase):
    pass
