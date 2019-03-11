import unittest

import numpy as np

from templatefitter.histograms import Hist1d
from templatefitter.templates import Channel, Template1d

iris_data = np.array(
    [5.1, 4.9, 4.7, 4.6, 5. , 5.4, 4.6, 5. , 4.4, 4.9, 5.4, 4.8, 4.8,
     4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5. ,
     5. , 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5. , 5.5, 4.9, 4.4,
     5.1, 5. , 4.5, 4.4, 5. , 5.1, 4.8, 5.1, 4.6, 5.3, 5. , 7. , 6.4,
     6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5. , 5.9, 6. , 6.1, 5.6,
     6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7,
     6. , 5.7, 5.5, 5.5, 5.8, 6. , 5.4, 6. , 6.7, 6.3, 5.6, 5.5, 5.5,
     6.1, 5.8, 5. , 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3,
     6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5,
     7.7, 7.7, 6. , 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
     7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6. , 6.9, 6.7, 6.9, 5.8,
     6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9]
)

setosa = iris_data[:50]
versicolor = iris_data[50:100]
virginica = iris_data[100:]


class TestChannel(unittest.TestCase):

    def setUp(self):

        self.bins = 10
        self.range = (3, 8)
        self.variable = "length"

        hsetosa = Hist1d(self.bins, range=self.range, data=setosa)
        self.tsetosa = Template1d(
            "setosa", "length", hsetosa
        )
        hversico = Hist1d(self.bins, range=self.range, data=versicolor)
        self.tversico = Template1d(
            "versicolor", "length", hversico
        )
        hvirgini = Hist1d(self.bins, range=self.range, data=virginica)
        self.tvirgini = Template1d(
            "virginica", "length", hvirgini
        )

        self.channel = Channel("test", self.bins, self.range)

        self.processes = ["setosa", "versicolor", "virginica"]
        self.templates = [self.tsetosa, self.tversico, self.tvirgini]
        self.num_templates = len(self.templates)
        self.efficiencies = 0.8
        for process, template in zip(self.processes, self.templates):
            self.channel.add_template(
                process, template, efficiency=self.efficiencies
            )

    def test_update_parameters(self):

        new_yields = np.array([35, 60, 80])
        new_nui_params = np.random.randn(self.bins*self.num_templates)
        per_template_nui_params = np.split(new_nui_params, self.num_templates)

        self.channel.update_parameters(new_yields, new_nui_params)

        for i, template in enumerate(self.channel.templates.values()):
            self.assertEqual(template.yield_param, new_yields[i]*self.efficiencies)
            np.testing.assert_array_equal(template.nui_params, per_template_nui_params[i])

    def test_process_indices(self):
        outer_process_list = ("bla", "virginica", "bla", "setosa", "versicolor")
        self.assertListEqual(self.channel.process_indices(outer_process_list),
                             [3, 4, 1])

    def test_num_templates(self):
        self.assertEqual(self.channel.num_templates, len(self.templates))

    def test_add_non_comp_template(self):
        with self.assertRaises(RuntimeError) as e:
            hsetosa = Hist1d( 5, (3, 4), data=setosa)
            self.channel.add_template(
                'bla', Template1d('test', 'test', hsetosa)
            )

        self.assertEqual("Trying to add a non compatible template with the Channel.",
                         str(e.exception))

    def test_add_data(self):
        hiris = Hist1d(self.bins, range=self.range, data=iris_data)
        self.channel.add_data(hiris)

    def test_add_data_wo_templates(self):
        empty_channel = Channel("test_empty", self.bins, self.range)
        hiris = Hist1d(self.bins, range=self.range, data=iris_data)
        empty_channel.add_data(hiris)

    def test_add_not_comp_data(self):
        hiris = Hist1d(2, range=self.range, data=iris_data)
        with self.assertRaises(RuntimeError) as e:
            self.channel.add_data(hiris)

        self.assertEqual("Given data histogram is not compatible with the Channel.",
                         str(e.exception))

    def test_inv_corr_mat_shape(self):
        inv_corr = self.channel._create_block_diag_inv_corr_mat()

        self.assertEqual(inv_corr.shape,
                         (self.num_templates*self.bins, self.num_templates*self.bins))

    def test_nll_contrib_is_scalar(self):
        np.random.seed(5)
        yields = np.random.randn(self.num_templates)+50
        nui_params = np.random.randn(self.num_templates*self.bins)
        hiris = Hist1d(self.bins, range=self.range, data=iris_data)
        self.channel.add_data(hiris)

        nll_contrib = self.channel.nll_contribution(yields, nui_params)
        self.assertEqual(type(nll_contrib), np.float64)

    def test_efficiency_array(self):
        expected = np.full((self.num_templates,), self.efficiencies)

        np.testing.assert_array_equal(
            self.channel._get_efficiency(), expected
        )
