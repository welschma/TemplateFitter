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

        self.tsetosa = Template1d(
            "setosa", "length", self.bins, self.range, data=setosa
        )
        self.tversico = Template1d(
            "versicolor", "length", self.bins, self.range, data=versicolor
        )
        self.tvirgini = Template1d(
            "virginica", "length", self.bins, self.range, data=virginica
        )

        self.channel = Channel("test")

        self.processes = ["setosa", "versicolor", "virginica"]
        self.templates = [self.tsetosa, self.tversico, self.tvirgini]
        self.num_templates = len(self.templates)
        for process, template in zip(self.processes, self.templates):
            self.channel.add_template(process, template)

    def test_num_templates(self):
        self.assertEqual(self.channel.num_templates, len(self.templates))

    def test_add_non_comp_template(self):
        with self.assertRaises(RuntimeError) as e:
            self.channel.add_template(
                'bla', Template1d('test', 'test', 5, (3, 4), data=setosa))

        self.assertEqual("Trying to add a non compatible template with the Channel.",
                         str(e.exception))

    def test_add_data(self):
        hiris = Hist1d(self.bins, range=self.range, data=iris_data)
        self.channel.add_data(hiris)

    def test_add_data_wo_templates(self):
        empty_channel = Channel("test_empty")
        hiris = Hist1d(self.bins, range=self.range, data=iris_data)
        with self.assertRaises(RuntimeError) as e:
            empty_channel.add_data(hiris)

        self.assertEqual("You have to add at least one template before the data.",
                         str(e.exception))

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

        nll_contrib = self.channel.nll_contibution(yields, nui_params)
        print(nll_contrib)
        self.assertEqual(type(nll_contrib), np.float64)
