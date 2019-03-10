import unittest

import numpy as np
import pandas as pd

from templatefitter.histograms import Hist1d
from templatefitter.templates import MultiChannelTemplate, Channel, Template1d

iris = pd.read_csv("./iris.csv", sep=",", header=0)

setosa = iris.query("class_id==1")["sepal_length"].values
versicolor = iris.query("(class_id==2)")["sepal_length"].values
virginica = iris.query("(class_id==3)")["sepal_length"].values

channel_1 = iris.query("sepal_width <= 3.1")
channel_2 = iris.query("sepal_width > 3.1")


class TestMultiChannelTemplate(unittest.TestCase):

    def setUp(self):
        self.bins = 10
        self.range = (3, 8)
        self.variable = "sepal_length"

        hsetosa = Hist1d(self.bins, range=self.range, data=setosa)
        self.tsetosa = Template1d("setosa", "length", hsetosa)
        hversico = Hist1d(self.bins, range=self.range, data=versicolor)
        self.tversico = Template1d("versicolor", "length", hversico)
        hvirgini = Hist1d(self.bins, range=self.range, data=virginica)
        self.tvirgini = Template1d("virginica", "length", hvirgini)

        self.processes = ["setosa", "versicolor", "virginica"]

        self.channels = ["Test1", "Test2"]

        self.templates = [self.tsetosa, self.tversico, self.tvirgini]
        self.num_templates = len(self.templates)
        self.efficiencies = 0.8

    def test_add_processes(self):
        mct = MultiChannelTemplate()

        for process in self.processes:
            mct.define_process(process)

        self.assertSetEqual(mct.processes, set(self.processes))
        self.assertDictEqual(mct.channels, {})

        mct.define_channel("test1", "length", self.bins, self.range)
        self.assertSetEqual(mct.processes, set(self.processes))

        for channel in mct.channels.values():
            for process in channel.processes:
                channel_temp = channel._template_dict[process]
                self.assertTrue(isinstance(channel_temp, Template1d))
                np.testing.assert_array_equal(channel_temp.values, np.zeros(self.bins))

    def test_add_channel(self):
        mct = MultiChannelTemplate()

        for channel in self.channels:
            mct.define_channel(channel, self.variable, self.bins, self.range)

        self.assertSetEqual(mct.processes, set())

        for channel in mct.channels.values():
            self.assertTrue(isinstance(channel, Channel))

        for process in self.processes:
            mct.define_process(process)

        self.assertSetEqual(mct.processes, set(self.processes))

        for channel in mct.channels.values():
            for process in channel.processes:
                channel_temp = channel._template_dict[process]
                self.assertTrue(isinstance(channel_temp, Template1d))
                np.testing.assert_array_equal(channel_temp.values, np.zeros(self.bins))
