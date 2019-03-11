import unittest
from functools import reduce

import numpy as np
import pandas as pd

from templatefitter.histograms import Hist1d, Hist2d
from templatefitter.templates import MultiChannelTemplate, Channel, Template1d, Template2d, NegLogLikelihood

iris = pd.read_csv("./iris.csv", sep=",", header=0)

setosa = iris.query("class_id==1")["sepal_length"].values
versicolor = iris.query("(class_id==2)")["sepal_length"].values
virginica = iris.query("(class_id==3)")["sepal_length"].values

channel_1 = iris.query("sepal_width <= 3.1")
channel_2 = iris.query("sepal_width > 3.1")


class TestMultiChannelTemplate1d(unittest.TestCase):

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

        self.processes = ("setosa", "versicolor", "virginica")

        self.channels = ["Test1", "Test2"]

        self.templates = [self.tsetosa, self.tversico, self.tvirgini]
        self.num_templates = len(self.templates)
        self.efficiencies = 0.8

    def setup_two_channel_mct(self):
        mct = MultiChannelTemplate()

        for channel_name in self.channels:
            mct.define_channel(
                name=channel_name, bins=self.bins, range=self.range, variable=self.variable
            )

        for process in self.processes:
            mct.define_process(process)

        for iris_channel, name in zip([channel_1, channel_2], self.channels):
            setosa = iris_channel.query("class_id==1")["sepal_length"].values
            versicolor = iris_channel.query("(class_id==2)")["sepal_length"].values
            virginica = iris_channel.query("(class_id==3)")["sepal_length"].values

            hsetosa = Hist1d(self.bins, range=self.range, data=setosa)
            tsetosa = Template1d("setosa", "length", hsetosa)
            hversico = Hist1d(self.bins, range=self.range, data=versicolor)
            tversico = Template1d("versicolor", "length", hversico)
            hvirgini = Hist1d(self.bins, range=self.range, data=virginica)
            tvirgini = Template1d("virginica", "length", hvirgini)

            mct.add_template(name, "setosa", tsetosa)
            mct.add_template(name, "versicolor", tversico)
            mct.add_template(name, "virginica", tvirgini)

        return mct

    def test_add_processes(self):
        mct = MultiChannelTemplate()

        for process in self.processes:
            mct.define_process(process)

        self.assertTupleEqual(mct.processes, self.processes)
        self.assertDictEqual(mct.channels, {})

        mct.define_channel("test1", "length", self.bins, self.range)
        self.assertListEqual(list(mct.channels.keys()), ["test1"])

    def test_add_channel(self):
        mct = MultiChannelTemplate()

        for channel in self.channels:
            mct.define_channel(channel, self.variable, self.bins, self.range)
        self.assertTupleEqual(mct.processes, tuple())
        for channel in mct.channels.values():
            self.assertTrue(isinstance(channel, Channel))
        for process in self.processes:
            mct.define_process(process)
        self.assertTupleEqual(mct.processes, self.processes)

    def test_add_channels_processes_and_templates(self):

        mct = self.setup_two_channel_mct()

        self.assertEqual(mct.num_channels, len(self.channels))
        self.assertEqual(mct.num_processes, len(self.processes))
        self.assertTupleEqual(mct.processes, self.processes)
        self.assertEqual(mct.num_nui_params,
                         self.bins * len(self.processes) * len(self.channels))

    def test_add_data(self):

        mct = self.setup_two_channel_mct()

        hdata_ch1 = Hist1d(self.bins, self.range, data= channel_1["sepal_length"])
        hdata_ch2 = Hist1d(self.bins, self.range, data= channel_2["sepal_length"])
        mct.add_data(Test1=hdata_ch1, Test2=hdata_ch2)

        for channel in mct.channels.values():
            self.assertTrue(
                channel.has_data
            )

    def test_add_data_as_dict(self):

        mct = self.setup_two_channel_mct()

        hdata_ch1 = Hist1d(self.bins, self.range, data= channel_1["sepal_length"])
        hdata_ch2 = Hist1d(self.bins, self.range, data= channel_2["sepal_length"])
        mct.add_data(**{"Test1": hdata_ch1, "Test2": hdata_ch2})

        for channel in mct.channels.values():
            self.assertTrue(
                channel.has_data
            )

    def test_process_yields(self):
        mct = self.setup_two_channel_mct()
        expected = iris.class_id.value_counts().values
        np.testing.assert_array_equal(mct.process_yields, expected)

    def test_nui_params(self):
        mct = self.setup_two_channel_mct()
        expected = np.zeros(len(self.channels)*len(self.processes)*self.bins)
        np.testing.assert_array_equal(mct.nui_params, expected)

    def test_generate_per_channel_parameters(self):
        mct = self.setup_two_channel_mct()
        yields = iris.class_id.value_counts().values
        nui_params = np.random.randn(len(self.channels)*len(self.processes)*self.bins)

        per_channel_params = mct.generate_per_channel_parameters(np.concatenate((yields, nui_params)))

        for y, n in zip(*per_channel_params):
            self.assertEqual(y.shape, (len(self.processes),))
            self.assertEqual(n.shape, (len(self.processes)*self.bins,))


class TestMultiChannelTemplate2d(unittest.TestCase):

    def setUp(self):

        self.bins = (3, 3)
        self.num_bins = reduce(lambda x,y: x*y, self.bins)
        self.range = ((3, 8), (2,5))
        self.variable = ("sepal_length", "sepal_width")

        self.processes = ("setosa", "versicolor", "virginica")
        self.channels = ["Test1", "Test2"]

        self.mct = MultiChannelTemplate()

        for channel_name in self.channels:
            self.mct.define_channel(
                name=channel_name, bins=self.bins, range=self.range, variable=self.variable
            )

        for process in self.processes:
           self.mct.define_process(process)

        ch1_setosa_length = channel_1.query("class_id==1")["sepal_length"].values
        ch1_setosa_width = channel_1.query("class_id==1")["sepal_width"].values
        ch1_versicolor_length = channel_1.query("(class_id==2)")["sepal_length"].values
        ch1_versicolor_width = channel_1.query("(class_id==2)")["sepal_width"].values
        ch1_virginica_length = channel_1.query("(class_id==3)")["sepal_length"].values
        ch1_virginica_width = channel_1.query("(class_id==3)")["sepal_width"].values

        ch1_hsetosa = Hist2d(self.bins, range=self.range,
                             data=(ch1_setosa_length, ch1_setosa_width))
        ch1_tsetosa = Template2d("setosa", "length", ch1_hsetosa)
        ch1_hversico = Hist2d(self.bins, range=self.range,
                              data=(ch1_versicolor_length, ch1_versicolor_width))
        ch1_tversico = Template2d("versicolor", "length", ch1_hversico)
        ch1_hvirgini = Hist2d(self.bins, range=self.range,
                              data=(ch1_virginica_length, ch1_virginica_width))
        ch1_tvirgini = Template2d("virginica", "length", ch1_hvirgini)

        self.mct.add_template("Test1", "setosa", ch1_tsetosa)
        self.mct.add_template("Test1", "versicolor", ch1_tversico)
        self.mct.add_template("Test1", "virginica", ch1_tvirgini)

        ch2_setosa_length = channel_2.query("class_id==1")["sepal_length"].values
        ch2_setosa_width = channel_2.query("class_id==1")["sepal_width"].values
        ch2_virginica_length = channel_2.query("(class_id==3)")["sepal_length"].values
        ch2_virginica_width = channel_2.query("(class_id==3)")["sepal_width"].values

        ch2_hsetosa = Hist2d(self.bins, range=self.range,
                             data=(ch2_setosa_length, ch2_setosa_width))
        ch2_tsetosa = Template2d("setosa", "length", ch2_hsetosa)
        ch2_hvirgini = Hist2d(self.bins, range=self.range,
                              data=(ch2_virginica_length, ch2_virginica_width))
        ch2_tvirgini = Template2d("versicolor", "length", ch2_hvirgini)

        self.mct.add_template("Test2", "setosa", ch2_tsetosa)
        self.mct.add_template("Test2", "virginica", ch2_tvirgini)

    def test_generate_per_channel_parameters(self):
        ch1_yields = channel_1.class_id.value_counts().values
        ch2_yields = channel_2.class_id.value_counts().values
        ch2_yields[1] = 0

        yields = ch1_yields + ch2_yields

        nui_params = np.random.randn(len(self.channels)*len(self.processes)*self.num_bins - self.num_bins)

        per_channel_yields, per_channel_nui_params = self.mct.generate_per_channel_parameters(np.concatenate((yields, nui_params)))

        self.assertEqual(per_channel_yields[0].shape, (3,))
        self.assertEqual(per_channel_yields[1].shape, (2,))
        self.assertEqual(per_channel_nui_params[0].shape,
                         (len(self.processes)*self.num_bins,))

        self.assertEqual(per_channel_nui_params[1].shape,
                         (2*self.num_bins,))

    def test_create_nll(self):
        nll = self.mct.create_nll()
        self.assertTrue(isinstance(nll, NegLogLikelihood))







