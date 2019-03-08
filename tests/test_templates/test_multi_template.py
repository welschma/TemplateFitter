import unittest

import numpy as np

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
        self.range = (2, 7)

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
        for process, template in zip(self.processes, self.templates):
            self.channel.add_template(process, template, 1.)

    def test_set_process_indices(self):

        test_processes = ["test1", "setosa", "virginica", "test2", "test3", "versicolor"]
        self.channel.set_process_indices(test_processes)
        expected = [test_processes.index(process) for process in self.processes]
        self.assertListEqual(self.channel._process_indices, expected)



