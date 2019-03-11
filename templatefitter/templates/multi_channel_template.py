import logging

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from templatefitter.histograms import Hist1d, Hist2d
from templatefitter.templates import Channel, Template1d, Template2d
from templatefitter.utility import array_split_into

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "MultiChannelTemplate",
    "NegLogLikelihood"
]


class MultiChannelTemplate:

    def __init__(self):

        self._channel_dict = OrderedDict()
        self._processes = tuple()

    def define_channel(self, name, variable, bins, range):
        """Creates and stores `Channel` instances in the internal
        channel map.

        name : str
            Channel name.
        """
        ch = Channel(name=name, variable=variable, bins=bins, range=range)
        self._channel_dict[name] = ch

    def define_process(self, name):
        """Defines a process"""
        if name not in self._processes:
            self._processes = (*self._processes, name)
        else:
            raise RuntimeError(f"Process {name} already defined.")

    def add_template(self, channel, process, template, efficiency=1.0):
        if channel not in list(self.channels.keys()):
            raise RuntimeError(f"Channel '{channel}' not defined!")
        if process not in self.processes:
            raise RuntimeError(f"Process '{process}' not defined!")
        
        self.channels[channel].add_template(process, template, efficiency)

    def add_data(self, **kwargs):

        for channel, hdata in kwargs.items():
            if channel not in list(self.channels.keys()):
                raise RuntimeError(f"Channel '{channel}' not defined!")

            self.channels[channel].add_data(hdata)

    @property
    def processes(self):
        """tuple of str: Names of defined processes."""
        return self._processes

    def process_to_index(self, process):
        """int: Index of the process in the internal tuple."""
        return self._processes.index(process)

    @property
    def num_processes(self):
        """int: Number of defined processes."""
        return len(self._processes)

    @property
    def num_nui_params(self):
        """int: Total number of nuissance parameters. """
        return sum([channel.num_nui_params for channel in self.channels.values()])

    @property
    def process_yields(self):
        yields = np.zeros(self.num_processes)

        for process in self.processes:
            for channel in self.channels.values():
                try:
                    yields[self.process_to_index(process)] += channel[process].yield_param
                except KeyError:
                    continue

        return yields

    @property
    def nui_params(self):
        nui_params_per_channel = list()

        for channel in self.channels.values():
            for template in channel.templates.values():
                nui_params_per_channel.append(template.nui_params)

        return np.concatenate(nui_params_per_channel)

    @property
    def num_channels(self):
        """int: Number of defined channels."""
        return len(self._channel_dict)

    @property
    def channels(self):
        """dict: Channel dictionary."""
        return self._channel_dict

    def generate_per_channel_parameters(self, x):
        yields = x[:self.num_processes]
        nui_params = x[self.num_processes: self.num_nui_params+self.num_processes]

        per_channel_yields = [yields[channel.process_indices(self.processes)]
                              for channel in self.channels.values()]

class NegLogLikelihood:

    def __init__(self, multi_channel_template):
        self._mcl = multi_channel_template

    def __call__(self, x):
        pass

