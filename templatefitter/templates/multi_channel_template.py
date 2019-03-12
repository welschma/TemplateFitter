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

    def define_channel(self, name, bins, range):
        """Creates and stores `Channel` instances in the internal
        channel map.

        name : str
            Channel name.
        """
        ch = Channel(name=name, bins=bins, range=range)
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

    def set_yield(self, process_id, value):
        if process_id not in self.processes:
            raise RuntimeError(f"Process '{process}' not defined!")

        for channel in self.channels.values():
            try:
                channel[process_id].yield_param = value*channel.efficiencies[process_id]
            except KeyError:
                continue

    def reset_parameters(self):
        for channel in self.channels.values():
            channel.reset()

    def generate_per_channel_parameters(self, x):
        yields = x[:self.num_processes]
        nui_params = x[self.num_processes: self.num_nui_params+self.num_processes]

        per_channel_yields = [yields[channel.process_indices(self.processes)]
                              for channel in self.channels.values()]

        per_channel_num_nui_params = [channel.num_nui_params for channel in self.channels.values()]
        per_channel_nui_params = list(array_split_into(nui_params, per_channel_num_nui_params))

        return per_channel_yields, per_channel_nui_params

    def update_parameters(self, new_values):

        per_ch_yields, per_ch_nui_params = self.generate_per_channel_parameters(new_values)

        for channel, ch_yields, ch_nui_params in zip(self.channels.values(), per_ch_yields, per_ch_nui_params):
            channel.update_parameter(ch_yields, ch_nui_params)

    def create_nll(self):

        return NegLogLikelihood(self)


class AbstractTemplateCostFunction(ABC):
    """Abstract base class for all cost function to estimate
    yields using the template method.
    """

    def __init__(self):
        pass
    # -- abstract properties

    @property
    @abstractmethod
    def x0(self):
        """numpy.ndarray: Starting values for the minimization."""
        pass

    @property
    @abstractmethod
    def param_names(self):
        """list of str: Parameter names. Used for convenience."""
        pass

    # -- abstract methods --

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass


class NegLogLikelihood(AbstractTemplateCostFunction):

    def __init__(self, multi_channel_template: MultiChannelTemplate):
        super().__init__()
        self._mct = multi_channel_template

    @property
    def x0(self):
        """

        :return:
        """
        yields = self._mct.process_yields
        nui_params = self._mct.nui_params

        return np.concatenate((yields, nui_params))

    @property
    def param_names(self):
        yield_names = [process + "_yield" for process in self._mct.processes]

        nui_param_names = []

        for ch_name, channel in self._mct.channels.items():
            per_channel_names = []
            for temp_name, template in channel.templates.items():
                per_channel_names.extend([
                    f"{ch_name}_{temp_name}_bin_{i}" for i in range(template.num_bins)
                ])
            nui_param_names.extend(per_channel_names)

        return yield_names + nui_param_names

    def __call__(self, x):

        ch_yields, ch_nui_params = self._mct.generate_per_channel_parameters(x)

        nll_value = 0

        for channel, yields, nui_params in zip(self._mct.channels.values(), ch_yields, ch_nui_params):

            nll_value += channel.nll_contribution(yields, nui_params)

        return nll_value


