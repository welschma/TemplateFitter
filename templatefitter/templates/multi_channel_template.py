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

        self._rate_uncertainties = dict()
        self._num_rate_uncertainties = 0
        self._rate_uncertainties_nui_params = dict()
        self._per_process_num_rate_unc = dict()

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
    def rate_uncertainties_dict(self):
        return self._rate_uncertainties

    @property
    def rate_uncertainties(self):
        rate_uncertainties_per_process = list()
        for process, uncertainties in self._rate_uncertainties.items():
            rate_uncertainties_per_process.append(uncertainties)
        return np.concatenate(rate_uncertainties_per_process)

    @property
    def rate_uncertainties_nui_params(self):
        rate_nui_params_per_process = list()
        for process, uncertainties in self._rate_uncertainties_nui_params.items():
            rate_nui_params_per_process.append(uncertainties)
        return np.concatenate(rate_nui_params_per_process)

    @rate_uncertainties_nui_params.setter
    def rate_uncertainties_nui_params(self, new_values):
        new_values_per_process =  np.split(
            new_values, np.cumsum(self.num_rate_uncertainties_per_process)[1:]
        )
        for process, values in zip(self._rate_uncertainties_nui_params.keys(), new_values_per_process):
            self._rate_uncertainties_nui_params[process] = values

    @property
    def num_rate_uncertainties(self):
        return self._num_rate_uncertainties

    @property
    def num_rate_uncertainties_per_process(self):
        return np.array(list(self._per_process_num_rate_unc.values()))

    @property
    def num_channels(self):
        """int: Number of defined channels."""
        return len(self._channel_dict)

    @property
    def channels(self):
        """dict: Channel dictionary."""
        return self._channel_dict

    @property
    def num_processes(self):
        """int: Number of defined processes."""
        return len(self._processes)

    @property
    def processes(self):
        """tuple of str: Names of defined processes."""
        return self._processes

    def define_channel(self, name, bins, range):
        """Creates and stores `Channel` instances in the internal
        channel map.

        name : str
            Channel name.
        """
        ch = Channel(name=name, bins=bins, range=range)
        self._channel_dict[name] = ch

    def define_process(self, name):
        """

        Parameters
        ----------
        name

        Returns
        -------

        """
        if name not in self._processes:
            self._processes = (*self._processes, name)
            self._rate_uncertainties[name] = np.array([])
            self._rate_uncertainties_nui_params[name] = np.array([])
            self._per_process_num_rate_unc[name] = 0
        else:
            raise RuntimeError(f"Process {name} already defined.")

    def add_rate_uncertainty(self, process, rel_uncertainty):
        if process not in self._processes:
            raise RuntimeError(f"Process {process} not defined.")
        else:
            self._rate_uncertainties[process] = np.array(
                [*self._rate_uncertainties[process], rel_uncertainty]
            )
            self._rate_uncertainties_nui_params[process] = np.array(
                [*self._rate_uncertainties_nui_params[process], 0]
            )
            self._num_rate_uncertainties += 1
            self._per_process_num_rate_unc[process] += 1

    def add_template(self, channel, process, template, efficiency=1.0):
        """

        Parameters
        ----------
        channel
        process
        template
        efficiency

        Returns
        -------

        """
        if channel not in list(self.channels.keys()):
            raise RuntimeError(f"Channel '{channel}' not defined!")
        if process not in self.processes:
            raise RuntimeError(f"Process '{process}' not defined!")

        self.channels[channel].add_template(process, template, efficiency)

    def add_data(self, **kwargs):
        """

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        for channel, hdata in kwargs.items():
            if channel not in list(self.channels.keys()):
                raise RuntimeError(f"Channel '{channel}' not defined!")

            self.channels[channel].add_data(hdata)

    def process_to_index(self, process):
        """int: Index of the process in the internal tuple."""
        return self._processes.index(process)

    def set_yield(self, process_id, value):
        """

        Parameters
        ----------
        process_id
        value

        Returns
        -------

        """
        if process_id not in self.processes:
            raise RuntimeError(f"Process '{process}' not defined!")

        for channel in self.channels.values():
            try:
                channel[process_id].yield_param = value*channel.efficiencies[process_id]
            except KeyError:
                continue

    def get_yield(self, process_id):
        """

        Parameters
        ----------
        process_id
        value

        Returns
        -------

        """
        if process_id not in self.processes:
            raise RuntimeError(f"Process '{process_id}' not defined!")

        return self.process_yields[self.process_to_index(process_id)]

    def reset_parameters(self):
        """

        Returns
        -------

        """
        for channel in self.channels.values():
            channel.reset()

    def multiplicative_rate_uncertainty(self, nui_params):
        epsilon = 1 + nui_params*self.rate_uncertainties
        per_yield_epsilon = np.split(epsilon, np.cumsum(self.num_rate_uncertainties_per_process)[1:])
        return np.array([np.prod(eps) for eps in per_yield_epsilon])

    def generate_per_channel_parameters(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        yields = x[:self.num_processes]
        nui_params = x[self.num_processes: self.num_nui_params+self.num_processes]
        per_yield_rate_unc = self.multiplicative_rate_uncertainty(
            x[self.num_processes+self.num_nui_params:self.num_processes+self.num_nui_params+self.num_rate_uncertainties]
        )
        yields *= per_yield_rate_unc
        per_channel_yields = [yields[channel.process_indices(self.processes)]
                              for channel in self.channels.values()]

        per_channel_num_nui_params = [channel.num_nui_params for channel in self.channels.values()]
        per_channel_nui_params = np.split(nui_params, np.cumsum(per_channel_num_nui_params)[:-1])
        # per_channel_nui_params = list(array_split_into(nui_params, per_channel_num_nui_params))

        return per_channel_yields, per_channel_nui_params

    def update_parameters(self, new_values):
        """

        Parameters
        ----------
        new_values

        Returns
        -------

        """
        self.rate_uncertainties_nui_params = new_values[-self.num_rate_uncertainties:]
        per_ch_yields, per_ch_nui_params = self.generate_per_channel_parameters(new_values)

        for channel, ch_yields, ch_nui_params in zip(self.channels.values(), per_ch_yields, per_ch_nui_params):
            channel.update_parameters(ch_yields, ch_nui_params)

    def create_nll(self):
        """

        Returns
        -------

        """
        return NegLogLikelihood(self)

    def generate_toy_dataset(self):
        """Generates a toy dataset from the given templates.
        This is a binned dataset where each bin is treated a
        random number following a poisson distribution with
        mean equal to the bin content of all templates.

        Returns
        -------
        toy_datasets : Dictionary of instances of a histogram class that
        inherits from `AbstractHist` with keys equal to the channel names.
        """
        toy_datasets = {}
        for ch_name, channel in self.channels.items():
            toy_datasets[ch_name] = channel.generate_toy_dataset()
        return toy_datasets

    def generate_asimov_dataset(self, integer_values=False):
        """Generates an Asimov dataset from the given templates.
        This is a binned dataset which corresponds to the current
        expectation values. Since data takes only integer values,
        the template expectation in each bin is rounded to the
        nearest integer.

        Parameters
        ----------
        integer_values : bool, optional
            Whether to round Asimov data points to integer values
            or not. Default is False.

        Returns
        -------
        asimov_datasets : Dictionary of instances of a histogram class that
        inherits from `AbstractHist` with keys equal to the channel names.
        """
        asimov_datasets = {}
        for ch_name, channel in self.channels.items():
            asimov_datasets[ch_name] = channel.generate_asimov_dataset(integer_values)
        return asimov_datasets


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
        yields = self._mct.process_yields
        nui_params = self._mct.nui_params
        rate_uncertainties = self._mct.rate_uncertainties_nui_params

        return np.concatenate((yields, nui_params, rate_uncertainties))

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

        rate_uncertainties_names = []

        for process, uncertainties in self._mct.rate_uncertainties_dict.items():
            rate_uncertainties_names.extend(
                [f"rate_{process}_{i}" for i in range(len(uncertainties))]
            )

        return yield_names + nui_param_names + rate_uncertainties_names

    def __call__(self, x):
        ch_yields, ch_nui_params = self._mct.generate_per_channel_parameters(x)
        nll_value = 0

        for channel, yields, nui_params in zip(self._mct.channels.values(), ch_yields, ch_nui_params):

            nll_value += channel.nll_contribution(yields, nui_params)

        # constrain rate uncertainties

        nll_value += 0.5 * np.sum(
            x[self._mct.num_processes+self._mct.num_nui_params:self._mct.num_processes+self._mct.num_nui_params+self._mct.num_rate_uncertainties]**2
        )

        return nll_value


