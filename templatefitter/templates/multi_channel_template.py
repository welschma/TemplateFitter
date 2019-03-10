import logging

from collections import OrderedDict

from templatefitter.histograms import Hist1d, Hist2d
from templatefitter.templates import Channel, Template1d, Template2d

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "MultiChannelTemplate"
]


class MultiChannelTemplate:

    def __init__(self):

        self._channel_dict = OrderedDict()
        self._processes = set()

    def define_channel(self, name, variable, bins, range):
        """Creates and stores `Channel` instances in the internal
        channel map.

        name : str
            Channel name.
        """
        ch = Channel(name=name, variable=variable, bins=bins, range=range)

        if isinstance(bins, int):
            dim = 1
        else:
            dim = len(ch.bins)

        for process in self.processes:
            hist = self.hist_factory(dim)(bins, range)
            template = self.template_factory(dim)(process, variable, hist)
            ch.add_template(process, template)

        self._channel_dict[name] = ch

    def define_process(self, name):
        """Defines a process """

        self._processes.add(name)

        for channel in self._channel_dict.values():
            dim = len(channel.bins)
            hist = self.hist_factory(dim)(channel.bins, channel.range)
            template = self.template_factory(dim)(name, channel.variable, hist)
            channel.add_template(name, template)

    def add_template(self, channel, process, template, efficiency=1.0):
        if channel not in list(self.channels.keys()):
            raise RuntimeError(f"Channel '{channel}' not defined!")
        if process not in self.processes:
            raise RuntimeError(f"Process '{process}' not defined!")
        
        self.channels[channel].add_template(process, template, efficiency)

    def add_data(self, channel, process, hdata):
        if channel not in list(self.channels.keys()):
            raise RuntimeError(f"Channel '{channel}' not defined!")
        if process not in self.processes:
            raise RuntimeError(f"Process '{process}' not defined!")

        self.channels[channel].add_data(hdata)
    @property
    def processes(self):
        """list of str: Names of defined processes."""
        return self._processes

    @property
    def num_processes(self):
        """int: Number of defined processes."""
        return len(self._processes)

    @property
    def num_channels(self):
        """int: Number of defined channels."""
        return len(self._channel_dict)

    @property
    def channels(self):
        """dict: Channel dictionary."""
        return self._channel_dict

    @staticmethod
    def hist_factory(dim):

        hists = {
            1: Hist1d,
            2: Hist2d,
        }

        if dim not in list(hists.keys()):
            raise RuntimeError("Only histograms for 1 and 2 dimensions "
                               "are available.")

        return hists[dim]

    @staticmethod
    def template_factory(dim):

        templates = {
            1: Template1d,
            2: Template2d,
        }

        if dim not in list(templates.keys()):
            raise RuntimeError("Only histograms for 1 and 2 dimensions "
                               "are available.")

        return templates[dim]


class NegLogLikelihood:

    def __init__(self, multi_channel_template):
        self._mcl = multi_channel_template

    def __call__(self, x):
        pass

