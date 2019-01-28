"""This module provides several classes which help to implement templates
used for binned likelihood fits where the expected number of events is
estimated from different histograms.
"""
import logging

from abc import ABC, abstractmethod

import numpy as np

from templatefitter.histogram import Hist1d

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AbstractTemplate",
    "Template",
    "StackedTemplate",
    "SimultaneousTemplate"
]


class AbstractTemplate(ABC):

    def __init__(self, id):
        self._id = id
        self._num_bins = None

    @property
    def id(self):
        """str: Template identifier."""
        return self._id

    @property
    def num_bins(self):
        return self._num_bins

    # -- abstract methods

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def fractions(self):
        pass

    @abstractmethod
    def plot_on(self):
        pass


class Template(AbstractTemplate):

    def __init__(self, id, variable, num_bins, limits, df, weight="weight"):
        super().__init__(id)

        self._variable = variable
        self._num_bins = num_bins
        self._limits = limits

        self._hist = Hist1d(num_bins, limits,
                            df[variable].values, df[weight].values)

        #TODO decide wether to add this to the histogram itself or not
        # statistical covariance matrix as diagonal matrix of the
        # sum of weights squared per bin
        # set errors for empty bins to 1e-7 and set the correlation
        # matrix by hand for the statistical error as 100% uncorrelated.
        # the total covariance matrix is the sum of the statistical
        # covariance and any additional covariance matrix that has been
        # added
        stat_mc_errors = np.copy(self._hist.bin_errors_sq)
        stat_mc_errors[stat_mc_errors == 0] = 1e-14




        self._param_yield = np.sum(self._hist.bin_counts)
        self._param_nui = np.zeros(self._hist.num_bins)

    def values(self):
        pass

    def fractions(self, nui_params):
        pass

    def plot_on(self):
        pass


class StackedTemplate(AbstractTemplate):
    pass


class SimultaneousTemplate(AbstractTemplate):
    pass
