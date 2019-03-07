"""Defines the interface for all histogram classes.
"""
import numpy as np
from abc import ABC, abstractmethod

__all__ = [
    "AbstractHist"
]


class AbstractHist(ABC):
    """Abstract histogram class. Used as base class by all histogram
    implementations.
    """

    def __init__(self):
        self._bin_counts = None
        self._bin_errors_sq = None
        self._bin_edges = None
        self._num_bins = None

    @abstractmethod
    def fill(self, data, weights):
        pass

    @classmethod
    @abstractmethod
    def from_binned_data(cls, bin_counts, bin_edges, bin_errors=None):
        pass

    @property
    def bin_counts(self):
        return self._bin_counts

    @property
    def bin_errors(self):
        return np.sqrt(self.bin_errors_sq)

    @property
    def num_bins(self):
        return self._num_bins

    @property
    def bin_errors_sq(self):
        return self._bin_errors_sq

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    @abstractmethod
    def bin_mids(self):
        pass

    @property
    @abstractmethod
    def bin_widths(self):
        pass


