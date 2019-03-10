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
        self._shape = None
        self._range = None

        self._is_empty = True

    @property
    def bin_counts(self):
        return self._bin_counts

    @property
    def bin_errors(self):
        return np.sqrt(self.bin_errors_sq)

    @property
    def bin_errors_sq(self):
        return self._bin_errors_sq

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def num_bins(self):
        return self._num_bins

    @property
    def shape(self):
        return self._shape

    @property
    def range(self):
        return self._range

    @property
    def is_empty(self):
        return self._is_empty

    @abstractmethod
    def fill(self, data, weights):
        pass

    @classmethod
    @abstractmethod
    def from_binned_data(cls, bin_counts, bin_edges, bin_errors=None):
        pass

    @property
    @abstractmethod
    def bin_mids(self):
        pass

    @property
    @abstractmethod
    def bin_widths(self):
        pass



