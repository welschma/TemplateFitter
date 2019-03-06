"""Defines the interface for all histogram classes.
"""

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
        self._bin_mids = None
        self._bin_widths = None
        self._num_bins = None

    @property
    @abstractmethod
    def bin_counts(self):
        pass

    @property
    @abstractmethod
    def bin_errors(self):
        pass

    @property
    @abstractmethod
    def bin_errors_sq(self):
        pass

    @property
    @abstractmethod
    def bin_edges(self):
        pass

    @property
    @abstractmethod
    def bin_mids(self):
        pass

    @property
    @abstractmethod
    def bin_widths(self):
        pass

    @property
    def num_bins(self):
        """int: Number of bins."""
        return self._num_bins

