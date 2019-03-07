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
        pass

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
    @abstractmethod
    def num_bins(self):
        pass

