"""Histogram module.
"""
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import scipy.stats

__all__ = ["Histogram",
           "AbstractHist",
           "Hist1d",
           "Hist2d"]


class Histogram:
    """Histogram container for data. Bins are of equal width.

    Parameters
    ----------
    nbins : int
        Number of bins in the histogram.
    limits : Tuple[float, float]
        Lower and upper limit of the range to histogram.
    data : np.array, optional
        Data to be filled in the histogram.
    weights: np.array, optional
        Weights for each entry in the histogram. If none are
        given, a weight of 1. will be assigned to each event.
    """

    def __init__(self, nbins, limits, data=None, weights=None):
        self._nbins = nbins
        self._limits = limits
        self._bin_edges = np.linspace(
            self.lower_limit, self.upper_limit, nbins + 1)
        self._bin_counts = np.zeros(nbins)
        self._bin_entries = np.zeros(nbins)
        self._bin_errors_sq = np.zeros(nbins)

        if data is not None:
            self.fill(data, weights)

    def fill(self, data, weights=None):
        """Fills the histogram with given data. If no weights are
        given, each data point is weighted with 1.0.

        Parameters
        ----------
        data : np.array
            Data to be filled in the histogram.
        weights: np.array, optional
            Weights for each entry in the histogram.
        """
        if isinstance(data, list):
            data = np.array(data)
        if weights is not None and isinstance(weights, list):
            weights = np.array(weights)

        if weights is None:
            weights = np.ones_like(data)

        if len(data) != len(weights):
            raise ValueError(
                "Shape of data array does not match weight array.")
        self._bin_counts += scipy.stats.binned_statistic(
            x=data, values=weights, statistic="sum", bins=self._bin_edges
        )[0]

        self._bin_entries += scipy.stats.binned_statistic(
            x=data, values=weights, statistic="count", bins=self._bin_edges
        )[0]

        self._bin_errors_sq += scipy.stats.binned_statistic(
            x=data, values=weights ** 2, statistic="sum", bins=self._bin_edges
        )[0]

    def scale(self, c):
        """Multiplies the histogram by the constant c.
        This means that the bin_contents are set to c*bin_contents.
        The bin_errors_sq are recalculated to c**2*bin_errors_sq.

        Arguments
        ---------
        c : float
            Multiplicative constant value.
        """
        self._bin_counts *= c
        self._bin_errors_sq *= c ** 2

    @property
    def nbins(self):
        """int: Number of bins in the histogram."""
        return self._nbins

    @property
    def bin_edges(self):
        """np.ndarray: Bin edges of the histogram. Shape (nbins + 1,)."""
        return self._bin_edges

    @property
    def bin_width(self):
        """float: Bin width of the histogram."""
        return self.bin_edges[1] - self.bin_edges[0]

    @property
    def bin_mids(self):
        """np.ndarray: Bin mids of the histogram."""
        edges = self.bin_edges
        return (edges[:-1] + edges[1:]) / 2.0

    @property
    def bin_counts(self):
        """np.ndarray: Current bin counts in each bin."""
        return self._bin_counts

    @bin_counts.setter
    def bin_counts(self, new_values):
        self._bin_counts = new_values

    @property
    def bin_entries(self):
        """ np.ndarray: Current bin entries in each bin."""
        return self._bin_entries

    @property
    def bin_errors(self):
        """np.ndarray. Current bin errors in each bin."""
        return np.sqrt(self._bin_errors_sq)

    @bin_errors.setter
    def bin_errors(self, new_errors):
        self._bin_errors_sq = new_errors ** 2

    @property
    def bin_rel_errors(self):
        """np.ndarray. Current relative bin errors in each bin."""
        rel_errors = np.sqrt(self._bin_errors_sq) / self.bin_counts
        rel_errors[rel_errors == 0] = 1e-7
        return rel_errors

    @property
    def bin_errors_sq(self):
        """np.ndarray. Current bin errors squared in each bin."""
        return self._bin_errors_sq

    @property
    def limits(self):
        """tuple of float: Lower and upper limit of the histogram range."""
        return self._limits

    @property
    def lower_limit(self):
        """float: Lower  limit of the histogram range."""
        return self._limits[0]

    @property
    def upper_limit(self):
        """float: Upper  limit of the histogram range."""
        return self._limits[1]

    def __str__(self):
        return (
                f"Bin Edges: {self.bin_edges}"
                + f"\nBin Counts: {self.bin_counts}"
                + f"\nBin Errors: {self.bin_errors}"
        )


class AbstractHist(ABC):
    """
    """
    def __init__(self):
        self._bin_edges = None
        self._num_bins = None
        self._bin_counts = None
        self._bin_errors_sq = None
        self._bin_mids = None

    @abstractmethod
    def fill(self, data, weights=None):
        pass

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_mids(self):
        return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

    @property
    def bin_counts(self):
        return self._bin_counts

    @property
    def bin_errors(self):
        return np.sqrt(self._bin_errors_sq)

    @property
    def bin_errors_sq(self):
        return self._bin_errors_sq

    @property
    def num_bins(self):
        return self._num_bins


class Hist1d(AbstractHist):
    """
    """

    def __init__(self, bins, limits=None, data=None, weights=None):
        super().__init__()
        if isinstance(bins, int):
            self._num_bins = bins
            self._bin_edges = np.linspace(*limits, self._num_bins + 1)
            self._limits = limits
        else:
            self._num_bins = len(bins) - 1
            self._bin_edges = bins
            self._limits = (np.amin(self._bin_edges), np.amax(self._bin_edges))

        self._bin_counts = np.zeros(self._num_bins)
        self._bin_errors_sq = np.zeros(self._num_bins)

        if data is not None:
            self.fill(data, weights)

    def fill(self, data, weights=None):
        if weights is None:
            weights = np.ones_like(data)
        if isinstance(weights, list):
            weights = np.array(weights)

        self._bin_counts += scipy.stats.binned_statistic(
            data,
            weights,
            statistic='sum',
            bins=self._bin_edges,
            range=self._limits
        )[0]

        self._bin_errors_sq += scipy.stats.binned_statistic(
            data,
            weights ** 2,
            statistic='sum',
            bins=self._bin_edges,
            range=self._limits
        )[0]


class Hist2d(AbstractHist):
    """
    """

    def __init__(self, bins, limits=None, data=None, weights=None):
        super().__init__()
        if all(isinstance(num_bins, int) for num_bins in bins):
            self._num_bins = bins
            self._bin_edges = [
                np.linspace(*limit, num_bins + 1) for limit, num_bins in zip(limits, bins)
            ]
            self._limits = limits

        else:
            self._num_bins = [len(edges) - 1 for edges in bins]
            self._bin_edges = bins
            self._limits = [np.array([np.amin(edges), np.amax(edges)]) for edges in bins]

        self._bin_counts = np.zeros(self._num_bins)
        self._bin_errors_sq = np.zeros(self._num_bins)

        if data is not None:
            self.fill(data, weights)

    def fill(self, data, weights=None):
        if weights is None:
            weights = np.ones(len(data[:, 0]))
        if isinstance(weights, list):
            weights = np.array(weights)

        self._bin_counts += scipy.stats.binned_statistic_2d(
            x=data[:, 0],
            y=data[:, 1],
            values=weights,
            statistic='sum',
            bins=self._bin_edges,
            range=np.array(self._limits)
        )[0]

        self._bin_errors_sq += scipy.stats.binned_statistic_2d(
            x=data[:, 0],
            y=data[:, 1],
            values=weights ** 2,
            statistic='sum',
            bins=self._bin_edges,
            range=np.array(self._limits)
        )[0]

        self.x_projection.cache_clear()
        self.y_projection.cache_clear()

    @property
    def xlimits(self):
        return self._limits[0]

    @property
    def ylimits(self):
        return self._limits[1]

    @property
    def x_edges(self):
        return self._bin_edges[0]

    @property
    def y_edges(self):
        return self._bin_edges[1]

    @lru_cache()
    def x_projection(self):
        return np.sum(self.bin_counts, axis=1), np.sqrt(np.sum(self._bin_errors_sq, axis=1))

    @lru_cache()
    def y_projection(self):
        return np.sum(self.bin_counts, axis=0), np.sqrt(np.sum(self._bin_errors_sq, axis=0))
