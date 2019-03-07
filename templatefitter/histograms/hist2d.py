from functools import lru_cache

import numpy as np

from scipy.stats import binned_statistic_2d

from templatefitter.histograms import AbstractHist, Hist1d
from templatefitter.histograms import bin_mids, bin_widths, get_bin_range

__all__ = ["Hist2d"]


class Hist2d(AbstractHist):
    """A 2 dimensional histogram.
    """

    def __init__(self, bins, range=None, data=None, weights=None):
        super().__init__()
        self._init_bin_edges(bins, range)
        self._bin_counts = np.zeros(self._num_bins)
        self._bin_errors_sq = np.zeros(self._num_bins)

        if data is not None:
            self.fill(data, weights)

    def _init_bin_edges(self, bins, range):
        if all(isinstance(num_bins, int) for num_bins in bins):
            self._num_bins = bins
            self._bin_edges = [
                np.linspace(*limit, num_bins + 1)
                for limit, num_bins in zip(range, bins)
            ]
            self._range = range

        else:
            self._num_bins = tuple(len(edges) - 1 for edges in bins)
            print(self._num_bins)
            self._bin_edges = bins
            self._range = tuple(map(get_bin_range, bins))

    def fill(self, data, weights=None):
        if weights is None:
            weights = np.ones(len(data[0]))
        if isinstance(weights, list):
            weights = np.array(weights)

        self._bin_counts += binned_statistic_2d(
            x=data[0], y=data[1], values=weights, statistic="sum", bins=self._bin_edges
        )[0]

        self._bin_errors_sq += binned_statistic_2d(
            x=data[0],
            y=data[1],
            values=weights ** 2,
            statistic="sum",
            bins=self._bin_edges,
        )[0]

        self.x_projection.cache_clear()
        self.y_projection.cache_clear()

    @classmethod
    def from_binned_data(cls, bin_counts, bin_edges, bin_errors=None):
        """Creates a `Hist2d` from a binned dataset.

        Parameters
        ----------
        bin_counts : numpy.ndarray
            Array of bin counts.
        bin_edges : tuple of numpy.ndarray
            Array with of bin edges.
        bin_errors : numpy.ndarray
            Array of bin errors.

        Returns
        -------
        histogram : Hist1d
        """
        instance = cls(bin_edges)
        instance._bin_counts = bin_counts

        if bin_errors is None:
            bin_errors = np.sqrt(bin_counts)

        instance._bin_errors_sq = bin_errors ** 2

        return instance

    @property
    def bin_mids(self):
        return list(map(bin_mids, self.bin_edges))

    @property
    def bin_widths(self):
        return list(map(bin_widths, self.bin_edges))

    @property
    def xrange(self):
        return self._range[0]

    @property
    def yrange(self):
        return self._range[1]

    @property
    def x_edges(self):
        return self._bin_edges[0]

    @property
    def y_edges(self):
        return self._bin_edges[1]

    @lru_cache()
    def x_projection(self):
        return Hist1d.from_binned_data(
            np.sum(self.bin_counts, axis=1),
            self.x_edges,
            np.sqrt(np.sum(self._bin_errors_sq, axis=1))
        )

    @lru_cache()
    def y_projection(self):
        return Hist1d.from_binned_data(
            np.sum(self.bin_counts, axis=0),
            self.y_edges,
            np.sqrt(np.sum(self._bin_errors_sq, axis=0))
        )
