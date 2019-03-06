import numpy as np

from scipy.stats import binned_statistic

from templatefitter.histograms import AbstractHist

__all__ = [
    "Hist1d"
]

class Hist1d(AbstractHist):
    """A 1 dimensional histogram.
    """
    def __init__(self, sample, bins, weights=None, range=None):
        super().__init__()

        self._set_bin_counts(sample, bins, weights, range)
        self._bin_widths = self._bin_edges[1:] - self._bin_edges[:-1]
        self._bin_mids = (self._bin_edges[1:] + self._bin_edges[:-1]) / 2
        self._num_bins = len(self._bin_edges)

    def _set_bin_counts(self, sample, bins, weights, range):

        if weights is None:
            weights = np.ones(len(sample))

        self._bin_counts, self._bin_edges, _ = binned_statistic(
            sample, weights, statistic='sum', bins=bins, range=range
        )

        self._bin_errors_sq, _, _ = binned_statistic(
            sample, weights**2, statistic='sum', bins=bins, range=range
        )

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
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_mids(self):
        return self._bin_mids

    @property
    def bin_widths(self):
        return self._bin_widths

    @property
    def num_bins(self):
        """int: Number of bins."""
        return self._num_bins
