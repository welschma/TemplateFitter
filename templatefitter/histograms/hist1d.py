import numpy as np

from scipy.stats import binned_statistic

from templatefitter.histograms import AbstractHist

__all__ = [
    "Hist1d"
]


class Hist1d(AbstractHist):
    """A 1 dimensional histogram.
    """
    def __init__(self, bins, data=None, weights=None, range=None):
        super().__init__()

        self._init_bin_edges(bins, range)
        self._bin_counts = np.zeros(self.num_bins)
        self._bin_errors_sq = np.zeros(self.num_bins)

        if data is not None:
            self.fill(data, weights)


    def _init_bin_edges(self, bins, range):
        if isinstance(bins, int):
            self._num_bins = bins
            self._bin_edges = np.linspace(*range, self._num_bins + 1)
            self._range = range
        else:
            self._num_bins = len(bins) - 1
            self._bin_edges = bins
            self._range = (np.amin(self._bin_edges), np.amax(self._bin_edges))

    def _set_bin_counts(self, sample, bins, weights, range):

        if weights is None:
            weights = np.ones(len(sample))

        self._bin_counts, self._bin_edges, _ = binned_statistic(
            sample, weights, statistic='sum', bins=bins, range=range
        )

        self._bin_errors_sq, _, _ = binned_statistic(
            sample, weights**2, statistic='sum', bins=bins, range=range
        )

    def fill(self, data, weights):
        pass

    @classmethod
    def from_binned_data(cls, bin_counts, bin_edges):
        pass

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
        return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

    @property
    def bin_widths(self):
        return self.bin_edges[1:] - self.bin_edges[:-1]

    @property
    def num_bins(self):
        """int: Number of bins."""
        return self._num_bins
