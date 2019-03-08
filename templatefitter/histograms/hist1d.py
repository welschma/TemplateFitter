import numpy as np

from scipy.stats import binned_statistic

from templatefitter.histograms import AbstractHist
from templatefitter.histograms import bin_mids, bin_widths, get_bin_range

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
        self._shape = self._bin_counts.shape

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
            self._range = get_bin_range(self._bin_edges)

    def fill(self, data, weights):
        """Fills the histogram with given data and weights.

        Parameters
        ----------
        data : numpy.ndarray, pandas.Series
            Sequence which is filled into the histogram.
        weights : numpy.ndarray, pandas.Series, optional
            Sequence which is used as event weights. If no weights are
            given, each event is weighted with 1.0. Default is None.
        """
        if weights is None:
            weights = np.ones_like(data)
        if isinstance(weights, list):
            weights = np.array(weights)

        self._bin_counts += binned_statistic(
            data,
            weights,
            statistic='sum',
            bins=self._bin_edges,
            range=self._range
        )[0]

        self._bin_errors_sq += binned_statistic(
            data,
            weights ** 2,
            statistic='sum',
            bins=self._bin_edges,
            range=self._range
        )[0]

    @classmethod
    def from_binned_data(cls, bin_counts, bin_edges, bin_errors=None):
        """Creates a `Hist1d` from a binned dataset.

        Parameters
        ----------
        bin_edges : numpy.ndarray
            Array with of bin edges.
        bin_counts : numpy.ndarray
            Array of bin counts.
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

        instance._bin_errors_sq = bin_errors**2

        return instance


    @property
    def bin_mids(self):
        return bin_mids(self.bin_edges)

    @property
    def bin_widths(self):
        return bin_widths(self.bin_edges)

