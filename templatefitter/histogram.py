import numpy as np
import scipy.stats


class HistogramError(Exception):
    pass


# TODO deal with underflow and overflow bins
# TODO deal with mathematical operation on histograms like scaling
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

    Attributes
    ----------
    nbins : int
        Number of bins in the histogram.
    bin_edges :  np.ndarray
        Bin edges of the histogram. Shape (nbins + 1,).
    bin_width : float
        Bin width of the histogram. 
    bin_mids : np.ndarray
        Bin mids of the histogram.
    bin_counts : np.ndarray
        Current bin counts in each bin.
    bin_entries : np.ndarray
        Current bin entries in each bin.
    bin_errors : np.ndarray
        Current bin errors in each bin.
    limits : tuple of float
        Lower and upper limit of the histogram range.
    lower_limit : float
        Lower  limit of the histogram range.
    upper_limit : float
        Upper  limit of the histogram range.
    """

    def __init__(self, nbins, limits, data=None, weights=None):
        self._nbins = nbins
        self._limits = limits
        self._bin_edges = np.linspace(
            self.lower_limit, self.upper_limit, nbins + 1
        )
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
            raise HistogramError(
                "Shape of data array does not match weight array."
            )
        self._bin_counts += scipy.stats.binned_statistic(
            x=data,
            values=weights,
            statistic="sum",
            bins=self._bin_edges
        )[0]

        self._bin_entries += scipy.stats.binned_statistic(
            x=data,
            values=weights,
            statistic="count",
            bins=self._bin_edges
        )[0]

        self._bin_errors_sq += scipy.stats.binned_statistic(
            x=data,
            values=weights ** 2,
            statistic="sum",
            bins=self._bin_edges
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
        return self._nbins

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_width(self):
        return self.bin_edges[1] - self.bin_edges[0]

    @property
    def bin_mids(self):
        edges = self.bin_edges
        return (edges[:-1] + edges[1:]) / 2.

    @property
    def bin_counts(self):
        return self._bin_counts

    @property
    def bin_entries(self):
        return self._bin_entries

    @property
    def bin_errors(self):
        return np.sqrt(self._bin_errors_sq)

    @property
    def bin_errors_sq(self):
        return self._bin_errors_sq

    @property
    def limits(self):
        return self._limits

    @property
    def lower_limit(self):
        return self._limits[0]

    @property
    def upper_limit(self):
        return self._limits[1]

    def __str__(self):
        return (f"Bin Edges: {self.bin_edges}"
                + f"\nBin Counts: {self.bin_counts}"
                + f"\nBin Errors: {self.bin_errors}")
