import numpy as np
import scipy.stats

class HistogramError(Exception):
    pass


#TODO deal with underflow and overflow bins
#TODO deal with mathematical operation on histograms like scaling
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
    nbins
    bin_eges
    bin_width
    bin_mids
    bin_counts
    bin_entries
    bin_errors
    limits
    lower_limit
    upper_limit
    """

    def __init__(self, nbins, limits, data=None, weights=None):
        self._nbins = nbins
        self._limits = limits
        self._bin_edges = np.linspace(
            self.lower_limit, self.upper_limit, nbins+1
            )
        self._bin_counts = np.zeros(nbins)
        self._bin_entries= np.zeros(nbins)
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

        if len(data) != len(weights):
            raise HistogramError(
                "Shape of data array does not match weight array."
                )
        if weights is None:
            weights = np.ones_like(data)

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
            values=weights**2,
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

        Returns
        -------
        None
        """
        self._bin_counts *= c
        self._bin_errors_sq *= c**2

    
    @property
    def nbins(self):
        """Number of bins in the histogram.

        Returns
        -------
        int
        """
        return self._nbins

    @property
    def bin_edges(self):
        """Bin edges of the histogram.

        Returns
        -------
        np.ndarray
            Shape (nbins + 1,)
        """
        return self._bin_edges

    @property
    def bin_width(self):
        """Bin width of the histogram.

        Returns
        -------
        float
        """
        return self.bin_edges[1] - self.bin_edges[0]

    @property
    def bin_mids(self):
        """Bin mids of the histogram.

        Returns
        -------
        np.ndarray
            Shape (nbins,)
        """
        edges = self.bin_edges
        return (edges[:-1 + edges[1:]])/2.

    @property
    def bin_counts(self):
        """Current bin counts in each bin.

        Returns
        -------
        np.ndarray
            Shape (nbins,)
        """
        return self._bin_counts

    @property
    def bin_entries(self):
        """Current bin entries in each bin.

        Returns
        -------
        np.ndarray
            Shape (nbins,)
        """
        return self._bin_entries

    @property
    def bin_errors(self):
        """Current bin errors in each bin.

        Returns
        -------
        np.ndarray
            Shape (nbins,)
        """
        return np.sqrt(self._bin_errors_sq)

    @property
    def limits(self):
        """Lower and upper limit of the histogram range.

        Returns
        -------
        tuple of float
        """
        return self._limits

    @property
    def lower_limit(self):
        """Lower  limit of the histogram range.

        Returns
        -------
        float
        """
        return self._limits[0]

    @property
    def upper_limit(self):
        """Upper  limit of the histogram range.

        Returns
        -------
        float
        """
        return self._limits[1]

    def __str__(self):
        return (f"Bin Edges: {self.bin_edges}" 
        + f"\nBin Counts: {self.bin_counts}"
        + f"\nBin Errors: {self.bin_errors}")
