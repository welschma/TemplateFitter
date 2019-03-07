import logging

import numpy as np

from templatefitter.histograms import Hist1d
from templatefitter.templates import AbstractTemplate
from templatefitter.utility import cov2corr, get_systematic_cov_mat

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["Template1d"]


class Template1d(AbstractTemplate):
    
    def __init__(self, name, num_bins, range, data=None, weights=None,
                 color=None, pretty_label=None):
        super(Template1d, self).__init__(name=name)

        self._hist = Hist1d(
            bins=num_bins, range=range, data=data, weights=weights
        )
        self._range = range

        self._cov_mats = list()
        self._cov = None
        self._corr = None
        self._inv_corr = None
        self._relative_errors = None

        self._init_params()
        self._init_errors()

        self.color = color
        self.pretty_label = pretty_label

    def _init_errors(self):
        """The statistical covariance matrix is initialized as diagonal
        matrix of the sum of weights squared per bin in the underlying
        histogram. For empty bins, the error is set to 1e-7. The errors
        are initialized to be 100% uncorrelated. The relative errors per
        bin are set to 1e-7 in case of empty bins.
        """
        stat_errors_sq = np.copy(self._hist.bin_errors_sq)
        stat_errors_sq[stat_errors_sq == 0] = 1e-14

        self._cov = np.diag(stat_errors_sq)
        self._cov_mats.append(np.copy(self._cov))

        self._corr = np.diag(np.ones(self._num_bins))
        self._inv_corr = np.diag(np.ones(self._num_bins))

        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._hist.bin_counts,
            out=np.full(self._num_bins, 1e-7),
            where=self._hist.bin_counts != 0,
        )

    def add_variation(self, data, weights_up, weights_down):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        hup = Hist1d(
            bins=self._hist.num_bins,
            range=self._range,
            data=data,
            weights=weights_up
        )
        hdown = Hist1d(
            bins=self._hist.num_bins,
            range=self._range,
            data=data,
            weights=weights_down
        )

        cov_mat = get_systematic_cov_mat(self._hist.bin_counts, hup, hdown)
        self._cov_mats.append(np.copy(cov_mat))

        self._cov += cov_mat
        self._corr = cov2corr(self._cov)
        self._inv_corr = np.linalg.inv(self._corr)
        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._hist.bin_counts,
            out=np.full(self.num_bins, 1e-7),
            where=self._hist.bin_counts != 0,
        )

    def plot_on(self, ax):
        """Plots the template on given axis.
        """
        ax.hist(
            self._hist.bin_mids,
            weights=self.values,
            bins=self._hist.bin_edges,
            color=self.color,
            edgecolor="black",
            histtype="stepfilled",
        )
        ax.bar(
            x=self._hist.bin_mids,
            height=2 * self.errors,
            width=self._hist.bin_widths,
            bottom=self.values - self.errors(),
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
        )

    # -- implementation of abstract methods --

    def _init_params(self):
        """Initializes template parameters.
        """
        self._params = np.zeros(self._num_bins + 1)
        self._params[0] = np.sum(self._hist.bin_counts)

    @property
    def values(self):
        """Calculates the expected number of events per bin using
        the current yield value and nuissance parameters. Shape
        is (`num_bins`,).
        """
        return self.yield_param * self.fractions(self.nui_params)

    @property
    def errors(self):
        """numpy.ndarray: Total uncertainty per bin. This value is the
        product of the relative uncertainty per bin and the current bin
        values. Shape is (`num_bins`,).
        """
        return self._relative_errors * self.values

    def fractions(self, nui_params):
        """Calculates the per bin fraction :math:`f_i` of the template.
        This value is used to calculate the expected number of events
        per bin :math:`\\nu_i` as :math:`\\nu_i=f_i\cdot\\nu`, where
        :math:`\\nu` is the expected yield. The fractions are given as

        .. math::

            f_i=\sum\limits_{i=1}^{n_\mathrm{bins}} \\frac{\\nu_i(1+\\theta_i\cdot\epsilon_i)}{\sum_{j=1}^{n_\mathrm{bins}} \\nu_j (1+\\theta_j\cdot\epsilon_j)},

        where :math:`\\theta_j` are the nuissance parameters and
        :math:`\epsilon_j` are the relative uncertainties per bin.

        Parameters
        ----------
        nui_params : numpy.ndarray
            An array with values for the nuissance parameters.
            Shape is (`num_bins`,)

        Returns
        -------
        numpy.ndarray
            Bin fractions of this template. Shape is (`num_bins`,).
        """
        per_bin_yields = self._hist.bin_counts * (
                1 + nui_params * self._relative_errors)
        return per_bin_yields / np.sum(per_bin_yields)
