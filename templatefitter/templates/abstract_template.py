import logging
from abc import ABC, abstractmethod

import numpy as np
from numba import jit

from templatefitter.utility import cov2corr, get_systematic_cov_mat

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["AbstractTemplate",]


class AbstractTemplate(ABC):
    """Defines the template interface.

    """

    def __init__(self, name):
        self._name = name
        self._params = None
        self._default_yield = None
        self._bins = None
        self._num_bins = None
        self._range = None
        self._hist = None
        self._flat_bin_counts = None
        self._flat_bin_errors_sq = None
        self._cov_mats = list()
        self._cov = None
        self._corr = None
        self._inv_corr = None
        self._relative_errors = None


    # -- properties --

    @property
    def name(self):
        """str: Template identifier."""
        return self._name

    @property
    def num_bins(self):
        """int: Number of bins."""
        return self._num_bins

    @property
    def shape(self):
        """tuple of int: Template shape."""
        return self._hist.shape

    @property
    def bins(self):
        """int or tuple of int: Number of bins."""
        return self._bins

    @property
    def bin_mids(self):
        return self._hist.bin_mids

    @property
    def bin_edges(self):
        return self._hist.bin_edges

    @property
    def bin_widths(self):
        return self._hist.bin_widths

    @property
    def range(self):
        """"""
        return self._range

    @property
    def params(self):
        """numpy.ndarray: Array of template parameters.
        The first entry is the yield, the rest are the
        nuissance parameters. Shape is (`num_bins + 1`,).
        """
        return self._params

    @params.setter
    def params(self, new_values):

        if new_values.shape != self._params.shape:
            raise RuntimeError(
                "Shape of new parameter array is not compatible to this template."
            )

        self._params = new_values

    @property
    def yield_param(self):
        """float: The current yield value.
        """
        return self._params[0]

    @yield_param.setter
    def yield_param(self, new_value):
        if isinstance(new_value, np.ndarray):
            if len(new_value) > 1:
                raise RuntimeError("Yield parameter has to be of type float.")
        elif not (isinstance(new_value, float) or isinstance(new_value, int)):
            raise RuntimeError("Yield parameter has to be of type float.")

        self._params[0] = new_value

    @property
    def nui_params(self):
        """numpy.ndarray: The current nuissance parameters.
        """
        return self._params[1:]

    @nui_params.setter
    def nui_params(self, new_values):
        if new_values.shape != (self.num_bins,):
            raise RuntimeError(
                "Shape of new nuissance parameters not compatible to this template."
            )

        self._params[1:] = new_values

    def reset(self):
        """Resets parameter to the original values.
        """
        self._init_params()

    def _init_params(self):
        """Initializes template parameters.
        """
        self._params = np.zeros(self._num_bins + 1)
        self._param_errors = np.zeros(self._num_bins + 1)
        self._params[0] = np.sum(self._hist.bin_counts)

    def _init_errors(self):
        """The statistical covariance matrix is initialized as diagonal
        matrix of the sum of weights squared per bin in the underlying
        histogram. For empty bins, the error is set to 1e-7. The errors
        are initialized to be 100% uncorrelated. The relative errors per
        bin are set to 1e-7 in case of empty bins.
        """
        stat_errors_sq = np.copy(self._flat_bin_errors_sq)
        stat_errors_sq[stat_errors_sq == 0] = 1e-14

        self._cov = np.diag(stat_errors_sq)
        self._cov_mats.append(np.copy(self._cov))

        self._corr = np.diag(np.ones(self._num_bins))
        self._inv_corr = np.diag(np.ones(self._num_bins))

        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._flat_bin_counts,
            out=np.full(self._num_bins, 1e-7),
            where=self._flat_bin_counts != 0,
        )

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
        return bin_fractions(nui_params, self._flat_bin_counts, self._relative_errors)

    def _add_cov_mat(self, hup, hdown):
        """Helper function. Calculates a covariance matrix from
        given histogram up and down variations.
        """
        cov_mat = get_systematic_cov_mat(
            hup.bin_counts.flatten(), hdown.bin_counts.flatten()
        )
        self._cov_mats.append(np.copy(cov_mat))

        self._cov += cov_mat
        self._corr = cov2corr(self._cov)
        self._inv_corr = np.linalg.inv(self._corr)
        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._flat_bin_counts,
            out=np.full(self._num_bins, 1e-7),
            where=self._flat_bin_counts != 0,
        )

    @property
    def values(self):
        """Calculates the expected number of events per bin using
        the current yield value and nuissance parameters. Shape
        is (`num_bins`,).
        """
        return self.yield_param * self.fractions(self.nui_params).reshape(
            self._hist.shape
        )

    @property
    def errors(self):
        """numpy.ndarray: Total uncertainty per bin. This value is the
        product of the relative uncertainty per bin and the current bin
        values. Shape is (`num_bins`,).
        """
        return self._relative_errors.reshape(self._hist.shape) * self.values

    @property
    def cov_mat(self):
        return self._cov

    @property
    def cov_mats(self):
        return self._cov_mats

    @property
    def corr(self):
        return self._corr

    @property
    def inv_corr_mat(self):
        return self._inv_corr

    @abstractmethod
    def add_variation(self, data, weights_up, weights_down):
        pass


@jit(nopython=True)
def bin_fractions(nui_params, bin_counts, rel_errors):
    per_bin_yields = bin_counts * (1. + nui_params * rel_errors)
    return per_bin_yields / np.sum(per_bin_yields)
