"""This module provides several classes which help to implement templates
used for binned likelihood fits where the expected number of events is
estimated from different histograms.
"""
import logging

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import scipy.stats

from templatefitter.histogram import Hist1d
from templatefitter.utility import cov2corr

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AbstractTemplate",
    "Template",
    "StackedTemplate",
    "SimultaneousTemplate"
]


class TemplateParameter:
    """

    """

    def __init__(self, value, error, name):
        self.value = value
        self.error = error
        self.name = name


class AbstractTemplate(ABC):
    """
    """

    def __init__(self, name):
        self._name = name
        self._num_bins = None

    @property
    def name(self):
        """str: Template identifier."""
        return self._name

    @property
    def num_bins(self):
        return self._num_bins

    # -- abstract methods

    @abstractmethod
    def generate_asimov_dataset(self):
        pass

    @abstractmethod
    def generate_toy_dataset(self):
        pass

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def fractions(self):
        pass

    @abstractmethod
    def plot_on(self):
        pass


class Template(AbstractTemplate):

    def __init__(self, name, variable, num_bins, limits, df, weight="weight"):
        super().__init__(name)

        self._variable = variable
        self._num_bins = num_bins
        self._limits = limits
        self._hist = Hist1d(num_bins, limits,
                            df[variable].values, df[weight].values)
        self._init_errors()
        self._param_yield = TemplateParameter(
            np.sum(self._hist.bin_counts),
            np.sqrt(np.sum(self._hist.bin_errors_sq)),
            f"{name}_yield"
        )
        self._param_nui = TemplateParameter(
            np.zeros(self.num_bins),
            np.ones(self.num_bins),
            f"{name}_nui"
        )

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
        self._corr = np.diag(np.ones(self.num_bins))
        self._inv_corr = np.diag(np.ones(self.num_bins))
        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._hist.bin_counts,
            out=np.full(self.num_bins, 1e-7),
            where=self._hist.bin_counts != 0
        )

    def add_covariance_matrix(self, covariance_matrix):
        """Add a covariance matrix for a systematic error to this template.
        This updates the total covariance matrix, the correlation matrix and
        the relative bin errors.

        Parameters
        ----------
        covariance_matrix : numpy.ndarray
            A covariance matrix. It is not checked if the matrix is
            valid (symmetric, positive semi-definite. Shape is
            (`num_bins`, `num_bins`).
        """
        self._cov += covariance_matrix
        self._corr = cov2corr(self._cov)
        self._inv_corr = np.linalg.inv(self._corr)
        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._hist.bin_counts,
            out=np.full(self.num_bins, 1e-7),
            where=self._hist.bin_counts != 0
        )

    def errors(self):
        """numpy.ndarray: Total uncertainty per bin. This value is the
        product of the relative uncertainty per bin and the current bin
        values. Shape is (`num_bins`,)."""
        return self._relative_errors * self.values()

    def values(self):
        """Calculates the expected number of events per bin using
        the current yield value and nuissance parameters.

        Returns
        -------
        numpy.ndarray
        """
        return self._param_yield.value * self.fractions(self._param_nui.value)

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
        nuiss_params : numpy.ndarray
            An array with values for the nuissance parameters.
            Shape is (`num_bins`,)

        Returns
        -------
        numpy.ndarray
            Bin fractions of this template. Shape is (`num_bins`,).
        """
        per_bin_yields = self._hist.bin_counts * (
                1 + nui_params * self._relative_errors
        )
        return per_bin_yields / np.sum(per_bin_yields)

    def plot_on(self, ax, **kwargs):
        ax.hist(
            self._hist.bin_mids,
            weights=self.values(),
            bins=self._hist.bin_edges,
            edgecolor="black",
            histtype="stepfilled",
            **kwargs,
        )
        ax.bar(
            x=self._hist.bin_mids,
            height=2 * self.errors(),
            width=self._hist.bin_width,
            bottom=self.values - self.errors(),
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
        )

    def generate_asimov_dataset(self):
        asimov_bin_counts = np.rint(self.values())
        return Hist1d.from_binned_data(self._hist.bin_edges, asimov_bin_counts)

    def generate_toy_dataset(self):
        toy_bin_counts = scipy.stats.poisson.rvs(self.values())
        return Hist1d.from_binned_data(self._hist.bin_edges, toy_bin_counts)

    # -- properties

    @property
    def yield_param(self):
        """TemplateParameter: Yield parameter."""
        return self._param_yield

    @property
    def nui_params(self):
        """TemplateParameter: Nuissance parameters."""
        return self._param_nui


class StackedTemplate(AbstractTemplate):

    def __init__(self, name, variable, num_bins, limits):
        super().__init__(name)
        self._variable = variable
        self._num_bins = num_bins
        self._limits = limits
        self._template_dict = OrderedDict()

    def add_template(self):
        pass

    def create_template(self):
        pass



class SimultaneousTemplate(AbstractTemplate):
    pass
