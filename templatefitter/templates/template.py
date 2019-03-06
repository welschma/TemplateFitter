import logging

from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import lru_cache
from typing import Tuple, List

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot

from templatefitter.histogram import Hist1d
from templatefitter.utility import cov2corr
from templatefitter.templates import AbstractTemplate, TemplateParameter
from templatefitter.nll import StackedTemplateNegLogLikelihood

logging.getLogger(__name__).addHandler(logging.NullHandler())



class Template(AbstractTemplate):
    """

    """

    def __init__(self, name, variable, num_bins, limits, df, weight="weight"):
        super().__init__(name)

        self._variable = variable
        self._limits = limits
        self._hist = Hist1d(num_bins, limits, df[variable].values,
                            df[weight].values)

        self._num_bins = num_bins
        self._bin_edges = self._hist.bin_edges
        self._bin_mids = self._hist.bin_mids
        self._bin_width = self._hist.bin_width

        self._cov = None
        self._corr = None
        self._inv_corr = None
        self._relative_errors = None
        self._init_errors()

        self._param_yield = TemplateParameter(
            np.sum(self._hist.bin_counts),
            np.sqrt(np.sum(self._hist.bin_errors_sq)),
            f"{name}_yield",
        )
        self._param_nui = TemplateParameter(
            np.zeros(self.num_bins), np.ones(self.num_bins), f"{name}_nui")

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
            where=self._hist.bin_counts != 0,
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
            where=self._hist.bin_counts != 0,
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

    def plot_on(self, ax, **kwargs):
        ax.hist(
            self.bin_mids,
            weights=self.values(),
            bins=self.bin_edges,
            edgecolor="black",
            histtype="stepfilled",
            **kwargs,
        )
        ax.bar(
            x=self.bin_mids,
            height=2 * self.errors(),
            width=self.bin_width,
            bottom=self.values - self.errors(),
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
        )

    def generate_asimov_dataset(self, integer_values=False):
        """Generates an Asimov dataset using the template.
        This is a binned dataset which corresponds to the
        current expectation values. Since data takes only
        integer values, the template expectation in each
        bin is rounded to the nearest integer

        Parameters
        ----------
        integer_values : bool, optional
            Wether to round Asimov data points to integer values
            or not. Default is False.

        Returns
        -------
        asimov_dataset : Hist1d
        """
        if integer_values:
            asimov_bin_counts = np.rint(self.values())
        else:
            asimov_bin_counts = self.values()
        return Hist1d.from_binned_data(self.bin_edges, asimov_bin_counts)

    def generate_toy_dataset(self):
        """Generates a toy dataset using the template. This
        is a binned dataset where each bin is treated a
        random number following a poisson distribution with
        mean equal to the bin content of all templates.

        Returns
        -------
        toy_dataset : Hist1d
        """
        toy_bin_counts = scipy.stats.poisson.rvs(self.values())
        return Hist1d.from_binned_data(self._hist.bin_edges, toy_bin_counts)

    def reset_parameters(self):
        """Sets all parameters to their original values.
        """
        self._param_nui.reset()
        self._param_yield.reset()

    # -- properties

    @property
    def yield_param(self):
        """TemplateParameter: Yield parameter."""
        return self._param_yield

    @property
    def yield_param_values(self):
        """float: Value of the yield parameter"""
        return self._param_yield.value

    @yield_param_values.setter
    def yield_param_values(self, new_value):
        self._param_yield.value = new_value

    @property
    def yield_param_errors(self):
        """float: Error of the yield parameter"""
        return self._param_yield.error

    @yield_param_errors.setter
    def yield_param_errors(self, new_error):
        self._param_yield.error = new_error

    @property
    def nui_params(self):
        """TemplateParameter: Nuissance parameters."""
        return self._param_nui

    @property
    def nui_param_values(self):
        """numpy.ndarray: Values of the the nuissance parameters.
        Shape is (`num_bins`,)."""
        return self._param_nui.value

    @nui_param_values.setter
    def nui_param_values(self, new_values):
        self._param_nui.value = new_values

    @property
    def nui_params_errors(self):
        """numpy.ndarray: Errors of the the nuissance parameters.
        Shape is (`num_bins`,)."""
        return self._param_nui.error

    @nui_params_errors.setter
    def nui_params_errors(self, new_errors):
        self._param_nui.error = new_errors

    @property
    def cov_mat(self):
        """numpy.ndarray: The covariance matrix of the template errors.
        Shape is (`num_bins`, `num_bins`)."""
        return self._cov

    @property
    def corr_mat(self):
        """numpy.ndarray: The correlation matrix of the template errors.
        Shape is (`num_bins`, `num_bins`)."""
        return self.corr_mat

    @property
    def inv_corr_mat(self):
        """numpy.ndarray: The invers correlation matrix of the
        template errors. Shape is (`num_bins`, `num_bins`)."""
        return self._inv_corr
