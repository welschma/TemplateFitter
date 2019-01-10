"""This module contains definitions for different likelihood
functions which are used as const function to be minimized in
the fit.
"""
from abc import ABC, abstractmethod, abstractproperty

import numpy as np

from scipy.linalg import block_diag

__all__ = [
    "AbstractTemplateCostFunction",
    "AdvancedPoissonNegativeLogLikelihood"
]


class AbstractTemplateCostFunction(ABC):
    """Abstract base class for all cost function to estimate
    yields using the template method.

    Parameters
    ----------
    hdata : Histogram
        Bin counts of the data histogram. Shape is (nbins,).
    composite_template : Implementation of an AbstractCompositeTemplate
        A CompositeTemplate instance. The templates are used to
        extract the contribution from each process described by
        the templates to the measured data set.
    """

    def __init__(self, hdata, composite_template):
        self._data = hdata
        self._templates = composite_template

    @property
    def x0(self):
        """numpy.ndarray: Starting values for the minimization."""
        return self._templates.yield_values

    # -- abstract methods --

    @abstractmethod
    def __call__(self):
        pass


class AdvancedPoissonNegativeLogLikelihood(AbstractTemplateCostFunction):
    """A negative log likelihood (NLL) function for binned data using
    template histograms shapes as pdfs. The NLL is calculated as
    :math:`-\log(L) = \sum\limits_i^{n_{\mathrm{bins}}} \\nu_i - n_i \log(\\nu_i)`,

    with:
 
    * :math:`\\nu_i` - expected number of events in bin :math:`i`
    * :math:`n_i` - measured number of events in bin :math:`i`

    Parameters
    ----------
    data : Histogram
        Bin counts of the data histogram. Shape is (nbins,).
    templates : CompositeTemplateModel
        A CompositeTemplateModel instance. The templates are used to
        extract the contribution from each process described by
        the templates to the measured data set.
    
    Attributes
    ----------
    x0 : np.ndarray
        Expected yields for all used templates. Used as initial
        fitting values. Shape is (ntemplates,).
    """

    def __init__(self, hdata, composite_template):
        super().__init__(hdata, composite_template)
        self._block_diag_inv_corr_mats = block_diag(*self._templates.inv_corr_mats)

    def __call__(self, x):
        """This function is called by the minimize method.
        `x` is an 1-D array with shape (n,). These are the parameters
        which are fitted.

        Returns
        -------
        float
            The value of the negative log likelihood at `x`.
        """
        poi = x[:self._templates.num_templates]
        nuiss_params = x[self._templates.num_templates:]

        exp_evts_per_bin = poi @ self._templates.bin_fractions(
                nuiss_params
        )
        poisson_term = np.sum(
            exp_evts_per_bin - self._data.bin_counts * np.log(exp_evts_per_bin)
                              )

        gauss_term = 0.5*(nuiss_params@(
                self._block_diag_inv_corr_mats@nuiss_params))

        return poisson_term + gauss_term



