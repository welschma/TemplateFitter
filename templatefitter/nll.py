"""This module contains definitions for different likelihood
functions which are used as const function to be minimized in
the fit.
"""
from abc import ABC, abstractmethod

import logging
import itertools
import numpy as np

from scipy.linalg import block_diag

__all__ = ["AbstractTemplateCostFunction", "StackedTemplateNegLogLikelihood"]

logging.getLogger(__name__).addHandler(logging.NullHandler())


class AbstractTemplateCostFunction(ABC):
    """Abstract base class for all cost function to estimate
    yields using the template method.

    Parameters
    ----------
    histdataset : AbstractHist
        Bin counts of the data histogram. Shape is (nbins,).
    templates : AbstractTemplate
        A CompositeTemplate instance. The templates are used to
        extract the contribution from each process described by
        the templates to the measured data set.
    """

    def __init__(self, histdataset, templates):
        self._dataset = histdataset
        self._templates = templates

    # -- abstract properties

    @property
    @abstractmethod
    def x0(self):
        """numpy.ndarray: Starting values for the minimization."""
        pass

    @property
    @abstractmethod
    def param_names(self):
        """list of str: Parameter names. Used for convenience."""
        pass

    # -- abstract methods --

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass


class StackedTemplateNegLogLikelihood(AbstractTemplateCostFunction):
    """A negative log likelihood (NLL) function for binned data using
    template histograms shapes as pdfs. The NLL is calculated as

    .. math::

    `-\log(L) = \sum\limits_{i=1}^{n_{\mathrm{bins}}} \\nu_i - n_i \log(\\nu_i)`,

    with:

    * :math:`\\nu_i` - total expected number of events in bin :math:`i`
    * :math:`n_i` - measured number of events in bin :math:`i`.

    The total expected number of events per bin is given by

    .. math::

    `\\nu_i = \sum\limits_{k=1}^{n_\mathrm{templates}} f_{ik}\\nu_{ik}`,

    with:

    * :math:`\\nu_{ik}` - expected number of events in bin :math:`i` of template :math:`k`
    * :math:`f_{ik}` - fraction of template :math:`k` in bin :math:`i`.

    :math:`f_{ik} does depend on a nuissance parameter :math:`\theta_{ik}`:

    .. math::

    `f_{ik} = \frac{\\nu_{ik}(1 + \theta_{ik}\epsilon{ik})}{\sum\limits_{j=1}^{n_\mathrm{bins}}\\nu_{jk}(1 + \theta_{jk}\epsilon{jk})},

    where :math:`\epsilon_{jk}` is the relative uncertainty of template
    :math:`k` in bin :math:`j`.

    Parameters
    ----------
    binned_dataset: Hist1d
        Histogram of the dataset.
    templates: StackedTemplate
        A StackedTemplate instance. The templates are used to
        extract the contribution from each process described by
        the templates to the measured data set.
    """

    def __init__(self, binned_dataset, templates):
        super().__init__(binned_dataset, templates)
        self._block_diag_inv_corr_mats = block_diag(
            *self._templates.inv_corr_mats)

    @property
    def x0(self):
        initial_yields = self._templates.yield_param_values
        initial_nui_params = self._templates.nui_param_values.reshape(-1)

        return np.concatenate((initial_yields, initial_nui_params))

    @property
    def param_names(self):
        yields = [
            template_id + "_yield"
            for template_id in self._templates.template_names
        ]
        nui_params = [[
            template_name + "_nui" + f"_{i}"
            for i in range(self._templates.num_bins)
        ] for template_name in self._templates.template_names]
        yields.extend(itertools.chain.from_iterable(nui_params))
        return yields

    def __call__(self, x: np.ndarray) -> float:
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

        exp_evts_per_bin = poi @ self._templates.fractions(nuiss_params)
        poisson_term = np.sum(exp_evts_per_bin - self._dataset.bin_counts *
                              np.log(exp_evts_per_bin))
        gauss_term = 0.5 * (
            nuiss_params @ self._block_diag_inv_corr_mats @ nuiss_params)

        return poisson_term + gauss_term
