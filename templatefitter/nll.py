"""This module contains definitions for different likelihood
functions which are used as const function to be minimized in
the fit.
"""

import numpy as np


class PoissonNLL:
    """A negative log likelihood (NLL) function for binned data using
    template histograms shapes as pdfs. The NLL is calculated as
    :math:`-\log(L) = \sum\limits_i^{n_{\mathrm{bins}}} \\nu_i - n_i \log(\\nu_i)`,

    with:
    
    * :math:`\\nu_i` - expected number of events in bin :math:`i`
    * :math:`n_i` - measured number of events in bin :math:`i`

    Parameters
    ----------
    data : np.ndarray
        Bin counts of the data histogram. Shape is (nbins,).
    templates : TemplateCollection
        A TemplateCollection instance. The templates are used to
        extract the contribution from each process described by 
        the templates to the measured data set.
    """

    def __init__(self, hdata, templates):
        self._data = hdata
        self._templates = templates

    def fraction_matrix(self):
        """Calculates the fractions of the templates in all bins.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_templates, n_bins)
        """
        values = self._templates.values
        return values/np.sum(values, axis=1).reshape(-1, 1)


    def __call__(self, x):
        """This function is called by the minimize method.
        `x` is an 1-D array with shape (n,). These are the parameters
        which are fitted.

        Returns
        -------
        float
            The value of the negative log likelihood at `x`.
        """
        poi = x
        exp_evts_per_bin = np.matmul(poi, self.fraction_matrix())

        return np.sum(exp_evts_per_bin - np.matmul(
            np.log(exp_evts_per_bin), self._data.reshape(-1, 1))
            )
