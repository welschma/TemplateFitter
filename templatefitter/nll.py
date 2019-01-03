"""This module contains definitions for different likelihood
functions which are used as const function to be minimized in
the fit.
"""

import numpy as np


class PoissonNLL:
    """PoissonNLL

    Parameters
    ----------
    data : pd.DataFrame
        A pd.DataFrame instance storing the data to be fitted
        to the templates.
    templates : TemplateCollection
        A TemplateCollection instance. The templates are used to
        extract the contribution from each process described by 
        the templates to the measured data set.

    Attributes
    ---------- 
    """

    def __init__(self, data, templates):
        self._data = np.histogram(
            data[templates.variable],
            bins=templates.bin_edges
        )[0]
        self._templates = templates

    def fraction_matrix(self):
        """Calculates the fractions of the templates in all bins.

        Returns
        -------
        np.ndarray
            An 2D array of shape (n_templates, n_bins)
        """
        values = self._templates.values
        return values/np.sum(values, axis=1).reshape(-1, 1)


    def __call__(self, x):
        """This function is called by the minimize method.
        x is an 1-D array with shape (n,). These are the parameters
        which are fitted.
        """
        poi = x
        exp_evts_per_bin = np.matmul(poi, self.fraction_matrix())

        return np.sum(exp_evts_per_bin - np.matmul(
            np.log(exp_evts_per_bin, self._data.reshape(-1, 1)))
            )
