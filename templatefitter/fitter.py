import logging
import numpy as np
import numdifftools as ndt

from templatefitter.minimizer import Minimizer
from templatefitter.utility import cov2corr

from templatefitter import Histogram

import tqdm

__all__ = [
    "TemplateFitter",
    "ToyStudy"
]

logging.getLogger(__name__).addHandler(logging.NullHandler())


# TODO verify if optimization should be done twice where we use the result
# TODO from the first minimization as starting values for the second one
class TemplateFitter:
    """This class performs the parameter estimation and calculation
    of a profile likelihood based on a constructed negative log
    likelihood function.

    Parameters
    ----------
    nll : implementation of a AbstractNLL
        An instance of a class which inherits from AbstractNLL.
        This represents a negative log likelihood function.
    """

    def __init__(self, hdata, templates, nll):
        self._hdata = hdata
        self._templates = templates
        self._nll = nll(hdata, templates)
        self._fit_result = None

    def do_fit(self, update_templates=True, get_hesse=True):

        minimizer = Minimizer(self._nll, self._nll.param_names)
        minimizer.minimize(self._nll.x0, get_hesse=get_hesse)

        if update_templates:
            self._templates.update_parameters(minimizer.param_values, minimizer.param_errors)

        return minimizer.params

    @staticmethod
    def _get_hesse_approx(param_id, minimizer, profile_points):

        result = minimizer.params.get_param_value(param_id)
        param_index = minimizer.params.param_id_to_index(param_id)
        hesse_val = minimizer.hesse[param_index, param_index]
        hesse_approx = (0.5 * hesse_val * (profile_points - result) ** 2 +
                        minimizer.fcn_min_val)

        return hesse_approx

    def profile(self, param_id,  n_points=100, sigma=2., subtract_min=True):

        logging.info(f"Calculating profile likelihood for parameter: '{param_id}'")

        minimizer = Minimizer(self._nll, self._nll.param_names)
        minimizer.minimize(self._nll.x0, get_hesse=True)
        minimum = minimizer.fcn_min_val

        result, uncertainty = minimizer.params[param_id]

        profile_points = np.linspace(
            result - sigma * uncertainty, result + sigma * uncertainty, n_points
        )

        hesse_approx = self._get_hesse_approx(param_id, minimizer, profile_points)

        profile_values = np.array([])

        param_index = minimizer.params.param_id_to_index(param_id)

        for point in tqdm.tqdm(profile_points, desc="Profile Progress"):
            minimizer.release_params()
            initial_values = self._nll.x0
            initial_values[param_index] = point
            minimizer.fix_param(param_id)
            minimizer.minimize(initial_values, get_hesse=False)

            profile_values = np.append(profile_values, minimizer.fcn_min_val)

        if subtract_min:
            profile_values -= minimum
            hesse_approx -= minimum

        return profile_points, profile_values, hesse_approx


class ToyStudy:
    """This class helps you to perform toy monte carlo studies
    using given templates and an implementation of a negative
    log likelihood function. This is useful to discover possible
    biases or a over/under estimation of errors for fit parameters.

    Parameters
    ----------
    templates : TemplateCollection
        A instance of the TemplateCollection class.
    nll
        A class used as negative log likelihood function.

    Attributes
    ----------
    result_parameters :  np.ndarray
        A 2D array of fit results for the parameters of the
        likelihood.
    result_uncertainties :  np.ndarray
        A 2D array of uncertainties fo the fit results for
        the parameters of the likelihood.
    """

    def __init__(self, templates, nll):
        self._templates = templates
        self._nll = nll

        self._toy_results = {
            "parameters": [],
            "uncertainties": []}

        self._is_fitted = False

    def do_experiments(self, n_exp=1000):
        """Performs fits using the given template and generated
        toy monte carlo (following a poisson distribution) as data.

        Parameters
        ----------
        n_exp : int
            Number of toy experiments to run.
        """

        for _ in range(n_exp):
            fake_data = self._templates.generate_toy_data()
            nll = self._nll(fake_data, self._templates)

            fitter = TemplateFitter(nll)
            result = fitter.do_fit()

            self._toy_results["parameters"].append(result.x)

            uncertainties = np.sqrt(np.diag(result.covariance))
            self._toy_results["uncertainties"].append(uncertainties)

        self._is_fitted = True



    @property
    def result_parameters(self):
        self._check_state()
        return np.array(self._toy_results["parameters"])

    @property
    def result_uncertainties(self):
        self._check_state()
        return np.array(self._toy_results["uncertainties"])

    def get_toy_results(self, param_index):
        """Returns results from the toy Monte Carlo study.

        Parameters
        ----------
        param_index : int, list of int
            Index or indices of the parameter of interest.

        Returns
        -------
        parameters : np.ndarray
            Results for the fitted values of parameters specified by
            `param_index`. Shape is (`n_exp`, `len(param_index)`).
        uncertainties : np.ndarray
            Results for the uncertainties of fitted values for parameters
            specified by `param_index`. Shape is (`n_exp`, `len(param_index)`).
        """
        self._check_state()
        parameters = self.result_parameters[:, param_index]
        uncertainties = self.result_uncertainties[:, param_index]

        return parameters, uncertainties

    def get_toy_result_pulls(self, param_index):
        """Returns pulls of the results from the toy Monte Carlo
        study. The pull is defined as

        :math:`p=\\frac{\\nu^{\mathrm{fit}} - \\nu^{\mathrm{exp}}}{\sigma_{\\nu^{\mathrm{exp}}}}`,
        
        and should follow a standard noraml distribution.

        Parameters
        ----------
        param_index : int, list of int
            Index or indices of the parameter of interest.
        
        Returns
        -------
        pulls : np.ndarray
            Pull values for the fitted values of parameters specified by
            `param_index`. Shape is (`n_exp`, `len(param_index)`).
        """

        self._check_state()

        parameters, uncertainties = self.get_toy_results(param_index)
        # this works only for template yield, for nuissance parameters
        # i have change this
        expected_yield = self._templates.yields[param_index]

        return (parameters - expected_yield) / uncertainties

    def _check_state(self):
        """Checks the state of the class instance. If no toy
        experiments have been performed, a RuntimeError will
        be raised.

        Raises
        ------
        RuntimeError
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Toy experiments have not yet been performed. "
                " Execute 'do_experiments' first."
            )
