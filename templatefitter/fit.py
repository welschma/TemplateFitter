import numpy as np
import numdifftools as ndt

from scipy.optimize import minimize

from templatefitter.utility import cov2corr

class LikelihoodFitter:

    def __init__(self, nll):
        self._nll = nll

    def minimize(self, method='SLSQP'):
        fit_result = minimize(
            fun=self._nll,
            x0=self._nll.x0,
            method=method
        )

        hesse = ndt.Hessian(self._nll)(fit_result.x)
        fit_result.covariance = np.linalg.inv(hesse)
        fit_result.correlation = cov2corr(fit_result.covariance)

        return fit_result

class LikelihoodProfiler:
    pass

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

            fitter = LikelihoodFitter(nll)
            result = fitter.minimize()

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

        return (parameters - expected_yield)/uncertainties

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

            


