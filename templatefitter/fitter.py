import logging
import numpy as np

from templatefitter.minimizer import Minimizer
from templatefitter import Histogram
from templatefitter import Parameters
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
        """Performs maximum likelihood fit by minimizing the
        provided negative log likelihoood function.

        The log likelihood is minimized using the scipy's minimize
        function with the 'SLSQP' method.

        Parameters
        ----------
        update_templates : bool, optional
            Wether to update the parameters of the given templates
            or not. Default is True.
        get_hesse : bool, optional
            Wether to calculate the Hesse matrix in the estimated
            minimum of the negative log likelihood function or not.
            Can be computionally expensive if the number of parameters
            in the likelihood is high. Default is True.

        Returns
        -------
        estimated_parameters : Parameters
            An instance of the Parameters class with the estimated
            fit parameters.
        """
        minimizer = Minimizer(self._nll, self._nll.param_names)
        params = minimizer.minimize(self._nll.x0, get_hesse=get_hesse)

        if update_templates:
            self._templates.update_parameters(params.values, params.errors)

        return params

    @staticmethod
    def _get_hesse_approx(param_id, param_estimates, hesse, fcn_min,
                          profile_points):
        """Calculates a gaussian approximation of the negative log
        likelihood function using the Hesse matrix.

        Parameters
        ----------
        param_id : int or string
            Parameter index or name.
        param_estimates : Parameters
            An instance of the Parameters class with the estimated
            fit parameters.
        hesse : np.ndarray
            Hesse matrix. Shape is (`num_params`, `num_params`).
        fcn_min : float
            Estimated minimum of the neg log likelihood function.
        profile_points : np.ndarray
            Points where the estimate is evaluated. Shape is
            (`num_points`,).

        Returns
        -------
        np.ndarray
            Hesse approximation of the negative log likelihood funciton.
            Shape is (`num_points`,).

        """

        result = param_estimates.get_param_value(param_id)
        param_index = param_estimates.param_id_to_index(param_id)
        hesse_val = hesse[param_index, param_index]
        hesse_approx = (0.5 * hesse_val * (profile_points - result) ** 2 +
                        fcn_min)

        return hesse_approx

    def profile(self, param_id, num_points=100, sigma=2., subtract_min=True):
        """Performs a profile scan of the negative log likelihood
        function for the specified parameter.

        Parameters
        ----------
        param_id : int or string
            Parameter index or name.
        num_points : int
            Number of points where the negative log likelhood is
            minimized.
        sigma : float
            Defines the width of the scan. The scan range is given by
            sigma*uncertainty of the given parameter.
        subtract_min : bool
            Wether to subtract the estimated minimum of the negative
            log likelihood function or not. Default is True.

        Returns
        -------
        np.ndarray
            Scan points. Shape is (`num_points`,).
        np.ndarray
            Profile values. Shape is (`num_points`,).
        np.ndarray
            Hesse approximation. Shape is (`num_points`,).
        """

        logging.info(f"Calculating profile likelihood for parameter: '{param_id}'")

        minimizer = Minimizer(self._nll, self._nll.param_names)
        param_estimates = minimizer.minimize(self._nll.x0, get_hesse=True)
        minimum = minimizer.fcn_min_val

        result, uncertainty = minimizer.params[param_id]

        profile_points = np.linspace(
            result - sigma * uncertainty, result + sigma * uncertainty, num_points
        )

        hesse_approx = self._get_hesse_approx(
            param_id,
            param_estimates,
            minimizer.hesse,
            minimum,
            profile_points
        )

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

        self._reset_state()

        for _ in tqdm.tqdm(range(n_exp), desc="Experiments Progess"):
            htoy_data = Histogram(self._templates.num_bins, self._templates.limits)
            htoy_data.bin_counts = self._templates.generate_toy_dataset()

            fitter = TemplateFitter(htoy_data, self._templates, self._nll)
            result = fitter.do_fit(update_templates=False)

            self._toy_results["parameters"].append(result.values)
            self._toy_results["uncertainties"].append(result.errors)

        self._is_fitted = True

    def do_linearity_test(self, template_id, limits, n_points=10, n_exp=200):
        """Performs a linearity test for the yield parameter of
        the specified template.

        Parameters
        ----------
        template_id : str
            Name of the template for which the linearity test
            should be performed.
        limits : tuple of float
            Range where the yield parameter will be tested in.
        n_points : int, optional
            Number of points to test in the given range. This
            samples `n_points` in a linear space in the range
            specified by `limits`. Default is 10.
        n_exp : int, optional
            Number of toy experiments to perform per point.
            Default is 100.
        """
        param_fit_results = list()
        param_fit_errors = list()
        param_points = np.linspace(*limits, n_points)

        for param_point in tqdm.tqdm(param_points, desc="Linearity Test Progress"):
            self._reset_state()
            self._templates.set_yield(template_id, param_point)

            for _ in tqdm.tqdm(range(n_exp), desc="Experiment Progess"):
                htoy_data = Histogram(self._templates.num_bins, self._templates.limits)
                htoy_data.bin_counts = self._templates.generate_toy_dataset()

                fitter = TemplateFitter(htoy_data, self._templates, self._nll)
                result = fitter.do_fit(update_templates=False, get_hesse=False)

                self._toy_results["parameters"].append(result.values)
                self._toy_results["uncertainties"].append(result.errors)

            self._is_fitted = True

            params, _ = self.get_toy_results(
                self._templates.template_ids.index(template_id)
            )
            param_fit_results.append(np.mean(params))
            param_fit_errors.append(np.std(params))

        return param_points, param_fit_results, param_fit_errors

    @property
    def result_parameters(self):
        """np.ndarray: A 2D array of fit results for the parameters
        of the likelihood."""
        self._check_state()
        return np.array(self._toy_results["parameters"])

    @property
    def result_uncertainties(self):
        """np.ndarray: A 2D array of uncertainties fo the fit
        results for the parameters of the likelihood."""
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
        expected_yield = self._templates.yield_values[param_index]

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

    def _reset_state(self):
        """
        Resets state of the ToyStudy. This removes the toy results
        and set the state to not fitted.
        """
        self._is_fitted = False
        self._toy_results["parameters"] = list()
        self._toy_results["uncertainties"] = list()
