import logging

import numpy as np
import tqdm


from templatefitter import Histogram
from templatefitter.minimizer import *

__all__ = ["TemplateFitter", "ToyStudy"]

logging.getLogger(__name__).addHandler(logging.NullHandler())


class TemplateFitter:
    """This class performs the parameter estimation and calculation
    of a profile likelihood based on a constructed negative log
    likelihood function.

    Parameters
    ----------
    hdata : Implemented AbstractHist
        Histogram filled with data events.
    templates : Implemented AbstractTemplate
        An instance of a template class that provides a negative
        log likelihood function via the `create_nll` method.
    minimizer_id : str
        A string specifying the method to be used for  the
        minimization of the Likelihood function. Available are
        'scipy' and 'iminuit'.
    """

    def __init__(self, hdata, templates, minimizer_id):
        self._hdata = hdata
        self._templates = templates
        self._nll = templates.create_nll(hdata)
        self._fit_result = None
        self._minimizer_id = minimizer_id

        self.n_templates = self._templates.num_templates
        self.n_bins = self._templates.num_bins

        self._fixed_parameters = list()
        self._bound_parameters = dict()

    def do_fit(self, update_templates=True, get_hesse=True, verbose=True, fix_nui_params=False):
        """Performs maximum likelihood fit by minimizing the
        provided negative log likelihoood function.

        Parameters
        ----------
        update_templates : bool, optional
            Whether to update the parameters of the given templates
            or not. Default is True.
        verbose : bool, optional
            Whether to print fit information or not. Default is True
        fix_nui_params : bool, optional
            Wheter to fix nuissance parameters in the fit or not.
            Default is False.
        get_hesse : bool, optional
            Whether to calculate the Hesse matrix in the estimated
            minimum of the negative log likelihood function or not.
            Can be computationally expensive if the number of parameters
            in the likelihood is high. It is only needed for the scipy
            minimization method. Default is True.

        Returns
        -------
        MinimizeResult : namedtuple
            A namedtuple with the most important informations about the
            minimization.
        """
        minimizer = minimizer_factory(
            self._minimizer_id, self._nll, self._nll.param_names
        )

        if fix_nui_params:
            for i in range(self.n_templates,
                           self.n_templates*self.n_bins+self.n_templates):
                minimizer.set_param_fixed(i)

        for param_id in self._fixed_parameters:
            minimizer.set_param_fixed(param_id)

        for param_id, bounds in self._bound_parameters.items():
            minimizer.set_param_bounds(param_id, bounds)

        fit_result = minimizer.minimize(
            self._nll.x0, get_hesse=get_hesse, verbose=verbose
        )

        if update_templates:
            self._templates.update_parameters(
                fit_result.params.values, fit_result.params.errors
            )

        return fit_result

    def set_parameter_fixed(self, param_id):
        """Adds parameter to the fixed parameter list.

        Parameters
        ----------
        param_id : str or int
            Parameter identifier.
        """
        self._fixed_parameters.append(param_id)

    def set_parameter_bounds(self, param_id, bounds):
        """Adds parameter and its boundaries to the bound
        parameter dictionary.

        Parameters
        ----------
        param_id : str or int
            Parameter identifier.
        boudns : tuple of float
            Lower and upper boundaries for this parameter.
        """

        self._bound_parameters[param_id] = bounds

    @staticmethod
    def _get_hesse_approx(param_id, fit_result, profile_points):
        """Calculates a gaussian approximation of the negative log
        likelihood function using the Hesse matrix.

        Parameters
        ----------
        param_id : int or string
            Parameter index or name.
        fit_result : MinimizeResult
            A namedtuple with the most important informations about the
            minimization.
        profile_points : np.ndarray
            Points where the estimate is evaluated. Shape is
            (`num_points`,).

        Returns
        -------
        np.ndarray
            Hesse approximation of the negative log likelihood funciton.
            Shape is (`num_points`,).

        """

        result = fit_result.params.get_param_value(param_id)
        param_index = fit_result.params.param_id_to_index(param_id)
        hesse_error = fit_result.params.errors[param_index]
        hesse_approx = (
            0.5 * (1 / hesse_error) ** 2 * (profile_points - result) ** 2
            + fit_result.fcn_min_val
        )

        return hesse_approx

    def profile(self, param_id, num_points=100, sigma=2.0, subtract_min=True):
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
        subtract_min : bool, optional
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
        print(f"\nCalculating profile likelihood for parameter: '{param_id}'")

        minimizer = minimizer_factory(
            self._minimizer_id, self._nll, self._nll.param_names
        )
        result = minimizer.minimize(self._nll.x0, get_hesse=True)
        minimum = result.fcn_min_val

        param_val, param_unc = minimizer.params[param_id]

        profile_points = np.linspace(
            param_val - sigma * param_unc, param_val + sigma * param_unc, num_points
        )

        hesse_approx = self._get_hesse_approx(param_id, result, profile_points)

        profile_values = np.array([])

        param_index = minimizer.params.param_id_to_index(param_id)

        for point in tqdm.tqdm(profile_points, desc="Profile Progress"):
            minimizer.release_params()
            initial_values = self._nll.x0
            initial_values[param_index] = point
            minimizer.set_param_fixed(param_id)
            result = minimizer.minimize(initial_values, get_hesse=False)
            loop_result = minimizer.minimize(result.params.values, get_hesse=False)

            profile_values = np.append(profile_values, loop_result.fcn_min_val)

        if subtract_min:
            profile_values -= minimum
            hesse_approx -= minimum

        return profile_points, profile_values, hesse_approx

    def get_significance(self, tid, verbose=True):
        """Calculate significance for yield parameter of template
        specified by `tid` using the profile likelihood ratio.

        The significance is base on the profile likelihood ratio

        .. math::

            \lambda(\\nu) = \\frac{L(\\nu, \hat{\hat{\\theta}})}{L(\hat{\\nu}, \hat{\\theta})},

        where :math:`\hat{\hat{\\theta}}` maximizes :math:`L`
        for a specified :math:`\\nu` and :math:`(\hat{\\nu}, \hat{\\theta})`
        maximizes :math:`L` totally.

        The test statistic used for discovery is

        .. math::

            q_0 = \left\{ \\begin{array}{lr} -2\log(\lambda(0)) & \hat{\\nu} \ge 0,\\\\ 0 & \hat{\\nu} < 0 \end{array} \\right.



        Parameters
        ----------
        tid : str
            Id of component in the composite template for which the
            significance of the yield parameter should be calculated.

        Returns
        -------
        significance : float
            Fit significance for the yield parameter in gaussian
            standard deviations.

        """

        # perform the nominal minimization

        minimizer = minimizer_factory(
            self._minimizer_id, self._nll, self._nll.param_names
        )

        print("Perform nominal minimization:")
        fit_result = minimizer.minimize(self._nll.x0, verbose=verbose)

        if fit_result.params["yield_" + tid][0] < 0:
            return 0

        # set signal of template specified by param_id to zero and profile the likelihood
        self._templates.set_yield(tid, 0)

asjkldfjhuasdgfhs
        minimizer = minimizer_factory(
            self._minimizer_id, self._nll, self._nll.param_names
        )
        minimizer.set_param_fixed(tid + "_yield")
        print("Background")
        profile_result = minimizer.minimize(self._nll.x0, verbose=verbose)

        q0 = 2 * (profile_result.fcn_min_val - fit_result.fcn_min_val)
        logging.debug(f"q0: {q0}")
        return np.sqrt(q0)


class ToyStudy:
    """This class helps you to perform toy monte carlo studies
    using given templates and an implementation of a negative
    log likelihood function. This is useful to discover possible
    biases or a over/under estimation of errors for fit parameters.

    Parameters
    ----------
    templates : TemplateCollection
        A instance of the TemplateCollection class.
    """

    def __init__(self, templates, minimizer_id):
        self._templates = templates
        self._mimizer_id = minimizer_id

        self._toy_results = {"parameters": [], "uncertainties": []}

        self._is_fitted = False

    def do_experiments(self, n_exp=1000, max_tries=10):
        """Performs fits using the given template and generated
        toy monte carlo (following a poisson distribution) as data.

        Parameters
        ----------
        n_exp : int
            Number of toy experiments to run.
        max_tries : int
            Maximum number of tries for an experiment if a RuntimeError
            occurs.
        """

        self._reset_state()

        print(f"Performing toy study with {n_exp} experiments...")
        for _ in tqdm.tqdm(range(n_exp), desc="Experiments Progress"):
                self._experiment(max_tries)

        self._is_fitted = True

    def _experiment(self, max_tries=10, get_hesse=True):
        """
        Helper function for toy experiments.
        """
        for _ in range(max_tries):
            try:

                htoy_data = self._templates.generate_toy_dataset()

                fitter = TemplateFitter(
                    htoy_data, self._templates, minimizer_id=self._mimizer_id
                )
                result = fitter.do_fit(update_templates=False,
                                       verbose=False,
                                       get_hesse=get_hesse)

                self._toy_results["parameters"].append(result.params.values)
                self._toy_results["uncertainties"].append(result.params.errors)

                return None

            except RuntimeError:
                logging.debug("RuntimeError occured in toy experiment. Trying again")
                continue

        raise RuntimeError("Experiment exceed max number of retries.")

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

        print(f"Performing linearity test for parameter: {template_id}")
        for param_point in tqdm.tqdm(param_points, desc="Linearity Test Progress"):
            self._reset_state()

            self._templates[template_id].yield_param_values = param_point

            for _ in tqdm.tqdm(range(n_exp), desc="Experiment Progress"):
                self._experiment(get_hesse=False)

            self._is_fitted = True

            params, _ = self.get_toy_results(
                self._templates.template_names.index(template_id)
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
        expected_yield = self._templates.yield_param_values[param_index]

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
