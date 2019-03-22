import logging

from multiprocessing import Pool

import numpy as np
import tqdm


from templatefitter.minimizer import *

__all__ = [
    "TemplateFitter",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())

# TODO work on fixing parameters and stuff

class TemplateFitter:
    """This class performs the parameter estimation and calculation
    of a profile likelihood based on a constructed negative log
    likelihood function.

    Parameters
    ----------
    templates : Implemented AbstractTemplate
        An instance of a template class that provides a negative
        log likelihood function via the `create_nll` method.
    minimizer_id : str
        A string specifying the method to be used for  the
        minimization of the Likelihood function. Available are
        'scipy' and 'iminuit'.
    """

    def __init__(self, templates, minimizer_id):
        self._templates = templates
        self._nll = templates.create_nll()
        self._fit_result = None
        self._minimizer_id = minimizer_id
        self._fixed_parameters = list()
        self._bound_parameters = dict()

    def fix_nui_params(self):
        pass

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
            A namedtuple with the most important information about the
            minimization.
        """
        minimizer = minimizer_factory(
            self._minimizer_id, self._nll, self._nll.param_names
        )

        if fix_nui_params:
            for i in range(self._templates.num_processes,
                           self._templates.num_nui_params +
                           self._templates.num_processes):
                minimizer.set_param_fixed(i)

        for param_id in self._fixed_parameters:
            minimizer.set_param_fixed(param_id)

        for param_id, bounds in self._bound_parameters.items():
            minimizer.set_param_bounds(param_id, bounds)

        # fit_result = minimizer.minimize(
        #     self._nll.x0, get_hesse=False, verbose=False
        # )
        fit_result = minimizer.minimize(
            self._nll.x0, get_hesse=get_hesse, verbose=verbose
        )

        if update_templates:
            self._templates.update_parameters(fit_result.params.values)

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
        bounds : tuple of float
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

    def profile(self, param_id, num_cpu=4, num_points=100, sigma=2.0, subtract_min=True, fix_nui_params=False):
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

        if fix_nui_params:
            for i in range(self._templates.num_processes,
                           self._templates.num_nui_params +
                           self._templates.num_processes):
                minimizer.set_param_fixed(i)
        print("Start nominal minimization")
        result = minimizer.minimize(self._nll.x0, get_hesse=True, verbose=True)
        minimum = result.fcn_min_val
        param_val, param_unc = minimizer.params[param_id]
        profile_points = np.linspace(
            param_val - sigma * param_unc, param_val + sigma * param_unc, num_points
        )
        hesse_approx = self._get_hesse_approx(param_id, result, profile_points)

        print(f"Start profiling the likelihood using {num_cpu} processes...")
        args = [(minimizer, point, result.params.values, param_id) for point in
                profile_points]
        with Pool(num_cpu) as pool:
            profile_values = np.array(
                list(tqdm.tqdm(pool.imap(self._profile_helper, args),
                               total=len(profile_points),
                               desc="Profile Progess"))
            )

        if subtract_min:
            profile_values -= minimum
            hesse_approx -= minimum

        return profile_points, profile_values, hesse_approx

    @staticmethod
    def _profile_helper(args):
        """Helper function for the calculation fo the profile nll.

        Parameters
        ----------
        args: tuple
            1st element: Minimizer object, 2nd element: parameter point,
            3rd element: Initial parameter values, 4th element: parameter
            identifier.

        Returns
        -------
        fcn_min_val : float
            Minimum function value.
        """

        minimizer = args[0]
        point = args[1]
        initial_values = args[2]
        param_id = args[3]

        minimizer.release_params()
        param_index = minimizer.params.param_id_to_index(param_id)
        initial_values[param_index] = point
        minimizer.set_param_fixed(param_id)

        try:
            loop_result = minimizer.minimize(initial_values, get_hesse=False)
        except RuntimeError as e:
            logging.info(e)
            logging.info(f"Minimization with point {point} was not "
                         f"sucessfull, trying again.")
            return np.nan

        return loop_result.fcn_min_val

    #TODO this is not yet generic, depens on param name in the likelihood
    def get_significance(self, process_id, verbose=True, fix_nui_params=False):
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
        process_id : str
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

        if fix_nui_params:
            for i in range(self._templates.num_processes,
                           self._templates.num_nui_params +
                           self._templates.num_processes):
                minimizer.set_param_fixed(i)

        print("Perform nominal minimization:")
        for param_id in self._fixed_parameters:
            minimizer.set_param_fixed(param_id)

        for param_id, bounds in self._bound_parameters.items():
            minimizer.set_param_bounds(param_id, bounds)
        fit_result = minimizer.minimize(self._nll.x0, verbose=verbose)

        if fit_result.params[f"{process_id}_yield"][0] < 0:
            return 0

        # set signal of template specified by param_id to zero and profile the likelihood
        self._templates.set_yield(process_id, 0)

        minimizer = minimizer_factory(
            self._minimizer_id, self._nll, self._nll.param_names
        )

        if fix_nui_params:
            for i in range(self._templates.num_processes,
                           self._templates.num_nui_params +
                           self._templates.num_processes):
                minimizer.set_param_fixed(i)
        for param_id in self._fixed_parameters:
            minimizer.set_param_fixed(param_id)

        for param_id, bounds in self._bound_parameters.items():
            minimizer.set_param_bounds(param_id, bounds)

        minimizer.set_param_fixed(process_id + "_yield")
        print("Background")
        profile_result = minimizer.minimize(self._nll.x0, verbose=verbose)

        q0 = 2 * (profile_result.fcn_min_val - fit_result.fcn_min_val)
        logging.debug(f"q0: {q0}")
        return np.sqrt(q0)


