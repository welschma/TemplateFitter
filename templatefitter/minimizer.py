"""
Implementation of a minimizer class based on the scipy.optimize.minimize
function.
"""
import functools
import logging
from abc import ABC, abstractmethod
from collections import namedtuple

import numdifftools as ndt
import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy.optimize import minimize
import tabulate

from templatefitter.utility import cov2corr, id_to_index

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Parameters",
    "AbstractMinizer",
    "IMinuitMinimizer",
    "ScipyMinimizer",
    "minimizer_factory",
]


class Parameters:
    """Container for parameters used by the Minimizer class.
    Maps parameters described as arrays to names and indices.
    Values for parameter values, errors, covariance and correlation
    matrices are only available after they've been set by the
    minimizer.

    Parameters
    ----------
    names : list of str
        List of parameter names.
    """

    def __init__(self, names):
        self._names = names
        self._nparams = len(names)
        self._values = np.zeros(len(names))
        self._errors = np.zeros(len(names))
        self._covariance = np.zeros((len(names), len(names)))
        self._correlation = np.zeros((len(names), len(names)))

    def __str__(self):
        data = {
            "No": list(range(self._nparams)),
            "Name": self._names,
            "Value": self._values,
            "Sym. Err": self.errors
        }
        return tabulate.tabulate(data, headers="keys")

    def get_param_value(self, param_id):
        """Returns value of parameter specified by `param_id`.

        Parameters
        ----------
        param_id : int or str
            Name or index in list of names of wanted paramater

        Returns
        -------
        float
        """
        param_index = self.param_id_to_index(param_id)
        return self.values[param_index]

    def get_param_error(self, param_id):
        """Returns error of parameter specified by `param_id`.

        Parameters
        ----------
        param_id : int or str
            Name or index in list of names of wanted paramater

        Returns
        -------
        float
        """
        param_index = self.param_id_to_index(param_id)
        return self.errors[param_index]

    def __getitem__(self, param_id):
        """Gets the value and error of the specified parameter.

        Parameters
        ----------
        param_id : int or str
            Parameter index or name.

        Returns
        -------
        float
            Parameter value.
        float
            Parameter error.
        """
        param_index = self.param_id_to_index(param_id)
        return self.values[param_index], self.errors[param_index]

    def param_id_to_index(self, param_id):
        """Returns the index of the parameter specified by `param_id`.

        Parameters
        ----------
        param_id : int or str
            Parameter index or name.

        Returns
        -------
        int
        """
        return id_to_index(self.names, param_id)

    @property
    def names(self):
        """list of str: List of parameter names."""
        return self._names

    @property
    def num_params(self):
        """int: Number of parameters."""
        return self._nparams

    @property
    def values(self):
        """np.ndarray: Parameter values. Shape is (`num_params`,)."""
        return self._values

    @values.setter
    def values(self, new_values):
        if not len(new_values) == self.num_params:
            raise ValueError(
                "Number of parameter values must be equal" " to number of parameters"
            )
        self._values = new_values

    @property
    def errors(self):
        """np.ndarray: Parameter errors. Shape is (`num_params`,)."""
        return self._errors

    @errors.setter
    def errors(self, new_errors):
        if not len(new_errors) == self.num_params:
            raise ValueError(
                "Number of parameter errors must be equal" " to number of parameters"
            )
        self._errors = new_errors

    @property
    def covariance(self):
        """np.ndarray: Parameter covariance matrix. Shape is
        (`num_params`, `num_params`)."""
        return self._covariance

    @covariance.setter
    def covariance(self, new_covariance):
        self._covariance = new_covariance

    @property
    def correlation(self):
        """np.ndarray: Parameter correlation matrix. Shape is
         (`num_params`, `num_params`)."""
        return self._correlation

    @correlation.setter
    def correlation(self, new_correlation):
        self._correlation = new_correlation


MinimizeResult = namedtuple("MinimizeResult", ["fcn_min_val", "params", "succes"])

MinimizeResult.__doc__ = """NamedTuple storing the minimization results."""
MinimizeResult.fcn_min_val.__doc__ = """float: Estimated minimum of the 
objective function."""
MinimizeResult.params.__doc__ = """Parameters: An instance of the parameters 
class."""
MinimizeResult.succes.__doc__ = """bool: Whether or not the optimizer exited 
successfully."""


class AbstractMinizer(ABC):
    def __init__(self, fcn, param_names):
        self._fcn = fcn
        self._params = Parameters(param_names)

        # this lists can be different for different minimizer implementations
        self._fixed_params = list()
        self._param_bounds = [(None, None) for _ in self._params.names]

        self._fcn_min_val = None

        self._success = None
        self._status = None
        self._message = None

    @abstractmethod
    def minimize(self, initial_params, verbose=False):
        pass

    @abstractmethod
    def set_param_fixed(self, param_id):
        pass

    @abstractmethod
    def release_params(self):
        pass

    def set_param_bounds(self, param_id, bounds):
        """Sets parameter boundaries which constrain the minimization.

        Parameters
        ----------
        param_id : int or str
            Parameter identifier, which can be it's name or its index
            in `param_names`.
        bounds : tuple of float or None
            A tuple specifying the lower and upper boundaries for the
            given parameter. A value of `None` corresponds to no
            boundary.
        """
        param_index = self.params.param_id_to_index(param_id)
        self._param_bounds[param_index] = bounds

    @property
    def fcn_min_val(self):
        """str: Value of the objective function at it's estimated minimum.
        """
        return self._fcn_min_val

    @property
    def params(self):
        """Parameters: Instance of the Parameter class. Stores the parameter values,
        errors, covariance and correlation matrix.
        """
        return self._params

    @property
    def param_values(self):
        """np.ndarray: Estimated parameter values at the minimum of fcn.
        Shape is (`num_params`).
        """
        return self._params.values

    @property
    def param_errors(self):
        """np.ndarray: Estimated parameter values at the minimum of fcn.
        Shape is (`num_params`).
        """
        return self._params.errors

    @property
    def param_covariance(self):
        """np.ndarray: Estimated covariance matrix of the parameters.
        Calculated from the inverse of the Hesse matrix of fcn evaluated
        at it's minimum. Shape is (`num_params`, `num_params`).
        """
        return self._params.covariance

    @property
    def param_correlation(self):
        """np.ndarray: Estimated correlation matrix of the parameters.
         Shape is (`num_params`, `num_params`).
         """
        return self._params.correlation


class IMinuitMinimizer(AbstractMinizer):
    def __init__(self, fcn, param_names):
        super().__init__(fcn, param_names)
        self._fixed_params = [False for _ in self.params.names]

    def minimize(self, initial_params, verbose=False, errordef=0.5, **kwargs):

        m = Minuit.from_array_func(
            self._fcn,
            initial_params,
            error=0.1 * initial_params,
            errordef=errordef,
            fix=self._fixed_params,
            name=self.params.names,
            limit=self._param_bounds,
            print_level=1 if verbose else 0,
        )

        fmin, _ = m.migrad()

        self._fcn_min_val = m.fval
        self._params.values = m.np_values()
        self._params.errors = m.np_errors()
        self._params.covariance = m.np_matrix()
        self._params.correlation = m.np_matrix(correlation=True)

        self._success = (
            fmin["is_valid"] and fmin["has_valid_parameters"] and fmin["has_covariance"]
        )

        if not self._success:
            raise RuntimeError(f"Minimization was not successful.\n" f"{fmin}\n")

        return MinimizeResult(m.fval, self._params, self._success)

    def set_param_fixed(self, param_id):
        param_index = self.params.param_id_to_index(param_id)
        self._fixed_params[param_index] = True

    def release_params(self):
        self._fixed_params = [False for _ in self.params.names]


class ScipyMinimizer(AbstractMinizer):
    """General wrapper class around scipy.optimize.minimize
    function. Allows mapping of parameter names to the array
    entries used by scipy's `minimize` function.

    Parameters
    ----------
    fcn : callable
        Objective function to be minimized of type ``fun(x, *args)``
        where `x` is an np.ndarray of shape (`n`,) and args is a tuple
        of fixed parameters.
    param_names : list of str
        A list of parameter names. This maps the entries from the `x`
        argument of `fcn` to strings.
    """

    def __init__(self, fcn, param_names):
        super().__init__(fcn, param_names)

    def minimize(
        self, initial_param_values, additional_args=(), get_hesse=True, verbose=False
    ):
        """Performs minimization of given objective function.

        Parameters
        ----------
        initial_param_values : np.ndarray or list of floats
            Initial values for the parameters used as starting values.
            Shape is (`num_params`,).
        additional_args : tuple
            Tuple of additional arguments for the objective function.
        get_hesse : bool
            If True, the Hesse matrix is estimated at the minimum
            of the objective function. This allows the calculation
            of parameter errors. Default is True.
        verbose: bool
            If True, a convergence message is printed. Default is False.

        Returns
        -------
        MinimizeResult
        """
        constraints = self._create_constraints(initial_param_values)

        opt_result = minimize(
            fun=self._fcn,
            x0=initial_param_values,
            args=additional_args,
            method="SLSQP",
            bounds=self._param_bounds,
            constraints=constraints,
            options={"disp": verbose},
        )

        self._success = opt_result.success
        self._status = opt_result.status
        self._message = opt_result.message

        if not opt_result.success:
            raise RuntimeError(
                f"Minimization was not successful.\n"
                f"Status: {opt_result.status}\n"
                f"Message: {opt_result.message}"
            )

        self._params.values = opt_result.x
        self._fcn_min_val = opt_result.fun

        if get_hesse:
            hesse = ndt.Hessian(self._fcn)(self._params.values, *additional_args)
            self._params.covariance = np.linalg.inv(hesse)
            self._params.correlation = cov2corr(self._params.covariance)
            self._params.errors = np.sqrt(np.diag(self._params.covariance))

        if verbose:
            print(self._params)

        result = MinimizeResult(opt_result.fun, self._params, self._success)

        return result

    def set_param_fixed(self, param_id):
        """Fixes specified parameter to it's initial value given in
        `initial_param_values`.

        Parameters
        ----------
        param_id : int or str
            Parameter identifier, which can be it's name or its index
            in `param_names`.
        """
        param_index = self.params.param_id_to_index(param_id)
        self._fixed_params.append(param_index)

    def release_params(self):
        """
        Removes all constraint specified.
        """
        self._fixed_params = list()

    def _create_constraints(self, initial_param_values):
        """Creates the dictionary used by scipy's minimize function
        to constrain parameters. The dictionary is used to fix
        parameters specified set in `fixed_param`.

        Parameters
        ----------
        initial_param_values : np.ndarray or list of floats
            Initial parameter values.

        Returns
        -------
        list of dict
            A list of dictionaries, which is passed to scipy's
            minimize function.
        """
        constraints = list()

        for fixed_param in self._fixed_params:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: x[fixed_param] - initial_param_values[fixed_param],
                }
            )

        return constraints

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def calculate_hesse_matrix(fcn, x, args):
        """Calculates the Hesse matrix of callable `fcn` numerically.

        Parameters
        ----------
        fcn : callable
            Objective function of type ``fun(x, *args)``.
        x : np.ndarray
            Parameters of `fcn` as np.ndarray of shape (`num_params`,).
        args : tuple
            Additional arguments for `fcn`.

        Returns
        -------
        np.ndarray
            Hesse matrix of `fcn` at point x. Shape is (`num_params`, `num_params`).
        """
        return ndt.Hessian(fcn)(x, *args)


def minimizer_factory(id, fcn, names):
    available_minimizer = {"scipy": ScipyMinimizer, "iminuit": IMinuitMinimizer}

    return available_minimizer[id.lower()](fcn, names)
