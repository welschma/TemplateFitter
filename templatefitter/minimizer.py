"""
Implementation of a minimizer class based on the scipy.optimize.minimize
function.
"""
import functools
import numpy as np
import numdifftools as ndt

from scipy.optimize import minimize

from templatefitter.utility import cov2corr

__all__ = [
    "Parameters",
    "Minimizer",
]


class Parameters:
    """Containter for parameters used by the Minimizer class.
    Maps parameters described as arrays to names and indices.
    Values for parameter values, errors, covariance and correlation
    matrices are only available after they've been set by the
    minimizer.

    Parameters
    ----------
    names : list of str
        List of parameter names.

    Attributes
    ----------
    names : list of str
        List of parameter names.
    nparams : int
        Number of parameters.
    values : np.ndarray
        Parameter values. Shape is (`nparams`,).
    errors : np.ndarray
        Parameter errors. Shape is (`nparams`,).
    covariance : np.ndarray
        Parameter covariance matrix. Shape is (`nparams`, `nparams`).
    correlation : np.ndarray
        Parameter correlation matrix. Shape is (`nparams`, `nparams`).
    """
    def __init__(self, names):
        self._names = names
        self._nparams = len(names)
        self._values = None
        self._errors = None
        self._covariance = None
        self._correlation = None

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
        if isinstance(param_id, str):
            param_index = self.names.index(param_id)
        elif isinstance(param_id, int):
            param_index = param_id
        else:
            raise ValueError(
                "Specify the parameter either by its name (as str) or by "
                "its index (as int)."
            )
        return param_index

    @property
    def names(self):
        return self._names

    @property
    def nparams(self):
        return self._nparams

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        if not len(new_values) == self.nparams:
            raise ValueError("Number of parameter values must be equal"
                             " to number of parameters")
        self._values = new_values

    @property
    def errors(self):
        return self._errors

    @errors.setter
    def errors(self, new_errors):
        if not len(new_errors) == self.nparams:
            raise ValueError("Number of parameter errors must be equal"
                             " to number of parameters")
        self._errors = new_errors

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, new_covariance):
        if not new_covariance.shape == (self.nparams, self.nparams):
            raise ValueError("New covariance matrix shape must be equal"
                             " to (nparams, nparams)")
        self._covariance = new_covariance

    @property
    def correlation(self):
        return self._correlation

    @correlation.setter
    def correlation(self, new_correlation):
        if not new_correlation.shape == (self.nparams, self.nparams):
            raise ValueError("New correlation matrix shape must be equal"
                             " to (nparams, nparams)")
        self._correlation = new_correlation


class Minimizer:
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
        self._fcn = fcn
        self._params = Parameters(param_names)

        self._fixed_params = list()

        self._fcn_min_val = None

        self._success = None
        self._status = None
        self._message = None

        self._hesse = None
        self._hesse_inv = None

    def minimize(self, initial_param_values, additional_args=(), get_hesse=True):
        """Performs minimization of given objective function.

        Parameters
        ----------
        initial_param_values : np.ndarray or list of floats
            Initial values for the parameters used as starting values.
            Shape is (`nparams`,).
        additional_args : tuple
            Tuple of additional arguemnts for the objective function.
        get_hesse : bool
            If True, the Hesse matrix is estimated at the minimum
            of the objective function. This allows the calculation
            of parameter errors. Default is True.
        """
        constraints = self._create_constraints(initial_param_values)

        opt_result = minimize(
            fun=self._fcn,
            x0=initial_param_values,
            args=additional_args,
            method="SLSQP",
            constraints=constraints,
            options={"disp": True}
        )

        self._success = opt_result.success
        self._status = opt_result.status
        self._message = opt_result.message

        if not self._success:
            raise RuntimeError(f"Minimization was not succesful.\n"
                               f"Status: {self._status}\n"
                               f"Message: {self._message}")

        self._params.values = opt_result.x
        self._fcn_min_val = opt_result.fun

        if get_hesse:
            self._hesse = ndt.Hessian(self._fcn)(self._params.values, *additional_args)
            self._hesse_inv = np.linalg.inv(self._hesse)
            self._params.covariance = self._hesse_inv
            self._params.correlation = cov2corr(self._hesse_inv)
            self._params.errors = np.sqrt(np.diag(self._hesse_inv))

    def fix_param(self, param_id):
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
                    "fun": lambda x: x[fixed_param] - initial_param_values[fixed_param]
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
            Parameters of `fcn` as np.ndarray of shape (`nparams`,).
        args : tuple
            Additional arguments for `fcn`.

        Returns
        -------
        np.ndarray
            Hesse matrix of `fcn` at point x. Shape is (`nparams`, `nparams`).
        """
        return ndt.Hessian(fcn)(x, *args)

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
        Shape is (`nparams`).
        """
        return self._params.values

    @property
    def param_errors(self):
        """np.ndarray: Estimated parameter values at the minimum of fcn.
        Shape is (`nparams`).
        """
        return self._params.errors

    @property
    def param_covariance(self):
        """np.ndarray: Estimated covariance matrix of the parameters.
        Calculated from the inverse of the Hesse matrix of fcn evaluated
        at it's minimum. Shape is (`nparams`, `nparams`).
        """
        return self._params.covariance

    @property
    def param_correlation(self):
        """np.ndarray: Estimated correlation matrix of the parameters.
         Shape is (`nparams`, `nparams`).
         """
        return self._params.correlation

    @property
    def hesse(self):
        """np.ndarray: Estimated Hesse matrix of fcn at it's minimum.
        Shape is (`nparams`, `nparams`).
        """
        return self._hesse

