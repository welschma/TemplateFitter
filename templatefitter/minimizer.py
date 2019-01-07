"""
Implementation of a minimizer class based on the scipy.optimize.minimize
function.
"""
import functools
import numpy as np
import numdifftools as ndt

from scipy.optimize import minimize

from templatefitter.utility import cov2corr


class Parameters:

    def __init__(self, names):
        self._names = names
        self._nparams = len(names)
        self._values = None
        self._errors = None
        self._covariance = None
        self._correlation = None

    @property
    def names(self):
        return self._names

    @property
    def nparams(self):
        return self._nparams

    @property
    def values(self):
        return self._values

    def get_param_value(self, name):
        param_index = self.names.index(name)
        return self.values[param_index]

    def get_param_error(self, name):
        param_index = self.names.index(name)
        return self.errors[param_index]

    def __getitem__(self, item):
        """Gets the value and error of the specified parameter.

        Parameters
        ----------
        item : int or str
            Parameter index or name.

        Returns
        -------
        float
            Parameter value.
        float
            Parameter error.
        """
        if isinstance(item, str):
            param_index = self.names.index(item)
        elif isinstance(item, int):
            param_index = item
        else:
            raise ValueError(
                "Specify the parameter either by its name (as str) or by its index (as int)."
            )
        return self.values[param_index], self.errors[param_index]

    @values.setter
    def values(self, new_values):
        if not len(new_values) == self.nparams:
            raise ValueError("Number of parameter values must be equal to number of parameters")
        self._values = new_values

    @property
    def errors(self):
        return self._errors

    @errors.setter
    def errors(self, new_errors):
        if not len(new_errors) == self.nparams:
            raise ValueError("Number of parameter errors must be equal to number of parameters")
        self._errors = new_errors

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, new_covariance):
        if not new_covariance.shape == (self.nparams, self.nparams):
            raise ValueError("New covariance matrix shape must be equal to (nparams, nparams)")
        self._covariance = new_covariance

    @property
    def correlation(self):
        return self._correlation

    @correlation.setter
    def correlation(self, new_correlation):
        if not new_correlation.shape == (self.nparams, self.nparams):
            raise ValueError("New correlation matrix shape must be equal to (nparams, nparams)")
        self._correlation = new_correlation


class Minimizer:
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
        constraints = self._create_constraints(initial_param_values, additional_args)

        opt_result = minimize(
            fun=self._fcn,
            x0=initial_param_values,
            args=additional_args,
            method="SLSQP",
            constraints=constraints
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
        if isinstance(param_id, int):
            param_index = param_id
        elif isinstance(param_id, str):
            param_index = self.params.names.index(param_id)
        else:
            raise ValueError(
                "Specify the parameter either by its name (as str) or by its index (as int)."
            )
        self._fixed_params.append(param_index)

    def _create_constraints(self, initial_param_values, args):
        constraints = list()

        for fixed_param in self._fixed_params:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: x[fixed_param] - initial_param_values[fixed_param],
                    "args": args
                }
            )

        return constraints

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def calculate_hesse_matrix(fcn, x, args):
        return ndt.Hessian(fcn)(x, *args)

    @property
    def fcn_min_val(self):
        return self._fcn_min_val

    @property
    def params(self):
        return self._params

    @property
    def param_values(self):
        return self._params.values

    @property
    def param_errors(self):
        return self._params.errors

    @property
    def param_covariance(self):
        return self._params.covariance

    @property
    def param_correlation(self):
        return self._params.correlation

    @property
    def hesse(self):
        return self._hesse

