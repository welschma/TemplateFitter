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
    pass