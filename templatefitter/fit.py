from scipy.optimize import minimize


class LikelihoodFitter:

    def __init__(self, nll):
        self._nll = nll

    def minimize(self, method='SLSQP'):
        fit_result = minimize(
            fun=self._nll,
            x0=self._nll.x0,
            method=method
        )

        return fit_result

class LikelihoodProfiler:
    pass

class ToyStudy:
    pass