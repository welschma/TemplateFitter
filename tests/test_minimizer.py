import unittest
import numpy as np

from templatefitter.minimizer import Minimizer

def rosen(x):
    """
    Rosebrock function.
    See https://en.wikipedia.org/wiki/Rosenbrock_function.

    Parameters
    ----------
    x : np.ndarray
        Variables of the Rosenbrock function as array.
        Shape is (2,).

    Returns
    -------
    float
    """
    a = 1.
    b = 100
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


class TestMinimizer(unittest.TestCase):

    def setUp(self):
        self.param_names = ["x", "y"]
        self.fcn_min_val = 0.0
        self.fcn_min_param_vals = np.array([1., 1.])
        self.minimizer = Minimizer(rosen, self.param_names)

    def test_init(self):
        self.assertListEqual(self.minimizer.params.names, self.param_names)
        self.assertEqual(self.minimizer.params.values, None)
        self.assertEqual(self.minimizer.params.errors, None)
        self.assertEqual(self.minimizer.params.covariance, None)
        self.assertEqual(self.minimizer.params.correlation, None)

    def test_minimize(self):
        self.minimizer.minimize([1.2, 1.1])

        self.assertAlmostEqual(self.minimizer.fcn_min_val, self.fcn_min_val, 5)

        params = self.minimizer.params

        self.assertAlmostEqual(params[self.param_names[0]][0], self.fcn_min_param_vals[0], 2)
        self.assertAlmostEqual(params[self.param_names[1]][0], self.fcn_min_param_vals[1], 2)

        self.assertTrue(isinstance(params.values, np.ndarray))
        np.testing.assert_array_almost_equal(
            params.values,
            self.fcn_min_param_vals,
            decimal=3,
        )

        self.assertTrue(params.errors.shape == (2,))
        self.assertTrue(params.covariance.shape == (2, 2))
        self.assertTrue(params.correlation.shape == (2, 2))

    def test_fix_param_by_string(self):
        self.minimizer.fix_param("x")
        self.minimizer.minimize([1.2, 1.1])
        self.assertEqual(self.minimizer.params["x"][0], 1.2)

    def test_fix_param_by_index(self):
        self.minimizer.fix_param(0)
        self.minimizer.minimize([1.2, 1.1])
        self.assertEqual(self.minimizer.params["x"][0], 1.2)
