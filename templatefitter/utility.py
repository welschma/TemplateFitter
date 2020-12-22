from itertools import islice

import numpy as np
from numba import jit, vectorize, float64, float32


__all__ = [
    "cov2corr",
    "corr2cov",
    "xlogyx",
    "get_systematic_cov_mat",
    "array_split_into",
]


def cov2corr(cov):
    """Calculates the correlation matrix from a given
    covariance matrix.

    Arguments
    ---------
    cov : np.ndarray
        Covariance matrix. Shape is (n,n).

    Return
    ------
    out : np.ndarray
        Correlation matrix. Shape is (n,n).
    """
    Dinv = np.nan_to_num(np.diag(1 / np.sqrt(np.diag(cov))))
    return np.matmul(Dinv, np.matmul(cov, Dinv))


def corr2cov(corr, var):
    """Calculates the covariance matrix from a given
    correlation matrix and a variance vector.

    Arguments
    ---------
    corr : np.ndarray
        Correlation matrix of shape (n,n).
    var : np.ndarray
        Variance vector of shape (n,).

    Return
    ------
    out : np.ndarray
        Covariance matrix. Shape is (n,n).
    """
    D = np.diag(var)
    return np.matmul(D, np.matmul(corr, D))


def id_to_index(names, param_id):
    """Returns the index of the parameter specified by `param_id`.
    If `param_id` is a string value in the `names` list, the index
    of the value in the list is returned.
    If `param_id` is an integer value, the same value is returned
    if its in the range of the `names` list.

    Parameters
    ----------
    names : list of str
        Parameter names.
    param_id : int or str
        Parameter index or name.

    Returns
    -------
    int
    """
    if isinstance(param_id, str) and (param_id in names):
        param_index = names.index(param_id)
    elif isinstance(param_id, int) and (param_id in range(len(names))):
        param_index = param_id
    else:
        raise ValueError(
            "Specify the parameter either by its name (as str) or by "
            "its index (as int)."
        )
    return param_index


# @jit(nopython=True, cache=True)
@vectorize([float32(float32, float32), float64(float64, float64)])
def xlogyx(x, y):
    """Compute :math:`x*log(y/x)`to a good precision when :math:`y~x`.
    The xlogyx function is taken from https://github.com/scikit-hep/probfit/blob/master/probfit/_libstat.pyx.
    """
    # result = np.where(x < y, x * np.log1p((y - x) / x), -x * np.log1p((x - y) / y))

    if np.isnan(x):
        return 0.0

    elif x < 1e-100:
        return 0.0

    elif x < y:
        return x * np.log1p((y - x) / x)
    else:
        return -x * np.log1p((x - y) / y)
    # result = np.zeros_like(x)
    #
    # for i in range(x.shape[0]):
    #     if x[i] < 1e-100:
    #         result[i] = 0
    #     elif x[i] < y[i]:
    #         result[i] = x[i] * np.log1p((y[i] - x[i])/x[i])
    #     else:
    #         result[i] = -x[i] * np.log1p((x[i] - y[i])/y[i])


def get_systematic_cov_mat(hnom, hup, hdown):
    """Calculates covariance matrix from systematic variations
    for a histogram.

    Returns
    -------
    Covariance Matrix : numpy.ndarray
        Shape is (`num_bins`, `num_bins`).
    """
    sign = np.ones_like(hup)
    mask = hup < hdown
    sign[mask] = -1

    diff_up = np.abs(hup - hnom)
    diff_down = np.abs(hdown - hnom)
    diff_sym = (diff_up + diff_down) / 2
    signed_diff = sign * diff_sym

    return np.outer(signed_diff, signed_diff)


def array_split_into(iterable, sizes):
    """Yields a list of arrays of size `n` from array iterable
    for each `n` in `sizes`.
    """

    itx = iter(iterable)

    for size in sizes:
        if size is None:
            yield np.array(list(itx))
            return
        else:
            yield np.array(list(islice(itx, size)))
