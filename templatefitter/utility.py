import numpy as np

__all__ = ["cov2corr", "corr2cov"]


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
