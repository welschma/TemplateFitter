import numpy as np

__all__ = [
    "cov2corr",
    "corr2cov"
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
    Dinv = np.diag(1 / np.sqrt(np.diag(cov)))
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
