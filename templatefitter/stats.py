import numpy as np
from scipy.stats import chi2
from scipy.integrate import quad

__all__ = ["pearson_chi2_test", "cowan_binned_likelihood_gof"]


# -- goodness of fit statistics --


def pearson_chi2_test(data, expectation,  dof, error = None):
    """Performs a Pearson :math:`\chi^2`-test.
    This test reflects the level of agreement between observed
    and expected histograms.
    The test statistic is

    .. math::

        \chi^2=\sum\limits_{i=1}^{n_\mathrm{bins}} \\frac{(n_i - \\nu_i)^2}{\\nu_i},

    where :math:`n_i` is the number of observations in bin
    :math:`i` and :math:`\\nu_i` is the expected number of
    events in bin :math:`i`.

    In the large sample limits, this test statistic follows a
    :math:`\chi^2`-distribution with :math:`n_\mathrm{bins} - m`
    degrees of freedom, where :math:`m` is the number of unconstrained
    fit parameters.

    Parameters
    ----------
    data : np.ndarray
        Data bin counts. Shape is (`num_bins`,)
    expectation : np.ndarray
        Expected bin counts. Shape is (`num_bins`,)
    dof : int
        Degrees of freedom. This is the number of bins minus the
        number of free fit parameters.

    Returns
    -------
    float
        :math:`\chi^2/\mathrm{dof}`
    float
        p-value.
    """

    if error is not None:
        chi_sq = np.sum((data - expectation) ** 2 / error)
    else:
        chi_sq = np.sum((data - expectation) ** 2 / expectation)

    pval = quad(chi2.pdf, chi_sq, np.inf, args=(dof,))[0]
    return chi_sq, dof, pval


def cowan_binned_likelihood_gof(data, expectation, dof):
    """Performs a GOF-test using a test statistic based on a
    binned likelihood function.
    The test statistic is the ratio :math:`\lambda(\\nu) = L(\\nu=\hat{\\nu})/L(\\theta=n)`,
    where :math:`\\nu` are the expected values in each bin. In the
    numerator (denominator), the likelihood is evaluated with the estimated
    values for :math:`\\nu` (the measured values).

    In the large sample limit, the test statistic

    .. math::

        \chi^2 = -2\log \lambda = 2\sum\limits_{i=1}^{n_\mathrm{bins}} n_i\log(\\frac{n_i}{\hat{\\nu_i}}) - \hat{\\nu_i} - n_i,

    follows a :math:`\chi^2`-distribution with :math:`n_\mathrm{bins} - m`
    degrees of freedom, where :math:`m` is the number of unconstrained
    fit parameters.

    Parameters
    ----------
    data : np.ndarray
        Data bin counts. Shape is (`num_bins`,)
    expectation : np.ndarray
        Expected bin counts. Shape is (`num_bins`,)
    dof : int
        Degrees of freedom. This is the number of bins minus the
        number of free fit parameters.

    Returns
    -------
    float
        :math:`\chi^2/\mathrm{dof}`
    float
        p-value.
    """

    chi_sq = 2 * np.sum(data * np.log(data / expectation) + expectation - data)
    pval = quad(chi2.pdf, chi_sq, np.inf, args=(dof,))[0]
    return chi_sq, dof, pval
