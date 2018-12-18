"""
Template class for a binned likelihood fit.
"""
import numpy as np


class TemplateValueError(Exception):
    pass


def bin_centers(bin_edges):
    """
    Calculates the bin centers for given bin edges.
    """
    return (bin_edges[1:] + bin_edges[:-1]) / 2


def bin_width(bin_edges):
    """
    Calculates the bin width for given bin edges assuming
    equal binning.
    """
    return bin_edges[1] - bin_edges[0]


class Template:
    def __init__(self, name, df, variable, nbins, limits, weight="weight"):
        self._name = name
        self._variable = variable
        self._weight = weight

        self._nbins = nbins
        self._limits = limits
        self._bin_edges = np.linspace(*limits, nbins + 1)

        self._template, self._template_error_sq = self._create_template(df)

        self._mc_expectation = np.sum(self._template)

    def _create_template(self, df):
        """Calculates the template for a given pd.DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame which contains input data used for templates.
            Has to have columns specified by the give variable and weight
            name.
        """
        data = df[self._variable].values
        weights = df[self._weight].values

        # subtract minus one to get array indices instead of bin numbers
        bin_indices = np.digitize(data, self._bin_edges) - 1

        bin_count = []
        bin_error_sq = []
        for bin_index in range(self._nbins + 1):
            bin_count.append(np.sum(weights[np.where(bin_indices == bin_index)]))
            bin_error_sq.append(np.sum(weights[np.where(bin_indices == bin_index)] ** 2))

        return np.array(bin_count), np.array(bin_error_sq)

    @property
    def values(self):
        return self._template

    @property
    def errors(self):
        return np.sqrt(self._template_error_sq)

    @property
    def rel_errors(self):
        return np.divide(self.errors, self.values, out=np.full_like(self.errors, 1e-9), where=self.values != 0)


class TemplateCollection:

    def __init__(self, variable, nbins, limits, weight="weight"):
        self._variable = variable
        self._weight = weight
        self._nbins = nbins
        self._limits = limits

        self._template_map = dict()

    def add_template(self, name, df):
        self._template_map[name] = Template(name, df, self._variable, self._nbins, self._limits, self._weight)

    def templates(self):
        return self._template_map.items()