"""
Template class for a binned likelihood fit.
"""

import collections

import logging
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod, abstractproperty

from scipy.stats import poisson
from templatefitter import Histogram
from templatefitter.utility import cov2corr

logging.getLogger(__name__).addHandler(logging.NullHandler())

class TemplateParameter:
    KNOWN_ERROR_TYPES = ["sym", "asym"]

    def __init__(self, name, value=None, error=None, error_type="sym"):
        self._name = name
        self._value = value
        self._error = error
        self._error_type = error_type

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, new_error):
        self._error = new_error

    @property
    def error_type(self):
        return self._error_type

    @error_type.setter
    def error_type(self, new_error_type):
        if new_error_type.lower() in self.KNOWN_ERROR_TYPES:
            self._error_type = new_error_type
        else:
            raise ValueError("Unknown error type!\n"
                             f"Value can only be set to {self.KNOWN_ERROR_TYPES}")


class AbstractTemplateModel(ABC):

    def __init__(self, name, var_id, nbins, limits, df, weight_id="weight"):
        self._name = name
        self._vid = var_id
        self._wid = weight_id

        self._nbins = nbins
        self._limits = limits
        self._hist = Histogram(nbins, limits)
        self._hist.fill(df[var_id].values, df[weight_id].values)

        # set initial yield paramter value equal to sum of all
        # weights in template histogram (error equal to sqrt of
        # sum of # all weights squared)
        self._param_yield_value = np.sum(self._hist.bin_counts)
        self._param_yield_error = np.sum(self._hist.bin_errors_sq)

    @property
    def name(self):
        return self._name

    @property
    def variable(self):
        return self._vid

    @property
    def num_bins(self):
        return self._nbins

    @property
    def limits(self):
        return self._limits

    @property
    def bin_edges(self):
        return self._hist.bin_edges

    @property
    def bin_mids(self):
        return self._hist.bin_mids

    @property
    def bin_width(self):
        return self._hist.bin_width

    @property
    def yield_value(self):
        return self._param_yield_value

    @yield_value.setter
    def yield_value(self, new_value):
        self._param_yield_value = new_value

    def reset_yield_value(self):
        self.yield_value = np.sum(self._hist.bin_counts)

    @property
    def yield_error(self):
        return self._param_yield_error

    @yield_error.setter
    def yield_error(self, new_error):
        self._param_yield_error = new_error

    @property
    def values(self):
        return self.bin_fractions() * self._param_yield_value

    @abstractmethod
    def bin_fractions(self):
        pass


class SimpleTemplateModel(AbstractTemplateModel):

    def __init__(self, name, var_id, nbins, limits, df, weight_id="weight"):
        super().__init__(name, var_id, nbins, limits, df, weight_id)

    def bin_fractions(self):
        return self._hist.bin_counts / np.sum(self._hist.bin_counts)

    def plot_on(self, ax, **kwargs):
        ax.hist(self.bin_mids, weights=self.values, bins=self.bin_edges, **kwargs)


class AdvancedTemplateModel(AbstractTemplateModel):
    def __init__(self, name, var_id, nbins, limits, df, weight_id="weight"):

        super().__init__(name, var_id, nbins, limits, df, weight_id)

        # values and errors of nuissance parameter are np arrays
        # of shape (self.num_bins,). The pre fit value is zero.
        self._param_nuissance_values = np.zeros(self.num_bins)
        self._param_nuissance_errors = np.ones(self.num_bins)  # this error is meant in "standard deviations"

        # statistical covariance matrix as diagonal matrix of the
        # sum of weights squared per bin
        # the total covariance matrix is the sum of the statistical
        # covariance and any additional covariance matrix that has been
        # added
        self._cov_mat = np.diag(self._hist.bin_errors_sq)
        self._uncertainties = np.sqrt(np.diag(self._cov_mat))

    def add_cov_mat(self, cov_mat):
        if cov_mat.shape != self._cov_mat.shape:
            raise ValueError("Shape of given covariance matrix does not"
                             " match template shape.")
        self._cov_mat += cov_mat
        self._uncertainties = np.sqrt(np.diag(self._cov_mat))

    def plot_on(self, ax, **kwargs):
        ax.hist(self.bin_mids, weights=self.values, bins=self.bin_edges,
                edgecolor='black', histtype="stepfilled", **kwargs)
        ax.bar(x=self.bin_mids, height=2 * self.uncertainties, width=self.bin_width,
               bottom=self.values - self.uncertainties, color='black', hatch="///////",
               fill=False, lw=0)

    @property
    def nuissance_params_values(self):
        return self._param_nuissance_values

    @nuissance_params_values.setter
    def nuissance_params_values(self, new_values):
        if new_values.shape != self.nuissance_params_values.shape:
            raise ValueError("Shape of given nuissance parameter array"
                             " does not match template shape.")
        self._param_nuissance_values = new_values

    @property
    def nuissance_params_errors(self):
        return self._param_nuissance_errors

    @nuissance_params_errors.setter
    def nuissance_params_errors(self, new_errors):
        if new_errors.shape != self.nuissance_params_errors.shape:
            raise ValueError("Shape of given nuissance parameter array"
                             " does not match template shape.")
        self._param_nuissance_errors = new_errors

    @property
    def cov_mat(self):
        return self._cov_mat

    @property
    def corr_mat(self):
        return cov2corr(self.cov_mat)

    @property
    def uncertainties(self):
        """numpy.ndarray: Total uncertainty per bin. This value is the
        product of the relative uncertainty per bin and the current bin
        values. Shape is (`num_bins`,)."""
        return self.rel_uncertainties*self.values

    @property
    def rel_uncertainties(self):
        """numpy.ndarray: Relative uncertainty per bin. This value is fix.
        Shape is (`num_bins`,)."""
        return self._uncertainties/self._hist.bin_counts

    @property
    def values(self):
        return self.bin_fractions(self._param_nuissance_values) * self.yield_value

    def bin_fractions(self, nuissance_param_values):
        per_bin_yields = self._hist.bin_counts * (
                1 + nuissance_param_values * self.rel_uncertainties)
        return per_bin_yields / np.sum(per_bin_yields)


class NewCompositeTemplateModel:

    KNOWN_TEMPLATE_TYPES = ("simple", "advanced")

    def __init__(self, var_id, num_bins, limits, weight_id="weight"):
        self._vid = var_id
        self._wid = weight_id

        self._num_bins = num_bins
        self._limits = limits
        self._bin_edges = np.linspace(*limits, num_bins + 1)

        self._simple_templates = list()
        self._advanced_templates = list()
        self._templates = collections.OrderedDict()

    def add_template(self, template_id, template, template_type="advanced"):
        self._check_template_validity(template)
        self._templates[template_id] = template
        self._register_template(template_id, template_type)

    def create_template(self, template_id, df, weight_id="weight",
                        ttype="advanced"):
        logging.info(f"Creating template with id='{template_id}' "
                     f"and type='{ttype}'")
        if ttype.lower() == "simple":
            self._templates[template_id] = SimpleTemplateModel(
                template_id,
                self._vid,
                self._num_bins,
                self._limits,
                df=df,
                weight_id=weight_id
            )
        elif ttype.lower() == "advanced":
            self._templates[template_id] = AdvancedTemplateModel(
                template_id,
                self._vid,
                self._num_bins,
                self._limits,
                df=df,
                weight_id=weight_id
            )
        else:
            raise ValueError("Given template type is not compatible "
                             "with this collection.")
        self._register_template(ttype, template_id)

    def __getitem__(self, template_id):
        return self._templates[template_id]

    @property
    def num_bins(self):
        return self._num_bins

    @property
    def limits(self):
        return self._limits

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_mids(self):
        return (self.bin_edges[1:] + self.bin_edges[:-1])/2

    @property
    def bin_width(self):
        return self.bin_edges[1] - self.bin_edges[0]

    @property
    def num_advanced_templates(self):
        return len(self._advanced_templates)

    @property
    def num_simple_templates(self):
        return len(self._simple_templates)

    @property
    def num_templates(self):
        return self.num_simple_templates + self.num_advanced_templates

    def bin_fractions(self, nuissance_params):

        logging.debug(f"Calling 'bin_fractions' with nuissance parameters:\n"
                      f"{nuissance_params}")

        nuiss_params_per_template = iter(np.split(nuissance_params, self.num_advanced_templates))

        fractions_per_template = list()

        for tid, template in self._templates.items():
            if tid in self._advanced_templates:
                fractions_per_template.append(template.bin_fractions(next(nuiss_params_per_template)))
            else:
                fractions_per_template.append(template.bin_fractions())

        logging.debug(f"Bin fractions:\n"
                      f"{np.array(fractions_per_template)}")

        return np.array(fractions_per_template)

    def plot_on(self, ax, **kwargs):
        bin_mids = [self.bin_mids for _ in range(self.num_templates)]
        bin_counts = [template.values for template in self._templates.values()]
        labels = [template.name for template in self._templates.values()]
        ax.hist(bin_mids, weights=bin_counts, bins=self.bin_edges,
                edgecolor='black', histtype="stepfilled", lw=0.5,
                label=labels, stacked=True, **kwargs)

        uncertainties_sq = np.array([self._templates[tid].uncertainties**2 for tid in self._advanced_templates])
        total_uncertainty = np.sqrt(np.sum(uncertainties_sq, axis=0))
        logging.debug(f"{total_uncertainty}")
        total_bin_count = np.sum(np.array(bin_counts), axis=0)
        logging.debug(f"{total_bin_count}")

        ax.bar(x=bin_mids[0], height=2 * total_uncertainty, width=self.bin_width,
               bottom=total_bin_count - total_uncertainty, color='black', hatch="///////",
               fill=False, lw=0)

    def _register_template(self, ttype, tid):
        if ttype.lower() == "simple":
            self._simple_templates.append(tid)
        elif ttype.lower() == "advanced":
            self._advanced_templates.append(tid)
        else:
            raise ValueError("Given template type is not compatible "
                             "with this collection.")

    def _check_template_validity(self, template):

        eq_vid = (self._vid == template.variable)
        eq_num_bins = (self._num_bins == template.num_bins)
        eq_limits = (self._limits == template.limits)

        if not(eq_vid and eq_num_bins and eq_limits):
            raise ValueError("Given template is not compatible with"
                             " this collection.")
