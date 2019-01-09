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
        return self._uncertainties

    @property
    def values(self):
        return self.bin_fractions(self._param_nuissance_values) * self.yield_value

    def bin_fractions(self, nuissance_param_values):
        per_bin_yields = self._hist.bin_counts * (
                1 + nuissance_param_values * self.uncertainties)
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

    def bin_fractions(self, nuissance_params):

        logging.debug(f"Calling 'bin_fractions' with nuissance parameters:\n"
                      f"{nuissance_params}")

        nuiss_params_per_template = iter(np.split(nuissance_params, self.num_advanced_templates))

        fractions_per_template = list()

        for tid,template in self._templates.items():

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
        logging.info(f"{total_uncertainty}")
        total_bin_count = np.sum(np.array(bin_counts), axis=0)
        logging.info(f"{total_bin_count}")

        ax.bar(x=bin_mids[0], height=2 * total_uncertainty, width=self.bin_width,
               bottom=total_bin_count - total_uncertainty, color='black', hatch="///////",
               fill=False, lw=0)




# TODO add covariance matrices to templates
class TemplateModel:
    """Template provides attributes and methods to compute a
    binned likelihood function based on a Poisson model for each bin.

    Parameters
    ----------
    name : str
        Template name.
    variable : str
        Column name of the fit variable in the given pd.DataFrames.
    nbins : int
        Number of bins for the histogram.
    limits : tuple of float
        Lower and upper limit for the range of the histogram.
    df : pd.DataFrame, optional
        DataFrame which holds the column specified by `variable`
        (the default is None, which implies no initial addition
        of data).
    weight_key : str
        Key of the column in `df` which holds the event weights
        (the default is 'weight').

    Attributes
    ----------
    name : str
    values : np.ndarray
        Expected value of each bin of the template. Shape (nbins,).
    cov_mat : np.ndarray
        Covariance matrix of the template. This is the sum of the
        diagonal matrix constructed form the bin errors squared and
        all added covariance matrices.
    errors : np.ndarray
        Expected error of each bin of the template.
        This is the square root of the sum of squared event
        weights. Shape (nbins,).
    rel_errors : np.ndarray
        Expected relative error of each bin of the template.
        This is the square root of the sum of squared event
        weights in each bin divided by each bin count. Shape
        (nbins,).
    expected_yield : float
        Expected yield from the Template. This is weighted
        sum of all bin_counts of the added Monte Carlo samples.
    bin_edges : np.ndarray
        Bin edges of the underlying histogram. Shape (nbins + 1,).
    """

    def __init__(
            self,
            name,
            variable,
            nbins,
            limits,
            df=None,
            weight_key="weight"
    ):

        self._name = name
        self._variable = variable
        self._nbins = nbins
        self._limits = limits
        self._hist = Histogram(nbins, limits)
        self._weight_key = weight_key

        self._cov_mats = list()

        if df is not None:
            self.add_df(df)

    def add_df(self, df):
        """Fills template histogram with events from the given
        pd.DataFrame. The column of interest is given by the
        variable attribute and the event weight is specified
        by the weight_key attribute. If no column with name
        weight_key is found, a weight of 1.0 is assigned to
        each event.

        Arguments
        ---------
        df : pd.DataFrame
            A pd.DataFrame with events for this template.
        """
        data = df[self._variable]

        try:
            weights = df[self._weight_key]
        except KeyError:
            weights = np.ones_like(data)

        self._hist.fill(data, weights)

    def add_cov_mat(self, cov_mat):
        """Appends a covariance matrix to the list of covariance
        matrices for this template.

        Parameters
        ----------
        cov_mat : np.ndarray
            Covariance matrix for the template. Shape is expected
            to be (`nbins`, `nbins`).
        """
        self._cov_mats.append(cov_mat)

    def add_cov_mats(self, cov_mats):
        """Extends to the list of covariance matrices for this
        template with the given list of covariance matrices.

        Parameters
        ----------
        cov_mat : list of np.ndarray
            Covariance matrices for the template. Shape is expected
            to be (`nbins`, `nbins`).
        """
        self._cov_mats.extend(cov_mats)

    @property
    def name(self):
        return self._name

    @property
    def cov_matrix(self):
        stat_error = np.diag(self._hist.bin_errors_sq)

        for cov_mat in self._cov_mats:
            stat_error += cov_mat

        return stat_error

    @property
    def values(self):
        return self._hist.bin_counts

    @property
    def errors(self):
        return self._hist.bin_errors

    @property
    def rel_errors(self):
        return self._hist.bin_errors / self._hist.bin_counts

    @property
    def expected_yield(self):
        return np.sum(self._hist.bin_counts)

    @expected_yield.setter
    def expected_yield(self, new_yield):
        scale_factor = new_yield / self.expected_yield
        self._hist.scale(scale_factor)

    @property
    def bin_edges(self):
        return self._hist.bin_edges


class CompositeTemplateModel:
    """TemplateCollection is a container that creates and
    stores Template instances and acts as an interface to
    other classes or functions.

    Parameters
    ----------
    variable : str
        Column name of the fit variable in the pd.DataFrames to
        be used for the Template creation.
    nbins : int
        Number of bins for the Template histogram.
    limits : tuple of float
        Lower and upper limit for the range of the histogram.
    weight_key : str
        Key of the column in pd.DataFrames which holds the event
        weights (the default is 'weight').


    Attributes
    ----------
    variable : str
        Name of the variable which is used to create the
        Templates.
    bin_edges : np.ndarray
        The bin edges of the created Templates. Shape is (nbins+1,).
    bin_mids : np.ndarray
        The bin mids of the created Templates. Shape is (nbins,).
    values : np.ndarray
        Matrix of bin counts of the stored templates. Shape is
        (number of templates, nbins). The first row corresponds
        to the first added template, the second row to the second
        added template and so on.
    yields : np.ndarray
        Expected number of events per template. Shape is (n_templates,).
    rel_errors : np.ndarray
        Matrix of relative bin errors of the stored templates. Shape
        is (number of templates, nbins). The first row corresponds
        to the first added template, the second row to the second
        added template and so on.
    """

    def __init__(self, variable, nbins, limits, weight_key="weight"):
        self._variable = variable
        self._weight_key = weight_key
        self._nbins = nbins
        self._limits = limits
        self._bin_edges = np.linspace(*limits, nbins + 1)
        self._template_map = collections.OrderedDict()

    def add_template(self, name, df):
        """Creates a template labeled `name` from the given
        pd.DataFrame. The templates are stored in an internal map.

        Arguments
        ---------
        name : str
            Name for the template and also the key in the internal
            map storing the templates.
        df : pd.DataFrame
            A pd.DataFrame instance which has to contain at least a
            column name `variable` (specified in the constructor).
            If the DataFrame does not have a column identified by
            `weight_key`, a weight of 1.0 is assigned to each event.
        """
        self._template_map[name] = TemplateModel(
            name,
            self._variable,
            self._nbins,
            self._limits,
            df,
            self._weight_key)

    def set_yields(self, **kwargs):
        """Set expected number of events of stored templates to a
        new value. Arguments are passes as kwargs where the key has
        to match a name of a template in the collection.

        Arguments
        ---------
        kwargs
            E.g. signal=100, background=5000.
        """
        for template, value in kwargs.items():
            self._template_map[template].expected_yield = value

    def generate_toy_data(self):
        """Generates toy data using the poisson distribution.
        For each bin, a random number following a poisson distribution
        with mean equal to the expected number of events in this bin
        is generated.

        Returns
        -------
        np.ndarray
            Toy dataset. Shape is (nbins,)
        """
        expected_evts_per_bin = np.sum(self.values, axis=0)
        return poisson.rvs(expected_evts_per_bin)

    @property
    def yields(self):
        return np.sum(self.values, axis=1).reshape(-1)

    @property
    def variable(self):
        return self._variable

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_mids(self):
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    @property
    def values(self):
        return np.array([template.values for template in self._template_map.values()])

    @property
    def rel_errors(self):
        return np.array([template.rel_errors for template in self._template_map.values()])
