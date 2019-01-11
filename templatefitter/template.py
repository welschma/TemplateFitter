"""
Template class for a binned likelihood fit.
"""

import collections

import logging
import numpy as np
import scipy.stats

from abc import ABC, abstractmethod, abstractproperty

from templatefitter import Histogram
from templatefitter.utility import cov2corr

__all__ = ["AbstractTemplate", "SimpleTemplate", "AdvancedTemplate",
           "AbstractCompositeTemplate", "AdvancedCompositeTemplate", ]

logging.getLogger(__name__).addHandler(logging.NullHandler())


class AbstractTemplate(ABC):
    """Abstract base class for template models. This class implements
    the minimal methods and properties expected of a template model.
    The template is based on histogram of a given variable which
    defines the shape of the template. By default, the template model
    keeps track of on parameter which is the total yield of the template.
    This parameter can be updated.

    Parameters
    ----------
    name : str
        Template name. Use as identify a template instance.
    var_id : str
        Identifier of the variable. Has to match a column in
        the given `pandas.DataFrame`.
    nbins : int
        Number of bins for the template histogram.
    limits : tuple of float
        Defines the lower and upper limit of the histogram.
    df : pandas.DataFrame
        A `pandas.DataFrame` instance. The column specified by
        `var_id` is used to construct the template histogram.
    weight_id : str, optional
        Optional string specifying the column name in `df` with
        the event weights. Default is 'weight'.
    """

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

    # -- public methods --

    def reset_yield_value(self):
        """Resets the yield parameter to it's initial value  which is the
        sum of weighted bin counts in the template histogram.
        """
        self._param_yield_value = np.sum(self._hist.bin_counts)

    # -- properties --

    @property
    def name(self):
        """str: Template name."""
        return self._name

    @property
    def variable(self):
        """str: Variable name."""
        return self._vid

    @property
    def num_bins(self):
        """int: Number of bins in the template histogram."""
        return self._nbins

    @property
    def limits(self):
        """tuple of float: Lower and upper histogram limits."""
        return self._limits

    @property
    def bin_edges(self):
        """numpy.ndarray: Bin edges of the template histogram.
        Shape is (`num_bins`,)."""
        return self._hist.bin_edges

    @property
    def bin_mids(self):
        """numpy.ndarray: Bin mids of the template histogram.
        Shape is (`num_bins`,)."""
        return self._hist.bin_mids

    @property
    def bin_width(self):
        """float: Bin width of the template histogram"""
        return self._hist.bin_width

    @property
    def yield_value(self):
        """float: Total yield of the template. By default this is the
        sum of weighted bin counts in the template histogram."""
        return self._param_yield_value

    @yield_value.setter
    def yield_value(self, new_value):
        self._param_yield_value = new_value

    @property
    def yield_error(self):
        """float: Error of the yield parameter. By default it is not set
        in the beginning and has to be set manually."""
        return self._param_yield_error

    @yield_error.setter
    def yield_error(self, new_error):
        self._param_yield_error = new_error

    @property
    def values(self):
        """numpy.ndarray: Current values of the template per bin. This is
        the product of the `bin_fractions` and the current value of the
        yield parameter."""
        return self.bin_fractions() * self._param_yield_value

    # -- abstract methods --

    @abstractmethod
    def bin_fractions(self):
        """Abstract method which, when implemented, will return the
        bin fractions of the template.
        """
        pass

    @abstractmethod
    def plot_on(self):
        """Plots the template as histogram on a given axis.
        """
        pass


class SimpleTemplate(AbstractTemplate):
    """This class implements an simple template model. This means that
    the only parameter of this model is the `yield`.

    The `bin_fractions` method does not depend on nuissance paramenters
    and therefore does not incorporate systematic errors.

    Parameters
    ----------
    name : str
        Template name. Use as identify a template instance.
    var_id : str
        Identifier of the variable. Has to match a column in
        the given `pandas.DataFrame`.
    nbins : int
        Number of bins for the template histogram.
    limits : tuple of float
        Defines the lower and upper limit of the histogram.
    df : pandas.DataFrame
        A `pandas.DataFrame` instance. The column specified by
        `var_id` is used to construct the template histogram.
    weight_id : str, optional
        Optional string specifying the column name in `df` with
        the event weights. Default is 'weight'.
    """

    def __init__(self, name, var_id, nbins, limits, df, weight_id="weight"):
        super().__init__(name, var_id, nbins, limits, df, weight_id)

    def bin_fractions(self):
        """Calculates the per bin fraction :math:`f_i` of the template.
        This value is used to calculate the expected number of events
        per bin :math:`\\nu_i` as :math:`\\nu_i=f_i\cdot\\nu`, where
        :math:`\\nu` is the expected yield.

        Returns
        -------
        numpy.ndarray
            Bin fractions of this template. Shape is (`num_bins`,).
        """
        return self._hist.bin_counts / np.sum(self._hist.bin_counts)

    def plot_on(self, ax, **kwargs):
        """Plots the template as histogram on a given axis.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            An instance of a matplotlib axis.
        **kwargs
            Additional keyword arguments used to change the plot.
        """
        ax.hist(self.bin_mids, weights=self.values, bins=self.bin_edges,
                label=self.name, **kwargs)


class AdvancedTemplate(AbstractTemplate):
    """This class implements an advanced template model. This
    means that the template model has a yield and nuissance
    parameters. The nuissance parameters allow to bin of the
    template fo vary inside their uncertainties.

    The `bin_fractions` method does depend on these nuissance
    paramenters and therefore does incorporate systematic errors.
    By default the only accounted systematic error is the statistical
    uncertainty from the template histgoram. Any additional uncertainties
    have to be added as covariance matrices.


    Parameters
    ----------
    name : str
        Template name. Use as identify a template instance.
    var_id : str
        Identifier of the variable. Has to match a column in
        the given `pandas.DataFrame`.
    nbins : int
        Number of bins for the template histogram.
    limits : tuple of float
        Defines the lower and upper limit of the histogram.
    df : pandas.DataFrame
        A `pandas.DataFrame` instance. The column specified by
        `var_id` is used to construct the template histogram.
    weight_id : str, optional
        Optional string specifying the column name in `df` with
        the event weights. Default is 'weight'.
    """

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
        hist_errors = np.copy(self._hist.bin_errors_sq)
        hist_errors[hist_errors == 0] = 1e-14
        self._cov_mat = np.diag(hist_errors)
        self._corr_mat = np.diag(np.ones(self.num_bins))
        self._inv_corr_mat = np.diag(np.ones(self.num_bins))
        self._uncertainties = np.sqrt(np.diag(self._cov_mat))

    def add_cov_mat(self, cov_mat):
        """Add a covariance matrix for a systematic error to this template.
        This updates the total covariance matrix.

        Parameters
        ----------
        cov_mat : numpy.ndarray
            A covariance matrix. It is not checked if the matrix is
            valid (symmetric, positive semi-definite.
            Shape is (`num_bins`, `num_bins`)
        """
        if cov_mat.shape != self._cov_mat.shape:
            raise ValueError("Shape of given covariance matrix does not"
                             " match template shape.")
        self._cov_mat += cov_mat
        self._corr_mat = cov2corr(self._cov_mat)
        self._inv_corr_mat = np.linalg.inv(self._corr_mat)
        self._uncertainties = np.sqrt(np.diag(self._cov_mat))

    def plot_on(self, ax, **kwargs):
        """Plots the template as histogram on a given axis. Also the
        uncertainty is plotted as hatched bars.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            An instance of a matplotlib axis.
        **kwargs
            Additional keyword arguments used to change the plot.
        """
        ax.hist(self.bin_mids, weights=self.values, bins=self.bin_edges,
                edgecolor='black', histtype="stepfilled", **kwargs)
        ax.bar(x=self.bin_mids, height=2 * self.uncertainties, width=self.bin_width,
               bottom=self.values - self.uncertainties, color='black', hatch="///////",
               fill=False, lw=0)

    @property
    def nuissance_params_values(self):
        """numpy.ndarray: Current values of the nuissance parameters.
        If not set manually, the default values are equal to zero."""
        return self._param_nuissance_values

    @nuissance_params_values.setter
    def nuissance_params_values(self, new_values):
        logging.debug(new_values.shape)
        logging.debug(self.nuissance_params_values.shape)
        if new_values.shape != self.nuissance_params_values.shape:
            raise ValueError("Shape of given nuissance parameter array"
                             " does not match template shape.")
        self._param_nuissance_values = new_values

    @property
    def nuissance_params_errors(self):
        """numpy.ndarray: Current error of the nuissance parameters.
        These are give in standard deviations of the error they
        represent. If not set manually, the default values are equal
        to one."""
        return self._param_nuissance_errors

    @nuissance_params_errors.setter
    def nuissance_params_errors(self, new_errors):
        if new_errors.shape != self.nuissance_params_errors.shape:
            raise ValueError("Shape of given nuissance parameter array"
                             " does not match template shape.")
        self._param_nuissance_errors = new_errors

    @property
    def cov_mat(self):
        """numpy.ndarray: The covariance matrix of the template errors.
        Shape is (`num_bins`, `num_bins`)."""
        return self._cov_mat

    @property
    def corr_mat(self):
        """numpy.ndarray: The correlation matrix of the template errors.
        Shape is (`num_bins`, `num_bins`)."""
        return self._corr_mat

    @property
    def inv_corr_mat(self):
        """numpy.ndarray: The invers correlation matrix of the
        template errors. Shape is (`num_bins`, `num_bins`)."""
        return self._inv_corr_mat

    @property
    def uncertainties(self):
        """numpy.ndarray: Total uncertainty per bin. This value is the
        product of the relative uncertainty per bin and the current bin
        values. Shape is (`num_bins`,)."""
        return self.rel_uncertainties * self.values

    @property
    def rel_uncertainties(self):
        """numpy.ndarray: Relative uncertainty per bin. This value is fix.
        Shape is (`num_bins`,)."""
        rel_uncertainties = np.divide(self._uncertainties, self._hist.bin_counts,
                                      out=np.full_like(self._uncertainties, 1e-7),
                                      where=self._hist.bin_counts != 0)
        return rel_uncertainties

    @property
    def values(self):
        return self.bin_fractions(self._param_nuissance_values) * self.yield_value

    def bin_fractions(self, nuissance_parameters):
        """
        Calculates the per bin fraction :math:`f_i` of the template.
        This value is used to calculate the expected number of events
        per bin :math:`\\nu_i` as :math:`\\nu_i=f_i\cdot\\nu`, where
        :math:`\\nu` is the expected yield. The fractions are given as

        .. math::

            f_i=\sum\limits_{i=1}^{n_\mathrm{bins}} \\frac{\\nu_i(1+\\theta_i\cdot\epsilon_i)}{\sum_{j=1}^{n_\mathrm{bins}} \\nu_j (1+\\theta_j\cdot\epsilon_j)},

        where :math:`\\theta_j` are the nuissance parameters and
        :math:`\epsilon_j` are the relative uncertainties per bin.

        Parameters
        ----------
        nuissance_parameters : numpy.ndarray
            An array with values for the nuissance parameters.
            Shape is (`num_bins`,)

        Returns
        -------
        numpy.ndarray
            Bin fractions of this template. Shape is (`num_bins`,).
        """
        per_bin_yields = self._hist.bin_counts * (
                1 + nuissance_parameters * self.rel_uncertainties)
        return np.nan_to_num(per_bin_yields / np.sum(per_bin_yields))


class AbstractCompositeTemplate(ABC):
    """A CompositeTemplateModel combines several template models
    into one. It can create and store template model instances.
    All templates in this model have the same underlying histogram
    properties.

    This container is useful to create Likelihood functions when
    performing template fits, since it can calculate the per bin
    fractions of all template as array (of shape
    (`num_templates`, `num_bins`)). This array is then used to
    calculate the expected number of events per bin.

    Parameters
    ----------
    var_id : str
        Identifier of the variable. Has to match a column in
        the given `pandas.DataFrame`.
    nbins : int
        Number of bins for the template histogram.
    limits : tuple of float
        Defines the lower and upper limit of the histogram.
    weight_id : str, optional
        Optional string specifying the column name in `df` with
        the event weights. Default is 'weight'.
    """

    def __init__(self, var_id, num_bins, limits, weight_id="weight"):
        self._vid = var_id
        self._wid = weight_id

        self._num_bins = num_bins
        self._limits = limits
        self._bin_edges = np.linspace(*limits, num_bins + 1)

        self._templates = collections.OrderedDict()

    # -- public methods --

    def add_template(self, tid, template):
        """Adds an instance of an SimpleTemplateModel or AdvancedTemplateModel
        to the container.

        Parameters
        ----------
        tid : str
            Id for the template which is used as key in the internal
            map which stores the templates.
        template : SimpleTemplate or AdvancedTemplate
            Instance of a template model.
        ttype : str, optional
            Specifies the template type. Possible values are 'simple'
            and 'advanced'. Default is 'advanced'.
        """
        self._check_template_validity(template)
        self._templates[tid] = template

    def set_yield(self, tid, value):
        """Sets a new yield value for template with id=`tid`.

        Parameters
        ----------
        tid : str
            Id for the template which is used as key in the internal
            map which stores the templates.
        value : float
            New yield value.
        """
        self._templates[tid].yield_value = value

    def reset_yield_values(self):
        """Sets all yield values to the original values.
        """
        for template in self._templates.values():
            template.reset_yield_value()

    def generate_asimov_dataset(self):
        """Generates an Asimov dataset from the given templates.
        This is a binned dataset which corresponds to the current expectation values.
        Since data takes only integer values, the template
        expectation in each bin is rounded to the nearest integer

        Returns
        -------
        asimov_dataset : numpy.ndarray
            Shape is (`num_bins`)
        """
        return np.rint(self.bin_counts)

    def generate_toy_dataset(self):
        """Generates a toy dataset from the given templates.
        This is a binned dataset where each bin is treated a
        random number following a poisson distribution with
        mean equal to the bin content of all templates.

        Returns
        -------
        toy_dataset : numpy.ndarray
            Shape is (`num_bins`)
        """
        return scipy.stats.poisson.rvs(self.bin_counts)

    # -- magic methods --

    def __getitem__(self, tid):
        """SimpleTemplateModel or AdvancedTemplateModel: The template
        stored with id `tid`"""
        return self._templates[tid]

    # -- private methods --

    def _check_template_validity(self, template):
        """Checks, if the given template is compatible with
        this container.

        Parameters
        ----------
        template : AdvancedTemplate or SimpleTemplate
            A template model instance.

        Raises
        ------
        ValueError
        """

        eq_vid = (self._vid == template.variable)
        eq_num_bins = (self._num_bins == template.num_bins)
        eq_limits = (self._limits == template.limits)

        if not (eq_vid and eq_num_bins and eq_limits):
            raise ValueError("Given template is not compatible with"
                             " this collection.")

    # -- properties --

    @property
    def num_bins(self):
        """int: Number of bins of the templates in this model."""
        return self._num_bins

    @property
    def limits(self):
        """tuple of float: Lower and upper limit of the templates in
        this model."""
        return self._limits

    @property
    def bin_edges(self):
        """numpy.ndarray: Bin edges of the templates in this model."""
        return self._bin_edges

    @property
    def bin_mids(self):
        """numpy.ndarray: Bin mids of the templates in this model."""
        return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

    @property
    def bin_width(self):
        """float: Bin width of the templates in this model."""
        return self.bin_edges[1] - self.bin_edges[0]

    @property
    def bin_counts(self):
        """numpy.ndarray: Sum over all template values in each
        bin (bin counts of the composite template)."""
        return np.sum(
            np.array([template.values for template in self._templates.values()]),
            axis=0
        )

    @property
    def num_templates(self):
        """int: Number templates in this model."""
        return len(self._templates)

    @property
    def yield_values(self):
        """numpy.ndarray: An array with current yield values
        of all templates."""
        return np.array([template.yield_value for template in self._templates.values()])

    @yield_values.setter
    def yield_values(self, new_yields):
        if len(self.yield_values) != len(new_yields):
            raise ValueError("You have to supply as many new yield "
                             "values as there are templates in this"
                             " container.")

        for template, new_yield in zip(
                self._templates.values(), iter(new_yields)):
            template.yield_value = new_yield

    @property
    def template_ids(self):
        """list of str: List of all template names."""
        return [template.name for template in self._templates.values()]

    # -- abstract methods --

    @abstractmethod
    def bin_fractions(self):
        """Evaluates all `bin_fractions` methods of all templates in this
        container.

        Returns
        -------
        numpy.ndarray
            A 2D array of bin fractions. The first axis represents the
            templates in this container and the second axis represents
            the bins of each template.
            Shape is (`num_templates`, `num_bins`).
        """

        pass

    @abstractmethod
    def create_template(self, tid, df, weight_id="weight"):
        """Adds an instance of an SimpleTemplateModel or AdvancedTemplateModel
        to the container.

        Parameters
        ----------
        tid : str
            Id for the template which is used as key in the internal
            map which stores the templates.
        df : pandas.DataFrame
            A `pandas.DataFrame` instance. The column specified by
            `var_id` is used to construct the template histogram.
        weight_id : str, optional
            Optional string specifying the column name in `df` with
            the event weights. Default is 'weight'.
        """

        pass

    @abstractmethod
    def plot_on(self, ax, **kwargs):
        """Plots the templates as stacked histogram on a given
        axis. Also the total uncertainty is plotted as hatched bars.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            An instance of a matplotlib axis.
        **kwargs
            Additional keyword arguments used to change the plot.
        """
        pass

    @abstractmethod
    def update_parameters(self, new_parameters):
        pass


class SimpleCompositeTemplate(AbstractCompositeTemplate):
    """An SimpleCompositeTemplateModel combines several instances
    of SimpleTemplate models into one. It can create and store
    template model instances. All templates in this model have the
    same underlying histogram properties.

    This container is useful to create Likelihood functions when
    performing template fits, since it can calculate the per bin
    fractions of all template as array (of shape
    (`num_templates`, `num_bins`)). This array is then used to
    calculate the expected number of events per bin.

    Parameters
    ----------
    var_id : str
        Identifier of the variable. Has to match a column in
        the given `pandas.DataFrame`.
    nbins : int
        Number of bins for the template histogram.
    limits : tuple of float
        Defines the lower and upper limit of the histogram.
    weight_id : str, optional
        Optional string specifying the column name in `df` with
        the event weights. Default is 'weight'.
    """

    def __init__(self, var_id, num_bins, limits, weight_id="weight"):
        super().__init__(var_id, num_bins, limits, weight_id)

    def create_template(self, tid, df, weight_id="weight"):
        """Adds an instance of an AdvancedTemplateModel to the
        container.

        Parameters
        ----------
        tid : str
            Id for the template which is used as key in the internal
            map which stores the templates.
        df : pandas.DataFrame
            A `pandas.DataFrame` instance. The column specified by
            `var_id` is used to construct the template histogram.
        weight_id : str, optional
            Optional string specifying the column name in `df` with
            the event weights. Default is 'weight'.
        """
        logging.info(f"Creating template with id='{tid}'")
        self._templates[tid] = SimpleTemplate(
            tid,
            self._vid,
            self._num_bins,
            self._limits,
            df=df,
            weight_id=weight_id
        )

    def bin_fractions(self):
        """Evaluates all `bin_fractions` methods of all templates in this
        container.


        Returns
        -------
        numpy.ndarray
            A 2D array of bin fractions. The first axis represents the
            templates in this container and the second axis represents
            the bins of each template.
            Shape is (`num_templates`, `num_bins`).
        """
        fractions_per_template = np.array([
            template.bin_fractions() for template in self._templates.values()
        ])

        return fractions_per_template

    def plot_on(self, ax, **kwargs):
        """Plots the templates as stacked histogram on a given
        axis. Also the total uncertainty is plotted as hatched bars.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            An instance of a matplotlib axis.
        **kwargs
            Additional keyword arguments used to change the plot.
        """
        bin_mids = [self.bin_mids for _ in range(self.num_templates)]
        bin_counts = [template.values for template in self._templates.values()]
        labels = [template.name for template in self._templates.values()]
        ax.hist(bin_mids, weights=bin_counts, bins=self.bin_edges,
                edgecolor='black', histtype="stepfilled", lw=0.5,
                label=labels, stacked=True, **kwargs)

    def update_parameters(self, new_parameters):
        """Updates all template yields.

        Parameters
        ----------
        new_parameters : np.ndarray
            New yield values. Shape is (`num_templates`)
        """
        self.yield_values = new_parameters


class AdvancedCompositeTemplate(AbstractCompositeTemplate):
    """An AdvancedCompositeTemplateModel combines several template
    models into one. It can create and store template model instances.
    All templates in this model have the same underlying histogram
    properties.

    This container is useful to create Likelihood functions when
    performing template fits, since it can calculate the per bin
    fractions of all template as array (of shape
    (`num_templates`, `num_bins`)). This array is then used to
    calculate the expected number of events per bin for a given
    parameter set which includes the different template yields as
    well as the nuissance parameters.

    Parameters
    ----------
    var_id : str
        Identifier of the variable. Has to match a column in
        the given `pandas.DataFrame`.
    nbins : int
        Number of bins for the template histogram.
    limits : tuple of float
        Defines the lower and upper limit of the histogram.
    weight_id : str, optional
        Optional string specifying the column name in `df` with
        the event weights. Default is 'weight'.
    """

    def __init__(self, var_id, num_bins, limits, weight_id="weight"):
        super().__init__(var_id, num_bins, limits, weight_id)

    @property
    def num_nuissance_params(self):
        """int: Number of nuissance parameters."""
        return self.num_bins*self.num_templates

    @property
    def nuissance_params_values(self):
        """numpy.ndarray: An array with current nuissance parameter values
        of all templates."""
        return np.concatenate([template.nuissance_params_values for template in self._templates.values()])

    @nuissance_params_values.setter
    def nuissance_params_values(self, new_values):
        if self.num_nuissance_params != new_values.shape[0]:
            raise ValueError("You have to supply as many new yield "
                             "values as there are templates in this"
                             " container.")

        for template, new_value in zip(
                self._templates.values(), np.split(new_values, self.num_templates)):
            template.nuissance_params_values = new_value

    @property
    def inv_corr_mats(self):
        """list of numpy.ndarray: A list of inverse correlation
        matrices for all templates."""
        return [template.inv_corr_mat for template in self._templates.values()]

    def set_nuissance_params(self, tid, values):
        """Sets new values for the nuissance parameters of template
         with id=`tid`.

        Parameters
        ----------
        tid : str
            Id for the template which is used as key in the internal
            map which stores the templates.
        values : numpy.ndarray
            New yield value.
        """
        self._templates[tid].nuissance_params_values = values

    def create_template(self, tid, df, weight_id="weight"):
        """Adds an instance of an AdvancedTemplateModel to the
        container.

        Parameters
        ----------
        tid : str
            Id for the template which is used as key in the internal
            map which stores the templates.
        df : pandas.DataFrame
            A `pandas.DataFrame` instance. The column specified by
            `var_id` is used to construct the template histogram.
        weight_id : str, optional
            Optional string specifying the column name in `df` with
            the event weights. Default is 'weight'.
        """
        logging.info(f"Creating template with id='{tid}'")
        self._templates[tid] = AdvancedTemplate(
            tid,
            self._vid,
            self._num_bins,
            self._limits,
            df=df,
            weight_id=weight_id
        )

    def bin_fractions(self, nuissance_params):
        """Evaluates all `bin_fractions` methods of all templates in this
        container. Here, the bin fractions depend on so called nuissance
        parameters which incorporate uncertainties on the template shape.

        Parameters
        ----------
        nuissance_params : numpy.ndarray
            Array of nuissance parameter values needed for the evaluation
            of the AdvancedTemplateModel `bin_fraction` method.

        Returns
        -------
        numpy.ndarray
            A 2D array of bin fractions. The first axis represents the
            templates in this container and the second axis represents
            the bins of each template.
            Shape is (`num_templates`, `num_bins`).
        """

        nuiss_params_per_template = iter(np.split(nuissance_params, self.num_templates))

        fractions_per_template = np.array([
            template.bin_fractions(next(nuiss_params_per_template)) for template in self._templates.values()
        ])

        return fractions_per_template

    def plot_on(self, ax, **kwargs):
        """Plots the templates as stacked histogram on a given
        axis. Also the total uncertainty is plotted as hatched bars.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            An instance of a matplotlib axis.
        **kwargs
            Additional keyword arguments used to change the plot.
        """
        bin_mids = [self.bin_mids for _ in range(self.num_templates)]
        bin_counts = [template.values for template in self._templates.values()]
        labels = [template.name for template in self._templates.values()]
        ax.hist(bin_mids, weights=bin_counts, bins=self.bin_edges,
                edgecolor='black', histtype="stepfilled", lw=0.5,
                label=labels, stacked=True, **kwargs)

        uncertainties_sq = np.array(
            [template.uncertainties ** 2 for template in self._templates.values()])
        total_uncertainty = np.sqrt(np.sum(uncertainties_sq, axis=0))
        total_bin_count = np.sum(np.array(bin_counts), axis=0)

        ax.bar(x=bin_mids[0], height=2 * total_uncertainty, width=self.bin_width,
               bottom=total_bin_count - total_uncertainty, color='black', hatch="///////",
               fill=False, lw=0)

    def update_parameters(self, new_parameters):
        """Updates all template yields and nuissance parameters.

        Parameters
        ----------
        new_parameters : np.ndarray
            New yield values. Shape is
            (`num_templates` + `num_templates`*`num_bins`,).
        """
        yields = new_parameters[:self.num_templates]

        nuissance_params = new_parameters[self.num_templates:]
        logging.debug(nuissance_params.shape)
        logging.debug(self.num_nuissance_params)
        self.yield_values = yields
        self.nuissance_params_values = nuissance_params
