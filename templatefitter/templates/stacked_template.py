import logging

from collections import OrderedDict

import numpy as np
import scipy.stats

from templatefitter.histogram import Hist1d
from templatefitter.templates import AbstractTemplate
from templatefitter.nll import StackedTemplateNegLogLikelihood

logging.getLogger(__name__).addHandler(logging.NullHandler())


class StackedTemplate(AbstractTemplate):
    """
    """

    def __init__(self, name, variable, num_bins, limits):
        super().__init__(name)
        self._variable = variable
        self._limits = limits

        self._num_bins = num_bins
        self._bin_edges = np.linspace(*limits, num_bins + 1)
        self._bin_mids = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        self._bin_width = self._bin_edges[1] - self._bin_edges[0]

        self._template_dict = OrderedDict()

    def add_template(self, tid, template):
        """Adds an instance of an Template to the container.

        Parameters
        ----------
        tid : str
            Id for the template which is used as key in the internal
            map which stores the templates.
        template : Template
            Instance of a template model.

        Raises
        ------
        ValueError
            If the given template is not compatible with the container.
        """
        self._check_template_validity(template)
        self._template_dict[tid] = template

    def create_template(self, tid, df, weight="weight"):
        """Creates an instance of Template and adds it to the
        container.

        Parameters
        ----------
        tid : str
            Id for the template which is used as key in the internal
            map which stores the templates.
        df : pandas.DataFrame
            A `pandas.DataFrame` instance. The column specified by
            `var_id` is used to construct the template histogram.
        weight : str, optional
            Optional string specifying the column name in `df` with
            the event weights. Default is 'weight'.
        """
        self._template_dict[tid] = Template(tid, self._variable, self.num_bins,
                                            self._limits, df, weight)

    def _check_template_validity(self, template):
        """Checks, if the given template is compatible with
        this container.

        Parameters
        ----------
        template : Template
            A template model instance.

        Raises
        ------
        ValueError
        """

        eq_vid = self._variable == template.variable
        eq_num_bins = self._num_bins == template.num_bins
        eq_limits = self._limits == template.limits

        if not (eq_vid and eq_num_bins and eq_limits):
            raise ValueError(
                "Given template is not compatible with this collection.")

    def errors(self):
        """numpy.ndarray: Sum over all template errors squared
        in each bin (bin errors of the stacked template).
        """
        errors = np.sqrt(
            np.sum(
                np.array([
                    template.errors()**2
                    for template in self._template_dict.values()
                ]),
                axis=0,
            ))
        return errors

    def values(self):
        """numpy.ndarray: Sum over all template values in each
        bin (bin counts of the stacked template).
        """
        return np.sum(
            np.array([
                template.values() for template in self._template_dict.values()
            ]),
            axis=0,
        )

    def fractions(self, nuiss_params):
        """Evaluates all `bin_fractions` methods of all templates in this
        container. Here, the bin fractions depend on so called nuissance
        parameters which incorporate uncertainties on the template shape.

        Parameters
        ----------
        nuiss_params:  numpy.ndarray
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
        nui_params_per_template = np.split(nuiss_params, self.num_templates)
        fractions_per_template = np.array([
            template.fractions(nui_params) for template, nui_params in zip(
                self._template_dict.values(), nui_params_per_template)
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
        bin_counts = [
            template.values() for template in self._template_dict.values()
        ]
        labels = [template.name for template in self._template_dict.values()]
        ax.hist(
            bin_mids,
            weights=bin_counts,
            bins=self.bin_edges,
            edgecolor="black",
            histtype="stepfilled",
            lw=0.5,
            label=labels,
            stacked=True,
            **kwargs,
        )

        uncertainties_sq = np.array(
            [template.errors()**2 for template in self._template_dict.values()])
        total_uncertainty = np.sqrt(np.sum(uncertainties_sq, axis=0))
        total_bin_count = np.sum(np.array(bin_counts), axis=0)

        ax.bar(
            x=bin_mids[0],
            height=2 * total_uncertainty,
            width=self.bin_width,
            bottom=total_bin_count - total_uncertainty,
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
        )

    def generate_asimov_dataset(self, integer_values=False):
        """Generates an Asimov dataset from the given templates.
        This is a binned dataset which corresponds to the current
        expectation values. Since data takes only integer values,
        the template expectation in each bin is rounded to the
        nearest integer.

        Parameters
        ----------
        integer_values : bool, optional
            Wether to round Asimov data points to integer values
            or not. Default is False.

        Returns
        -------
        asimov_dataset : Hist1d
        """
        if integer_values:
            asimov_bin_counts = np.rint(self.values())
        else:
            asimov_bin_counts = self.values()
        return Hist1d.from_binned_data(self.bin_edges, asimov_bin_counts)

    def generate_toy_dataset(self):
        """Generates a toy dataset from the given templates.
        This is a binned dataset where each bin is treated a
        random number following a poisson distribution with
        mean equal to the bin content of all templates.

        Returns
        -------
        toy_dataset : Hist1d
        """
        toy_bin_counts = scipy.stats.poisson.rvs(self.values())
        return Hist1d.from_binned_data(self.bin_edges, toy_bin_counts)

    def reset_parameters(self):
        """Sets all parameters of all templates to their original
        values.
        """
        for template in self._template_dict.values():
            template.reset_parameters()

    def create_nll(self, dataset):
        """Creates a negative log likelihood object from the

        Parameters
        ----------
        dataset : Hist1d
            Binned dataset in the form of a histogram object

        Returns
        -------
        StackedTemplateNegLogLikelihood
        """

        return StackedTemplateNegLogLikelihood(dataset, self)

    def update_parameters(self, new_parameters, new_errors):
        """Updates all template yields and nuissance parameters.

        Parameters
        ----------
        new_parameters : np.ndarray
            New yield and nuissance parameter values. Shape is
            (`num_templates` + `num_templates`*`num_bins`,).
        new_errors : np.ndarray
            New yield and nuissance parameter errors. Shape is
            (`num_templates` + `num_templates`*`num_bins`,).
        """
        yields = new_parameters[:self.num_templates]
        nuiss_params = new_parameters[self.num_templates:]
        yield_errors = new_errors[:self.num_templates]
        nuiss_param_errors = new_errors[self.num_templates:]

        self.yield_param_values = yields
        self.nui_param_values = nuiss_params
        self.yield_param_errors = yield_errors
        self.nui_param_errors = nuiss_param_errors

    # -- properties

    @property
    def num_templates(self):
        """int: Number of templates in this container."""
        return len(self._template_dict)

    @property
    def yield_params(self):
        """List of TemplateParameter: Yield parameters."""
        return [
            template.yield_param for template in self._template_dict.values()
        ]

    @property
    def nui_params(self):
        """List of TemplateParameter: Nuissance parameters."""
        return [
            template.nui_params for template in self._template_dict.values()
        ]

    @property
    def yield_param_values(self):
        """numpy.ndarray: An array with current yield parameter
        values of all templates."""
        return np.array([
            template.yield_param.value
            for template in self._template_dict.values()
        ])

    @yield_param_values.setter
    def yield_param_values(self, new_values):
        for template, value in zip(self._template_dict.values(), new_values):
            template.yield_param_values = value

    @property
    def yield_param_errors(self):
        """numpy.ndarray: An array with current yield parameter
        errors of all templates."""
        return np.array([
            template.yield_param.error
            for template in self._template_dict.values()
        ])

    @yield_param_errors.setter
    def yield_param_errors(self, new_errors):
        for template, error in zip(self._template_dict.values(), new_errors):
            template.yield_param_errors = error

    @property
    def nui_param_values(self):
        """numpy.ndarray: An array with current nuissance parameter
        values of all templates."""
        return np.array([
            template.nui_params.value
            for template in self._template_dict.values()
        ])

    @nui_param_values.setter
    def nui_param_values(self, new_values):
        for template, values in zip(self._template_dict.values(),
                                    np.split(new_values, self.num_templates)):
            template.nui_param_values = values

    @property
    def nui_param_errors(self):
        """numpy.ndarray: An array with current nuissance parameter
        errors of all templates."""
        return np.array([
            template.nui_params_errors
            for template in self._template_dict.values()
        ])

    @nui_param_errors.setter
    def nui_param_errors(self, new_errors):
        for template, errors in zip(self._template_dict.values(),
                                    np.split(new_errors, self.num_templates)):
            template.nui_params_errors = errors

    @property
    def template_names(self):
        """list of str: List of all template names."""
        return [template.name for template in self._template_dict.values()]

    @property
    def inv_corr_mats(self):
        """list of numpy.ndarray: A list of inverse correlation
        matrices for all templates."""
        return [
            template.inv_corr_mat for template in self._template_dict.values()
        ]

    # -- magic methods

    def __getitem__(self, tid):
        """Template: The template stored with id `tid`"""
        return self._template_dict[tid]
