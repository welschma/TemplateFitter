import logging

from collections import OrderedDict
from functools import lru_cache, reduce

import numpy as np

from scipy.linalg import block_diag
from scipy.stats import poisson
from numba import jit

from templatefitter.utility import xlogyx

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Channel",
]


class Channel:
    """
    TODO
    """

    def __init__(self, name, bins, range):
        self._name = name

        self._template_dict = OrderedDict()
        self._efficiency_dict = OrderedDict()
        self._processes = tuple()
        self._bins = bins if isinstance(bins, tuple) else (bins,)
        self._range = range
        self._hdata = None
        self._dim = None
        self._template_type = None
        self._htype = None

    @property
    def num_templates(self):
        """int: Number of templates/processes in this channel."""
        return len(self._processes)

    @property
    def bins(self):
        """list of int: Number of bins per dimension."""
        return self._bins

    @property
    def num_bins(self):
        """int: Number of bins per template."""
        return reduce(lambda x, y: x*y, self.bins)

    @property
    def bin_edges(self):
        return list(self.templates.values())[0].bin_edges

    @property
    def range(self):
        return self._range

    @property
    def has_data(self):
        if self._hdata is not None:
            return True
        else:
            return False

    @property
    def num_nui_params(self):
        """int: Number of nuissance parameters in this channel."""
        return self.num_templates*self.num_bins

    @property
    def processes(self):
        return tuple(self._template_dict.keys())

    def __getitem__(self, item):
        return self._template_dict[item]

    @property
    def templates(self):
        """dict: Returns the template dictionary."""
        return self._template_dict

    @property
    def efficiencies(self):
        return self._efficiency_dict

    def reset(self):
        for template in self.templates.values():
            template.reset()

    def add_template(self, process, template, efficiency=1.):
        """Adds a template for a specified process to the template.

        Parameters
        ----------
        process: str
            Process name.
        template: Template1d or Template2d
            A valid template instance (has to inherit from
            AbstractTemplate).
        efficiency: float
            Efficiency of the process to this channel.
        """

        if self._check_template(template):
            self._add_template(process, template, efficiency)
        else:
            raise RuntimeError("Trying to add a non compatible template with the Channel.")

    def add_data(self, hdata):
        """Adds a binned dataset to this channel.

        Parameters
        ----------
        hdata: Hist1d or Hist2d
            A valid histogram instance (has to inherit from
            AbstractHist).
        """

        if self._check_hist(hdata):
            self._hdata = hdata
        else:
            raise RuntimeError("Given data histogram is not compatible with the Channel.")

    @lru_cache()
    def process_indices(self, outer_process_list):
        """Returns a list of indices for the processes in this
        channel matching the `outer_process_list`.
        """
        return [outer_process_list.index(process) for process in self.processes]

    def update_parameters(self, yields, nui_params):
        for template, eff, new_yield, new_nui_params in \
                zip(self.templates.values(), self.efficiencies.values(),
                    np.split(yields, self.num_templates), np.split(nui_params, self.num_templates)):
            template.yield_param = new_yield*eff
            template.nui_params = new_nui_params

    def plot_stacked_on(self, ax, **kwargs):

        bin_mids = [template.bin_mids for template in self.templates.values()]
        bin_edges = next(iter(self.templates.values())).bin_edges
        bin_width = next(iter(self.templates.values())).bin_widths

        colors = [template.color for template in self.templates.values()]
        bin_counts = [template.values for template in self.templates.values()]
        labels = [template.name for template in self.templates.values()]

        if self._dim > 1:
            bin_counts = [self._get_projection(kwargs["projection"], bc) for bc
                          in bin_counts]
            axis = kwargs["projection"]
            ax_to_index = {
                "x": 0,
                "y": 1,
            }
            bin_mids = [mids[ax_to_index[axis]] for mids in bin_mids]
            bin_edges = bin_edges[ax_to_index[axis]]
            bin_width = bin_width[ax_to_index[axis]]

        ax.hist(
            bin_mids,
            weights=bin_counts,
            bins=bin_edges,
            edgecolor="black",
            histtype="stepfilled",
            lw=0.5,
            color=colors,
            label=labels,
            stacked=True
        )

        uncertainties_sq = [template.errors ** 2 for template in
                            self._template_dict.values()]
        if self._dim > 1:
            uncertainties_sq = [
                self._get_projection(kwargs["projection"], unc_sq) for unc_sq in uncertainties_sq
            ]

        total_uncertainty = np.sqrt(np.sum(np.array(uncertainties_sq), axis=0))
        total_bin_count = np.sum(np.array(bin_counts), axis=0)

        ax.bar(
            x=bin_mids[0],
            height=2 * total_uncertainty,
            width=bin_width,
            bottom=total_bin_count - total_uncertainty,
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
            label="MC Uncertainty"
        )

        if self._hdata is None:
            return ax

        data_bin_mids = self._hdata.bin_mids
        data_bin_counts = self._hdata.bin_counts
        data_bin_errors_sq = self._hdata.bin_errors_sq

        if self.has_data:

            if self._dim > 1:
                data_bin_counts = self._get_projection(
                    kwargs["projection"], data_bin_counts
                )
                data_bin_errors_sq = self._get_projection(
                    kwargs["projection"], data_bin_errors_sq
                )

                axis = kwargs["projection"]
                ax_to_index = {
                    "x": 0,
                    "y": 1,
                }
                data_bin_mids = data_bin_mids[ax_to_index[axis]]

            ax.errorbar(x=data_bin_mids, y=data_bin_counts, yerr=np.sqrt(data_bin_errors_sq),
                        ls="", marker=".", color="black", label="Data")

    def nll_contribution(self, process_yields, nui_params):
        """Calculates the contribution to the binned negative log
        likelihood function of this channel.

        Parameters
        ----------
        process_yields: numpy.ndarray
            An array holding the yield values for the processes in
            this channel. The order has to match order of the
            templates that have been added to the template. Shape is
            (`num_processes`).

        Returns
        -------
        float
        """
        data = self._hdata.bin_counts.flatten()
        exp_evts_per_bin = self._expected_evts_per_bin(process_yields, nui_params)
        poisson_term = np.sum(exp_evts_per_bin - data -
                              xlogyx(data, exp_evts_per_bin))
        gauss_term = self._gauss_term(nui_params)

        return poisson_term + gauss_term

    def generate_toy_dataset(self):
        template_bin_counts = sum([template.values for template in self.templates.values()])
        toy_bin_counts = poisson.rvs(template_bin_counts)

        return self._htype.from_binned_data(
            toy_bin_counts, self.bin_edges, np.sqrt(toy_bin_counts)
        )

    def generate_asimov_dataset(self, integer_values=False):
        template_bin_counts = sum([template.values for template in self.templates.values()])
        if integer_values:
            asimov_bin_counts = np.rint(template_bin_counts)
        else:
            asimov_bin_counts = template_bin_counts

        return self._htype.from_binned_data(
            asimov_bin_counts, self.bin_edges, np.sqrt(asimov_bin_counts)
        )

    @staticmethod
    def _get_projection(ax, bc):
        x_to_i = {
            "x": 1,
            "y": 0
        }

        return np.sum(bc, axis=x_to_i[ax])

    def _add_template(self, process, template, efficiency):
        if process in self.processes:
            raise RuntimeError(f"Process {name} already defined.")

        if self._dim is None:
            self._dim = len(template.bins)

        if self._template_type is None:
            self._template_type = type(template)
            self._htype = type(template._hist)

        if type(template) != self._template_type:
            raise RuntimeError(f"Given template type of {type(template)} "
                               f"does not match the template types in this "
                               f"channel ({self._template_type}.")

        self._processes = (*self._processes, process)
        self._template_dict[process] = template
        self._efficiency_dict[process] = efficiency

    def _fractions(self, nui_params):
        """Evaluates all `bin_fractions` methods of all templates in this
        channel. Here, the bin fractions depend on so called nuissance
        parameters which incorporate uncertainties on the template shape.

        Parameters
        ----------
        nui_params:  numpy.ndarray
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
        nui_params_per_template = np.split(nui_params, self.num_templates)
        fractions_per_template = np.array(
            [template.fractions(nui_params) for template, nui_params
             in zip(self._template_dict.values(), nui_params_per_template)]
        )

        return fractions_per_template

    @lru_cache()
    def _get_efficiencies_as_array(self):
        """Returns the efficiencies of the processes in this channel
        as `numpy.ndarray`.
        """
        return np.array([eff for eff in self._efficiency_dict.values()])

    def _expected_evts_per_bin(self, process_yields, nui_params):
        """Calculates the expected number of events given the process
        yields and nuissance parameters.

        Parameters
        ----------
        process_yields : numpy.ndarray
            Shape is (`num_processes`,).
        nui_params : numpy.ndarray
            Shape is (`num_templates*num_bins`,).

        Returns
        -------
        expected_num_evts_per_bin : numpy.ndarray
            Shape is (`num_bins`,).
        """
        return (process_yields * self._get_efficiencies_as_array()) @ self._fractions(nui_params)

    @lru_cache()
    def _create_block_diag_inv_corr_mat(self):
        inv_corr_mats = [template.inv_corr_mat for template
                         in self._template_dict.values()]
        return block_diag(*inv_corr_mats)

    def _gauss_term(self, nui_params):
        inv_corr_block = self._create_block_diag_inv_corr_mat()
        return 0.5 * (nui_params @ inv_corr_block @ nui_params)

    def _check_template(self, template):
        bins_cond = self._bins == template.bins
        range_cond = self._range == template.range
        return bins_cond and range_cond

    def _check_hist(self, hist):
        bins_cond = self._bins == hist.shape
        range_cond = self._range == hist.range
        return bins_cond and range_cond
