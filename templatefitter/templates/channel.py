import logging

from collections import OrderedDict
from functools import lru_cache, reduce

import numpy as np

from scipy.linalg import block_diag

from templatefitter.templates import template_compatible, hist_compatible
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

    def _check_template(self, template):
        bins_cond = self._bins == template.bins
        range_cond = self._range == template.range
        return bins_cond and range_cond

    def _check_hist(self, hist):
        bins_cond = self._bins == hist.shape
        range_cond = self._range == hist.range
        return bins_cond and range_cond

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

    def _add_template(self, process, template, efficiency):
        if process in self.processes:
            raise RuntimeError(f"Process {name} already defined.")

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
    def _get_efficiency(self):
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

        return (process_yields * self._get_efficiency()) @ self._fractions(nui_params)

    @lru_cache()
    def _create_block_diag_inv_corr_mat(self):
        inv_corr_mats = [template.inv_corr_mat for template
                         in self._template_dict.values()]
        return block_diag(*inv_corr_mats)

    def _gauss_term(self, nui_params):
        inv_corr_mat = self._create_block_diag_inv_corr_mat()
        return 0.5 * (nui_params @ inv_corr_mat @ nui_params)

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
        data = self._hdata.bin_counts
        exp_evts_per_bin = self._expected_evts_per_bin(process_yields, nui_params)
        poisson_term = np.sum(exp_evts_per_bin - data -
                              xlogyx(data, exp_evts_per_bin))
        gauss_term = self._gauss_term(nui_params)

        return poisson_term + gauss_term
