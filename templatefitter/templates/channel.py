import logging

from collections import OrderedDict
from functools import lru_cache

import numpy as np

from scipy.linalg import block_diag

from templatefitter.templates import template_compatible, hist_compatible
from templatefitter.utility import xlogyx

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Channel",
]


class Channel:

    def __init__(self, name):
        self._name = name

        self._template_dict = OrderedDict()
        self._efficiency_dict = OrderedDict()
        self._process_list = list()
        self._bins = None
        self._num_bins = None
        self._range = None
        self._hdata = None

        self._num_templates = 0

    @property
    def num_templates(self):
        """int: Number of templates/processes in this channel."""
        return self._num_templates

    def add_template(self, process, template, efficiency=1.):

        if not self._process_list:
            self._add_template(process, template, efficiency)
            self._bins = template.bins
            self._num_bins = template.num_bins
            self._range = template.range
        elif template_compatible(self._template_dict[self._process_list[0]],
                                 template):
            self._add_template(process, template, efficiency)
        else:
            raise RuntimeError("Trying to add a non compatible template with the Channel.")

    def add_data(self, hdata):

        if not (self._bins and self._range):
            raise RuntimeError("You have to add at least one template before the "
                               "data.")
        elif hist_compatible(self._template_dict[self._process_list[0]], hdata):
            self._hdata = hdata
        else:
            raise RuntimeError("Given data histogram is not compatible with the Channel.")

    def _add_template(self, process, template, efficiency):
        self._process_list.append(process)
        self._template_dict[process] = template
        self._efficiency_dict[process] = efficiency
        self._num_templates += 1

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

    def _expected_evts_per_bin(self, process_yields, nui_params):
        return process_yields @ self._fractions(nui_params)

    @lru_cache()
    def _create_block_diag_inv_corr_mat(self):
        inv_corr_mats = [template.inv_corr_mat for template
                         in self._template_dict.values()]
        return block_diag(*inv_corr_mats)

    def _gauss_term(self, nui_params):
        inv_corr_mat = self._create_block_diag_inv_corr_mat()
        return 0.5 * (nui_params @ inv_corr_mat @ nui_params)

    def nll_contibution(self, process_yields, nui_params):
        data = self._hdata.bin_counts
        exp_evts_per_bin = self._expected_evts_per_bin(process_yields, nui_params)
        poisson_term = np.sum(exp_evts_per_bin - data -
                              xlogyx(data, exp_evts_per_bin))
        gauss_term = self._gauss_term(nui_params)

        return poisson_term + gauss_term
