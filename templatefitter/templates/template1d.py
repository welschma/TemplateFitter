import logging

import numpy as np

from templatefitter.histograms import Hist1d
from templatefitter.templates import AbstractTemplate
from templatefitter.utility import get_systematic_cov_mat

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["Template1d"]


class Template1d(AbstractTemplate):
    """A 1d template class.
    """
    def __init__(
        self,
        name,
        variable,
        hist1d,
        color=None,
        pretty_variable=None,
        pretty_label=None,
    ):
        super().__init__(name=name)

        self._hist = hist1d
        self._flat_bin_counts = self._hist.bin_counts.flatten()
        self._flat_bin_errors_sq = self._hist.bin_errors_sq.flatten()
        self._bins = hist1d.shape
        self._num_bins = hist1d.num_bins
        self._range = hist1d.range

        self._init_params()
        self._init_errors()

        self._variable = variable
        self.pretty_variable = pretty_variable
        self.color = color
        self.pretty_label = pretty_label

    def add_variation(self, data, weights_up, weights_down):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        hup = Hist1d(
            bins=self._hist.num_bins, range=self._range, data=data, weights=weights_up
        )
        hdown = Hist1d(
            bins=self._hist.num_bins, range=self._range, data=data, weights=weights_down
        )
        cov_mat = get_systematic_cov_mat(
            self._flat_bin_counts, hup.bin_counts.flatten(), hdown.bin_counts.flatten()
        )
        self._add_cov_mat(cov_mat)


    def add_cov_mat(self, cov_mat: np.ndarray):
        if (self.num_bins, self.num_bins) != cov_mat.shape:
            raise ValueError(f"Covariance matrix shape does not match number of bins.")
        
        self._add_cov_mat(cov_mat)



    def plot_on(self, ax):
        """Plots the template on given axis.
        """
        ax.hist(
            self._hist.bin_mids,
            weights=self.values,
            bins=self._hist.bin_edges,
            color=self.color,
            edgecolor="black",
            histtype="stepfilled",
            label=self.pretty_label if self.pretty_label is not None else self.name,
        )
        ax.bar(
            x=self._hist.bin_mids,
            height=2 * self.errors,
            width=self._hist.bin_widths,
            bottom=self.values - self.errors,
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
        )

        ax.set_xlabel(self.pretty_variable if self.pretty_variable is not None
                      else self._variable)
