import logging
from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

from templatefitter.histograms import Hist2d
from templatefitter.templates import AbstractTemplate

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ["Template2d"]


class Template2d(AbstractTemplate):
    """A 2 dimensional template.
    """
    def __init__(
            self,
            name,
            vars,
            bins,
            range,
            data=None,
            weights=None,
            color=None,
            x_pretty_var=None,
            y_pretty_var=None,
            pretty_label=None,
    ):
        super(Template2d, self).__init__(name=name)

        self._hist = Hist2d(bins=bins, range=range, data=data, weights=weights)
        self._flat_bin_counts = self._hist.bin_counts.flatten()
        self._flat_bin_errors_sq = self._hist.bin_errors_sq.flatten()
        self._bins = bins
        self._num_bins = reduce(lambda x, y: x*y, bins)
        self._range = range

        self._init_params()
        self._init_errors()
        self._x_var = vars[0]
        self._y_var = vars[1]
        self.x_pretty_var = x_pretty_var
        self.y_pretty_var = y_pretty_var
        self.color = color
        self.pretty_label = pretty_label

    def add_variation(self, data, weights_up, weights_down):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        hup = Hist2d(
            bins=self._hist.num_bins, range=self._range, data=data, weights=weights_up
        )
        hdown = Hist2d(
            bins=self._hist.num_bins, range=self._range, data=data, weights=weights_down
        )

        self._add_cov_mat(hup, hdown)

    def plot_on(self, fig, ax):
        """Plots the 2d template on the given axis.
        """
        edges = self._hist.bin_mids
        xe = edges[0]
        ye = edges[1]

        xy = np.array(list(product(xe, ye)))

        viridis = cm.get_cmap("viridis", 256)
        cmap = viridis(np.linspace(0, 1, 256))
        cmap[0, :] = np.array([1, 1, 1, 1])
        newcm = ListedColormap(cmap)
        cax =ax.hist2d(
            x=xy[:, 0],
            y=xy[:, 1],
            weights=self.values.flatten(),
            bins=self._hist.bin_edges,
            cmap=newcm,
            label=self.name
        )
        ax.set_title(self.pretty_label if self.pretty_label is not None else
                     self.name)
        ax.set_xlabel(self.x_pretty_var if self.x_pretty_var is not None else
                      self._x_var)
        ax.set_ylabel(self.y_pretty_var if self.y_pretty_var is not None else
                      self._y_var)
        fig.colorbar(cax[3])

    def plot_x_projection_on(self, ax):
        """Plots the x projection of the template on the
        given axis.
        """
        values = np.sum(self.values, axis=1)
        errors = np.sqrt(np.sum(self.errors**2, axis=1))
        projection = self._hist.x_projection()
        self._plot_projection(ax, values, errors, projection)
        ax.set_xlabel(self.x_pretty_var if self.x_pretty_var is not None
                      else self._x_var)

    def plot_y_projection_on(self, ax):
        """Plots the y projection of the template on the
        given axis.
        """
        values = np.sum(self.values, axis=0)
        errors = np.sqrt(np.sum(self.errors**2, axis=0))
        projection = self._hist.y_projection()
        self._plot_projection(ax, values, errors, projection)
        ax.set_xlabel(self.y_pretty_var if self.y_pretty_var is not None
                      else self._y_var)

    def _plot_projection(self, ax, values, errors, projection):
        """Helper function to prevent code duplication.
        """
        ax.hist(
            projection.bin_mids,
            weights=values,
            bins=projection.bin_edges,
            color=self.color,
            edgecolor="black",
            histtype="stepfilled",
            label=self.pretty_label if self.pretty_label is not None else self.name,
        )
        ax.bar(
            x=projection.bin_mids,
            height=2 * errors,
            width=projection.bin_widths,
            bottom=values - errors,
            color="black",
            hatch="///////",
            fill=False,
            lw=0,
        )
