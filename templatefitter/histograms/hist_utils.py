"""Histogram utility module.
"""

__all__ = ["bin_mids", "bin_widths", "get_bin_range"]


def bin_mids(bin_edges):
    """Calculated bin mids from given bin edges.
    """
    return (bin_edges[1:] + bin_edges[:-1]) / 2


def bin_widths(bin_edges):
    """Calculates bin widths from given bin edges.
    """
    return bin_edges[1:] - bin_edges[:-1]


def get_bin_range(bin_edges):
    """Return first and last entry in the bin edges.
    """
    return bin_edges[0], bin_edges[-1]
