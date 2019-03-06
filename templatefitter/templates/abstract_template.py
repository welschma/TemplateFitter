import logging

from abc import ABC, abstractmethod

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AbstractTemplate",
    "TemplateParameter"
]


class TemplateParameter:
    """
    """

    def __init__(self, value, error, name):
        self.value = value
        self._original_value = value
        self.error = error
        self._original_error = error
        self.name = name

    def reset(self):
        self.value = self._original_value
        self.error = self._original_error


class AbstractTemplate(ABC):
    """
    """

    def __init__(self, name):
        self._name = name
        self._variable = None
        self._limits = None

        self._num_bins = None
        self._bin_edges = None
        self._bin_mids = None
        self._bin_width = None

    @property
    def name(self):
        """str: Template identifier."""
        return self._name

    @property
    def variable(self):
        """str: Variable identifier."""
        return self._variable

    @property
    def limits(self):
        """tuple of float: Limits of the bin edges."""
        return self._limits

    @property
    def num_bins(self):
        return self._num_bins

    @property
    def bin_edges(self):
        """numpy.ndarray: Bin edges of the templates in this model."""
        return self._bin_edges

    @property
    def bin_mids(self):
        """numpy.ndarray: Bin mids of the templates in this model."""
        return self._bin_mids

    @property
    def bin_width(self):
        """float: Bin width of the template histogram"""
        return self._bin_width

    # -- abstract methods

    @abstractmethod
    def generate_asimov_dataset(self):
        pass

    @abstractmethod
    def generate_toy_dataset(self):
        pass

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def errors(self):
        pass

    @abstractmethod
    def fractions(self, nuiss_params):
        pass

    @abstractmethod
    def plot_on(self, ax):
        pass

    @property
    @abstractmethod
    def yield_param_values(self):
        pass

    @yield_param_values.setter
    @abstractmethod
    def yield_param_values(self, new_val):
        pass

    @property
    @abstractmethod
    def nui_param_values(self):
        pass

    @nui_param_values.setter
    @abstractmethod
    def nui_param_values(self, new_val):
        pass
