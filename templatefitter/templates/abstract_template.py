import logging

from abc import ABC, abstractmethod

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = [
    "AbstractTemplate",
]


class AbstractTemplate(ABC):
    """Defines the template interface.

    """

    def __init__(self, name):
        self._name = name
        self._params = None
        self._default_yield = None

    # -- properties --

    @property
    def name(self):
        """str: Template identifier."""
        return self._name

    @property
    def params(self):
        """numpy.ndarray: Array of template parameters.
        The first entry is the yield, the rest are the
        nuissance parameters. Shape is (`num_bins + 1`,).
        """
        return self._params

    @params.setter
    def params(self, new_values):

        if new_values.shape != self._params.shape:
            raise RuntimeError("Shape of new parameter array is not compatible"
                               " to this template.")

        self._params = new_values

    @property
    def yield_param(self):
        """float: The current yield value.
        """
        return self._params[0]

    @property
    def nui_params(self):
        """numpy.ndarray: The current nuissance parameters.
        """
        return self._params[1:]

    def reset(self):
        """Resets parameter to the original values.
        """
        self._init_params()

    # -- abstract methods --

    @property
    @abstractmethod
    def values(self):
        pass

    @property
    @abstractmethod
    def errors(self):
        pass

    @abstractmethod
    def fractions(self, nui_params):
        pass

    @abstractmethod
    def _init_params(self):
        pass








