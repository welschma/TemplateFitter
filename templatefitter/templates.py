"""This module provides several classes which help to implement templates
used for binned likelihood fits where the expected number of events is
estimated from different histograms.
"""
import logging

from abc import ABC, abstractmethod

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AbstractTemplate",
    "Template",
    "StackedTemplate",
    "SimultaneousTemplate"
]


class AbstractTemplate(ABC):
    pass


class Template(AbstractTemplate):
    pass


class StackedTemplate(AbstractTemplate):
    pass


class SimultaneousTemplate(AbstractTemplate):
    pass
