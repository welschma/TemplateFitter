import logging
from collections import OrderedDict

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "MultiChannelTemplate"
]


class MultiChannelTemplate:

    def __init__(self):
        pass

    def define_channel(self):
        pass

    def define_channels(self):
        pass

    def define_process(self):
        pass

    def define_processes(self):
        pass

    def add_template(self):
        pass

    def add_templates(self):
        pass

    def add_data(self):
        pass


class NegLogLikelihood:

    def __init__(self, multi_channel_template):
        self._mcl = multi_channel_template

    def __call__(self, x):
        pass

