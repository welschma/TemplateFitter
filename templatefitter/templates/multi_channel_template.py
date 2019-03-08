import logging
from collections import OrderedDict

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Channel",
    "MultiChannelTemplate"
]

class Channel:
    def __init__(self, name):
        self._name = name
        self._processes = list()
        self._channel_process_inidces = None
        self._templates = OrderedDict()
        self._efficiencies = OrderedDict()
        self._hdata = None

    def add_template(self, process, template, efficiency):
        if process in self._processes:
            raise RuntimeError("Process already added.")

        self._processes.append(process)
        self._templates[process] = template
        self._efficiencies[process] = efficiency

    def add_data(self, hdata):
        self._hdata = hdata

    def __getitem__(self, item):
        if isinstance(item, int):
            key = self._processes[item]
            return self._templates[key]
        elif isinstance(item, str):
            if item not in self._processes:
                raise KeyError("No template for this process available.")
            return self._templates[item]
        else:
            raise RuntimeError("Templates only accessible by index or "
                               "process name.")

    def has_process(self, process):
        return True if process in self._processes else False

    def set_process_indices(self, processes):

        self._process_indices = [
            processes.index(process) for process in self._processes
        ]

    def expected_number_of_events(self, process_yields, nui_params):
        pass




class MultiChannelTemplate:
    """Custom container for managing templates for different
    processes and channels.
    """

    def __init__(self):

        self._channels = list()
        self._processes = list()
        self._template_dict = dict()
        self._efficiency_dict = dict()
        self._data_dict = dict()
        self._process_yields = None

    def define_channel(self, channel):
        self._channels.append(channel)
        self._template_dict[channel] = dict()
        self._efficiency_dict[channel] = dict()
        self._data_dict[channel] = dict()

    def define_process(self, process):
        self._processes.append(process)

    def channel_has_process(self, channel, process):
        pass

    def add_template(self, channel, process, template, efficiency):
        if channel not in self._channels or process not in self._processes:
            raise RuntimeError(f"Channel {channel} or process {process} "
                               f"not defined.")

        self._template_dict[channel] = template
        self._efficiency_dict[channel] = efficiency

    def add_data(self, channel_name, data_hist):
        if channel_name not in self._channels :
            raise RuntimeError(f"Channel {channel_name} not defined.")

        self._data_dict[channel_name] = data_hist




