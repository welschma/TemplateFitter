import logging

from collections import OrderedDict

import numpy as np
import scipy.stats

from templatefitter.histogram import Hist1d
from templatefitter.templates import AbstractTemplate, Template
from templatefitter.nll import StackedTemplateNegLogLikelihood

logging.getLogger(__name__).addHandler(logging.NullHandler())


class MulitChannelTemplate(AbstractTemplate):
    pass
