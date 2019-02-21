TemplateFitter Reference
========================

Reference page for the TemplateFitter package.

Histogram Module
################
AbstractHist
""""""""""""
.. autoclass:: templatefitter.AbstractHist
    :members:

Hist1d
""""""
.. autoclass:: templatefitter.Hist1d
    :members:
    :inherited-members:

Template Module
###############
AbstractTemplate
""""""""""""""""
.. autoclass:: templatefitter.AbstractTemplate
    :members:

SimpleTemplate
""""""""""""""
.. autoclass:: templatefitter.Template
    :members:
    :inherited-members:

AdvancedTemplate
""""""""""""""""
.. autoclass:: templatefitter.StackedTemplate
    :members:
    :inherited-members:

NegativeLogLikelihood Module
############################
AbstractTemplateCostFunction
""""""""""""""""""""""""""""
.. autoclass:: templatefitter.AbstractTemplateCostFunction
    :members:

StackedTemplateNegLogLikelihood
"""""""""""""""""""""""""""""""
.. autoclass:: templatefitter.StackedTemplateNegLogLikelihood
    :members:
    :inherited-members:

Fitter Module
#############

TemplateFitter
""""""""""""""
.. autoclass:: templatefitter.TemplateFitter
    :members:

ToyStudy
""""""""
.. autoclass:: templatefitter.ToyStudy
    :members:

Minimizer Module
################

MinimizeResult
""""""""""""""
.. autoattribute:: templatefitter.MinimizeResult
    :members:

Parameters
""""""""""
.. autoclass:: templatefitter.Parameters
    :members:

AbstractMinimizer
"""""""""""""""""
Defines the interface of the minizers methods
.. autoclass:: templatefitter.AbstractMinimizer
    :members:

IminuitMinimizer
""""""""""""""""
.. autoclass:: templatefitter.IMinuitMinimizer
    :members:
    :inherited-members:

ScipyMinimizer
""""""""""""""""
.. autoclass:: templatefitter.ScipyMinimizer
    :members:
    :inherited-members:

Stats Module
############
.. automodule:: templatefitter.stats
    :members:
