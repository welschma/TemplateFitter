TemplateFitter Reference
========================

Template Creation
-----------------
Helpful classes to create templates used to fit monte carlo 
distributions to measured data.

Histogram
#########
.. autoclass:: templatefitter.Histogram
    :members:

Template
########
.. autoclass:: templatefitter.TemplateModel
    :members:

TemplateCollection
##################
.. autoclass:: templatefitter.CompositeTemplateModel
    :members:

Likelihood Functions
--------------------
Default Binned Negative Log Likelihood
######################################
.. autoclass:: templatefitter.PoissonNLL
    :members:
    :special-members: __call__


Fitting Functions
-----------------
ToyStudy
########
.. autoclass:: templatefitter.ToyStudy
    :members:


Minimizer
---------
Parameters
##########
.. autoclass:: templatefitter.Parameters
    :members:

Minimizer
#########
.. autoclass:: templatefitter.Minimizer
    :members:



