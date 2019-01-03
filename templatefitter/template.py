"""
Template class for a binned likelihood fit.
"""

import collections
import numpy as np

from templatefitter import Histogram

#TODO add covariance matrices to templates
class Template:
    """Template provides attributes and methods to compute a
    binned likelihood function based on a Poisson model for each bin.

    Parameters
    ----------
    name : str
        Template name.
    variable : str
        Column name of the fit variable in the given pd.DataFrames.
    nbins : int
        Number of bins for the histogram.
    limits : tuple of float
        Lower and upper limit for the range of the histogram.
    df : pd.DataFrame, optional
        DataFrame which holds the column specified by `variable` 
        (the default is None, which implies no initial addition
        of data).
    weight_key : str
        Key of the column in `df` which holds the event weights 
        (the default is 'weight').

    Attributes
    ----------
    name
    values
    errors
    rel_errors
    expected_yield
    """

    def __init__(
        self,
        name,
        variable,
        nbins,
        limits,
        df=None,
        weight_key="weight"
        ):

       self._name = name
       self._variable = variable
       self._hist = Histogram(nbins, limits)
       self._weight_key = weight_key

       self._cov_mats = None

       if df is not None:
           self.add_df(df)

    def add_df(self, df):
        """Fills template histogram with events from the given 
        pd.DataFrame. The column of interest is given by the
        variable attribute and the event weight is specified
        by the weight_key attribute. If no column with name
        weight_key is found, a weight of 1.0 is assigned to
        each event.

        Arguments
        ---------
        df : pd.DataFrame
            A pd.DataFrame with events for this template.
        
        Returns
        -------
        None
        """
        data = df[self._variable]
        
        try:
            weights = df[self._weight_key]
        except KeyError:
            weights = np.ones_like(data)
        
        self._hist.fill(data, weights)

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        """Expected value of each bin of the template.

        Returns
        -------
        np.ndarray
            Shape (nbins,)
        """
        return self._hist.bin_counts

    @property
    def errors(self):
        """Expected error of each bin of the template.
        This is the square root of the sum of squared event
        weights.

        Returns
        -------
        np.ndarray
            Shape (nbins,)
        """
        return self._hist.bin_errors
    
    @property
    def rel_errors(self):
        """Expected relative error of each bin of the template.
        This is the square root of the sum of squared event
        weights in each bin divided by each bin count.

        Returns
        -------
        np.ndarray
            Shape (nbins,)
        """
        return self._hist.bin_errors/self._hist.bin_counts

    @property
    def expected_yield(self):
        """Expected yield from the Template. This is weighted
        sum of all bin_counts of the added Monte Carlo samples.

        Returns
        -------
        float
        """
        return np.sum(self._hist.bin_counts)

    @expected_yield.setter
    def expected_yield(self, new_yield):
        scale_factor = new_yield/self.expected_yield
        self._hist.scale(scale_factor)

    @property
    def bin_edges(self):
        """Bin edges of the underlying histogram.

        Returns
        -------
        np.ndarray
            Shape (nbins + 1,)
        """
        return self._hist.bin_edges


class TemplateCollection:
    """TemplateCollection is a container that creates and
    stores Template instances and acts as an interface to
    other classes or functions.

    Parameters
    ----------
    variable : str
        Column name of the fit variable in the pd.DataFrames to
        be used for the Template creation.
    nbins : int
        Number of bins for the Template histogram.
    limits : tuple of float
        Lower and upper limit for the range of the histogram.
    weight_key : str
        Key of the column in pd.DataFrames which holds the event
        weights (the default is 'weight').


    Attributes
    ----------
    variable
    bin_edges
    values
    rel_errors
    """

    def __init__(self, variable, nbins, limits, weight_key="weight"):
        self._variable = variable
        self._weight_key = weight_key
        self._nbins = nbins
        self._limits = limits
        self._bin_edges = np.linspace(*limits, nbins+1)
        self._template_map = collections.OrderedDict()
        
    def add_template(self, name, df):
        """Creates a template with labeled `name` from the given
        pd.DataFrame. The templates are stored in an internal map.

        Arguments
        ---------
        name : str
            Name for the template and also the key in the internal
            map storing the templates.
        df : pd.DataFrame
            A pd.DataFrame instance which has to contain at least a
            column name `variable` (specified in the constructor).
            If the DataFrame does not have a column identified by
            `weight_key`, a weight of 1.0 is assigned to each event.

        Returns
        -------
        None
        """
        self._template_map[name] = Template(
            name, 
            self._variable, 
            self._nbins,
            self._limits,
            df,
            self._weight_key)
    
    @property
    def variable(self):
        """Name of the variable which is used to create the 
        Templates

        Returns
        -------
        str
        """
        return self._variable

    @property
    def bin_edges(self):
        """The bin edges of the created Templates.

        Returns
        -------
        np.ndarray
            Shape is (nbins +1,)
        """
        return self._bin_edges

    @property
    def values(self):
        """Matrix of bin counts of the stored templates.

        Returns
        -------
        np.ndarray
            Shape is (number of templates, nbins). The first row
            corresponds to the first added template, the second
            row to the second added template and so on.
        """
        return np.array([template.values for template in self._template_map.values()])

    @property
    def rel_errors(self):
        """Matrix of relative bin errors of the stored templates.

        Returns
        -------
        np.ndarray
            Shape is (number of templates, nbins). The first row
            corresponds to the first added template, the second
            row to the second added template and so on.
        """
        return np.array([template.rel_errors for template in self._template_map.values()])