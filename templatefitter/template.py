"""
Template class for a binned likelihood fit.
"""

import collections
import numpy as np

from scipy.stats import poisson

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
    name : str
    values : np.ndarray
        Expected value of each bin of the template. Shape (nbins,).
    errors : np.ndarray
        Expected error of each bin of the template.
        This is the square root of the sum of squared event
        weights. Shape (nbins,).
    rel_errors : np.ndarray
        Expected relative error of each bin of the template.
        This is the square root of the sum of squared event
        weights in each bin divided by each bin count. Shape 
        (nbins,).
    expected_yield : float
        Expected yield from the Template. This is weighted
        sum of all bin_counts of the added Monte Carlo samples.
    bin_edges : np.ndarray
        Bin edges of the underlying histogram. Shape (nbins + 1,).
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
        return self._hist.bin_counts

    @property
    def errors(self):
        return self._hist.bin_errors
    
    @property
    def rel_errors(self):
        return self._hist.bin_errors/self._hist.bin_counts

    @property
    def expected_yield(self):
        return np.sum(self._hist.bin_counts)

    @expected_yield.setter
    def expected_yield(self, new_yield):
        scale_factor = new_yield/self.expected_yield
        self._hist.scale(scale_factor)

    @property
    def bin_edges(self):
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
    variable : str
        Name of the variable which is used to create the 
        Templates.
    bin_edges : np.ndarray
        The bin edges of the created Templates. Shape is (nbins+1,).
    bin_mids : np.ndarray
        The bin mids of the created Templates. Shape is (nbins,).
    values : np.ndarray
        Matrix of bin counts of the stored templates. Shape is 
        (number of templates, nbins). The first row corresponds
        to the first added template, the second row to the second
        added template and so on.
    yields : np.ndarray
        Expected number of events per template. Shape is (n_templates,).
    rel_errors : np.ndarray
        Matrix of relative bin errors of the stored templates. Shape
        is (number of templates, nbins). The first row corresponds
        to the first added template, the second row to the second
        added template and so on.
    """

    def __init__(self, variable, nbins, limits, weight_key="weight"):
        self._variable = variable
        self._weight_key = weight_key
        self._nbins = nbins
        self._limits = limits
        self._bin_edges = np.linspace(*limits, nbins+1)
        self._template_map = collections.OrderedDict()
        
    def add_template(self, name, df):
        """Creates a template labeled `name` from the given
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
        """
        self._template_map[name] = Template(
            name, 
            self._variable, 
            self._nbins,
            self._limits,
            df,
            self._weight_key)

    def set_yields(self, **kwargs):
        """Set expected number of events of stored templates to a
        new value. Arguments are passes as kwargs where the key has
        to match a name of a template in the collection.

        Arguments
        ---------
        kwargs
            E.g. signal=100, background=5000.
        """
        for key, val in kwargs.items():
            self._template_map[key].expected_yield = val
    
    def generate_toy_data(self):
        """Generates toy data using the poisson distribution.
        For each bin, a random number following a poisson distribution
        with mean equal to the expected number of events in this bin
        is generated.

        Returns
        -------
        np.ndarray
            Toy dataset. Shape is (nbins,)
        """
        expected_evts_per_bin = np.sum(self.values, axis=0)
        return poisson.rvs(expected_evts_per_bin)

    @property
    def yields(self):
        return np.sum(self.values, axis=1).reshape(-1)

    @property
    def variable(self):
        return self._variable

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_mids(self):
        return (self.bin_edges[:-1] + self.bin_edges[1:])/2

    @property
    def values(self):
        return np.array([template.values for template in self._template_map.values()])

    @property
    def rel_errors(self):
        return np.array([template.rel_errors for template in self._template_map.values()])