"""
Template class for a binned likelihood fit.
"""

import numpy as np

from templatefitter import Histogram

class Template:
    """
    TODO

    Parameters
    ----------
    name : str
    variable : str
    nbins : int
    limits : tuple of float
    df : pd.DataFrame, optional
        (the default is None, which implies no initial addition
        of data).
    weight_key : str
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

    @property
    def bin_edges(self):
        return self._hist.bin_edges





