import pandas as pd
import numpy as np


class LogTransformation:
    """Signed logarithmic transformation for numerical data.
    
    This class provides static methods to apply and inverse a signed logarithmic 
    transformation, useful for handling data that spans multiple orders of magnitude 
    while preserving the sign of the original values.
    
    The transformation is defined as: sign(x) * log(|x| + 1)
    The inverse is defined as: sign(xt) * (exp(|xt|) - 1)
    
    This transformation is particularly useful for error metrics and other values
    that may be both positive and negative, and where the magnitude matters more
    than the absolute value at extreme ranges.
    
    Methods
    -------
    transform(x)
        Apply the signed logarithmic transformation to the input.
    
    inverse_transform(xt)
        Reverse the signed logarithmic transformation.
    """
    @staticmethod
    def transform(x):
        xt = np.sign(x) * np.log(np.abs(x) + 1)

        return xt

    @staticmethod
    def inverse_transform(xt):
        x = np.sign(xt) * (np.exp(np.abs(xt)) - 1)

        return x


def train_test_split_horizon(df: pd.DataFrame,
                             horizon: int,
                             id_col: str = 'unique_id',
                             time_col: str = 'ds'):
    """Split time series data into train and test sets based on forecast horizon.
    
    For each unique time series (identified by id_col), splits the data such that
    the last 'horizon' observations form the test set and all preceding observations
    form the training set. This mimics a forecasting scenario where the model is
    trained on historical data and evaluated on future data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data.
        Must include columns for unique ID and timestamp.
    
    horizon : int
        Number of time steps at the end of each series to include in the test set.
        This represents the forecast horizon.
    
    id_col : str, default='unique_id'
        Column name identifying unique time series.
    
    time_col : str, default='ds'
        Column name for timestamps, used for sorting within each series.
    
    Returns
    -------
    train_df : pd.DataFrame
        Training dataset containing all but the last 'horizon' observations
        for each time series.
    
    test_df : pd.DataFrame
        Test dataset containing only the last 'horizon' observations
        for each time series.
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'unique_id': ['A', 'A', 'A', 'B', 'B', 'B'],
    ...     'ds': pd.date_range(start='2023-01-01', periods=3).tolist() * 2,
    ...     'y': [10, 20, 30, 40, 50, 60]
    ... })
    >>> train_df, test_df = train_test_split_horizon(data, horizon=1)
    >>> train_df
       unique_id          ds   y
    0         A  2023-01-01  10
    1         A  2023-01-02  20
    2         B  2023-01-01  40
    3         B  2023-01-02  50
    >>> test_df
       unique_id          ds   y
    0         A  2023-01-03  30
    1         B  2023-01-03  60
    
    Notes
    -----
    Each time series must have at least 'horizon' + 1 observations for this
    function to create meaningful train/test splits.
    """

    df_by_unq = df.groupby(id_col)

    train_l, test_l = [], []
    for _, df_ in df_by_unq:
        df_ = df_.sort_values(time_col)

        train_df_g = df_.head(-horizon)
        test_df_g = df_.tail(horizon)

        train_l.append(train_df_g)
        test_l.append(test_df_g)

    train_df = pd.concat(train_l).reset_index(drop=True)
    test_df = pd.concat(test_l).reset_index(drop=True)

    return train_df, test_df
