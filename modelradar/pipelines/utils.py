import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


class LogTransformation:

    @staticmethod
    def transform(x):
        xt = np.sign(x) * np.log(np.abs(x) + 1)

        return xt

    @staticmethod
    def inverse_transform(xt):
        x = np.sign(xt) * (np.exp(np.abs(xt)) - 1)

        return x


class DecompositionSTL:

    @classmethod
    def get_strengths(cls, series_df: pd.DataFrame, period):
        comps = cls.get_stl_components(series=series_df['y'], period=period)

        seasonal_str = cls.seasonal_strength(comps['Seasonal'], comps['Residuals'])
        trend_str = cls.trend_strength(comps['Trend'], comps['Residuals'])

        strs = {'seasonal_str': seasonal_str, 'trend_str': trend_str}

        return strs

    @staticmethod
    def get_stl_components(series: pd.Series,
                           period: int,
                           add_residuals: bool = True) -> pd.DataFrame:
        """
        Decomposes a time series into trend, seasonal, and optionally residual components using STL.

        Args:
            series (pd.Series): Time series data.
            period (int): Period for seasonal decomposition.
            add_residuals (bool, optional): Flag to include residuals in the output. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the decomposed components.
        """

        ts_decomp = STL(series, period=period).fit()

        components = {
            'Trend': ts_decomp.trend,
            'Seasonal': ts_decomp.seasonal,
        }

        if add_residuals:
            components['Residuals'] = ts_decomp.resid

        components_df = pd.DataFrame(components).reset_index()

        return components_df

    @staticmethod
    def seasonal_strength(seasonal: pd.Series, residuals: pd.Series):
        assert seasonal.index.equals(residuals.index)

        # variance of residuals + seasonality
        resid_seas_var = (residuals + seasonal).var()
        # variance of residuals
        resid_var = residuals.var()

        # seasonal strength
        result = 1 - (resid_var / resid_seas_var)
        result = max(0, result)
        result = np.round(result, 2)

        return result

    @staticmethod
    def trend_strength(trend: pd.Series, residuals: pd.Series):
        assert trend.index.equals(residuals.index)

        # variance of residuals + trend
        resid_trend_var = (residuals + trend).var()
        # variance of residuals
        resid_var = residuals.var()

        # seasonal strength
        result = 1 - (resid_var / resid_trend_var)
        result = max(0, result)
        result = np.round(result, 2)

        return result
