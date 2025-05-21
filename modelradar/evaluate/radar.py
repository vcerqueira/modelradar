from typing import List, Optional, Callable

import numpy as np
import pandas as pd

from utilsforecast.evaluation import evaluate as uf_evaluate

from modelradar.evaluate.rope import RopeAnalysis
from modelradar.visuals.plotter import ModelRadarPlotter


class BaseModelRadar:
    COLUMNS = {
        'metric': 'metric',
        'horizon': 'horizon',
        'cutoff': 'cutoff',
    }

    DF_RESULT_COLUMNS = ['Model', 'Result']

    def __init__(self,
                 cv_df: pd.DataFrame,
                 metrics: List[Callable],
                 model_names: Optional[List[str]],
                 agg_func: str = 'mean',
                 id_col: str = 'unique_id',
                 time_col: str = 'ds',
                 target_col: str = 'y'):
        """Base class for model evaluation and radar analysis.
        
        This class serves as the foundation for analyzing and comparing multiple forecasting
        models using cross-validation data based on several aspects. 
        It handles data preparation, identifies models,
        and provides utility methods for evaluation across different dimensions.
        
        Parameters
        ----------
        cv_df : pd.DataFrame
            Cross-validation pd.DataFrame containing forecasts from multiple models.
            Expected to have columns for unique identifiers, timestamps, target values,
            and forecast values for each model, following a Nixtla-based structure.
        
        metrics : List[Callable]
            List of metric functions to evaluate forecasts against actual values.
            Multiple metrics will be averaged to produce a single score.
            Each metric should accept arrays of actual and predicted values.
        
        model_names : Optional[List[str]]
            List of model names to evaluate. If None, model names will be
            automatically detected from the columns in cv_df.

        agg_func : str, default='mean'
            Aggregation function to use for error metrics. Either 'mean' or 'median'.
        
        id_col : str, default='unique_id'
            Column name in cv_df that identifies unique series.
        
        time_col : str, default='ds'
            Column name in cv_df that contains timestamps.
        
        target_col : str, default='y'
            Column name in cv_df that contains the target (actual) values.

        """

        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.agg_func = agg_func
        self.metrics = metrics

        self.cv_df = self._reset_on_uid(cv_df)
        self.cv_df = self._set_horizon_on_df(self.cv_df)

        self.train_df = None
        self.meta_data_cols = []
        self.model_order = []

        self.models = self._get_model_names(cv_df) if model_names is None else model_names

        self.added_metadata = [col for col in cv_df.columns
                               if col not in self.models + self.meta_data_cols]

    def _set_horizon_on_df(self, cv: pd.DataFrame) -> pd.DataFrame:
        """Set horizon values on the cross-validation DataFrame.
    
        Adds a horizon column to the DataFrame, representing the sequential 
        position of each timestamp within each group (unique_id and optionally cutoff).
        
        Parameters
        ----------
        cv : pd.DataFrame
            Cross-validation DataFrame to process.
        
        Returns
        -------
        pd.DataFrame
            Copy of the input DataFrame with added horizon column.
            
        Notes
        -----
        This method ensures datetime columns are properly formatted and 
        sorts the DataFrame by group and timestamp before calculating horizons.
        """

        cv_ = cv.copy()
        co = self.COLUMNS.get('cutoff')

        groups = [self.id_col, co] if co in cv_.columns else [self.id_col]
        dt_cols = [self.time_col, co] if co in cv_.columns else [self.time_col]

        for col in dt_cols:
            if not pd.api.types.is_datetime64_any_dtype(cv_[col]):
                cv_[col] = pd.to_datetime(cv_[col])

        cv_ = cv_.sort_values(groups + [self.time_col])

        cv_[self.COLUMNS.get('horizon')] = cv_.groupby(groups).cumcount() + 1

        return cv_

    def _get_model_names(self, cv: pd.DataFrame):
        """Extract model names from the DataFrame columns.
    
        Identifies model columns by excluding metadata columns from all columns.
        
        Parameters
        ----------
        cv : pd.DataFrame
            Cross-validation DataFrame containing model columns.
        
        Returns
        -------
        list
            List of column names identified as models.
        """
        self._set_meta_data()

        meta_cols_j = cv.columns.str.contains('|'.join(self.meta_data_cols))
        models = cv.loc[:, ~meta_cols_j].columns.tolist()

        return models

    def _set_meta_data(self):
        """Define metadata columns to exclude when identifying model columns.
    
        Sets the meta_data_cols attribute with standard column names that
        are not considered model forecast columns.
        """
        self.meta_data_cols = [self.id_col,
                               self.time_col,
                               self.COLUMNS.get('cutoff'),
                               self.COLUMNS.get('horizon'),
                               self.target_col,
                               'lo', 'hi']

    def _reset_on_uid(self, df: pd.DataFrame):
        """Reset DataFrame index if it matches the ID column.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to process.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with index reset if needed.
        """
        if df.index.name == self.id_col:
            df = df.reset_index()

        return df

    @classmethod
    def _to_df_and_rename(cls, s: pd.Series):
        """Convert a Series to a DataFrame with standardized column names.
    
        Parameters
        ----------
        s : pd.Series
            Series to convert to DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with DF_RESULT_COLUMNS as column names.
        """
        df = s.reset_index()
        df.columns = cls.DF_RESULT_COLUMNS

        return df


class ModelRadarAcrossId:
    """Class for analyzing model performance across unique identifiers.
    
    This class provides methods to identify "hard" cases (series with high errors)
    for a reference model and analyze model performance on these difficult cases.
    It also calculates expected shortfall (tail risk) as a measure of worst-case
    performance for each model.
    
    Parameters
    ----------
    reference : Optional[str]
        Name of the reference model used to identify hard cases.
        Must be a column name in the error DataFrame.
    
    hardness_quantile : float, default=0.95
        Quantile threshold to identify hard cases. Series with errors
        above this quantile for the reference model are considered "hard".
    
    cvar_quantile : float, default=0.9
        Quantile threshold for calculating expected shortfall (Conditional Value at Risk).
        Represents the average error in the worst cases beyond this quantile.

    agg_func : str, default='mean'
            Aggregation function to use for error metrics. Defaults to 'mean'.
        
    Attributes
    ----------
    hardness_threshold : float or None
        Computed threshold value that defines hard cases.
    
    hard_uid : List[str]
        List of unique identifiers classified as hard cases.
    """

    def __init__(self,
                 reference: Optional[str],
                 hardness_quantile: float = 0.95,
                 cvar_quantile: float = 0.9,
                 agg_func: str = 'mean'):
        self.reference = reference
        self.hardness_quantile = hardness_quantile
        self.cvar_quantile = cvar_quantile
        self.hardness_threshold = None
        self.hard_uid: List[str] = []
        self.agg_func = agg_func

    def get_hard_uids(self, err_df: pd.DataFrame, return_df: bool = True):
        """Identify hard time series based on reference model performance.
        
        Parameters
        ----------
        err_df : pd.DataFrame
            DataFrame containing error metrics for each unique ID and model.
            Index should be unique IDs and columns should be model names.
        
        return_df : bool, default=True
            If True, returns a DataFrame with rows filtered to hard unique IDs.
            If False, returns None but still updates the hard_uid attribute.
        
        Returns
        -------
        pd.DataFrame or None
            DataFrame filtered to contain only hard unique IDs if return_df is True,
            otherwise None.
            
        Raises
        ------
        AssertionError
            If the reference model is not found in err_df columns.
        
        Notes
        -----
        This method updates the hardness_threshold and hard_uid attributes.
        """

        assert self.reference in err_df.columns

        self.hardness_threshold = err_df[self.reference].quantile(self.hardness_quantile)
        err_df_h = err_df.loc[err_df[self.reference] > self.hardness_threshold, :]
        self.hard_uid = err_df_h.index.tolist()

        if return_df:
            err_df_h = err_df.loc[self.hard_uid, :]
            return err_df_h

        return None

    def accuracy_on_hard(self, err_df: pd.DataFrame):
        """Calculate average model performance on hard unique IDs.
        
        Parameters
        ----------
        err_df : pd.DataFrame
            DataFrame containing error metrics for each unique ID and model.
            Index should be unique IDs and columns should be model names.
        
        Returns
        -------
        pd.Series
            Series containing the average error for each model on hard unique IDs.
            
        Notes
        -----
        This method calls get_hard_uids internally to identify hard cases
        before calculating the averages.
        """

        err_df_uid = self.get_hard_uids(err_df=err_df)
        err_uid_avg = err_df_uid.agg(self.agg_func, numeric_only=True)
        err_uid_avg.name = 'On Hard'

        return err_uid_avg

    def expected_shortfall(self,
                           err_df: pd.DataFrame,
                           return_plot: bool = False,
                           **kwargs):
        """Calculate expected shortfall (CVaR) for each model.
        
        Expected shortfall measures the average error in the worst cases,
        specifically the average error beyond the cvar_quantile threshold.
        This is also known as Conditional Value at Risk (CVaR) in finance.
        
        Parameters
        ----------
        err_df : pd.DataFrame
            DataFrame containing error metrics for each unique ID and model.
            Index should be unique IDs and columns should be model names.
        
        return_plot : bool, default=False
            If True, returns a plotnine plot visualizing the expected shortfall.
            If False, returns a pandas Series with the values.
        
        **kwargs
            Additional keyword arguments passed to ModelRadarPlotter.error_barplot
            when return_plot is True.
        
        Returns
        -------
        pd.Series or plotnine.ggplot
            Series containing expected shortfall for each model if return_plot is False,
            otherwise a plotnine plot visualizing these values.
        """
        
        shortfall = err_df.apply(lambda x: x[x > x.quantile(self.cvar_quantile)].agg(self.agg_func))
        shortfall.name = 'Exp. Shortfall'

        if return_plot:
            shortfall_df = shortfall.reset_index()
            shortfall_df.columns = ['Model', 'Exp. Shortfall']

            plot = ModelRadarPlotter.error_barplot(data=shortfall_df,
                                                   x='Model',
                                                   y='Exp. Shortfall',
                                                   **kwargs)

            return plot

        return shortfall


class ModelRadar(BaseModelRadar):
    """ ModelRadar aspect-based forecast evaluation
    
    Extends BaseModelRadar to provide comprehensive model evaluation
    capabilities across multiple aspects, including overall performance,
    horizon-based analysis, anomaly handling, and group-based comparisons.
    It also integrates rope analysis for model comparison and identification
    of hard cases.
    
    Parameters
    ----------
    cv_df : pd.DataFrame
        Cross-validation DataFrame containing forecasts from multiple models.
    
    metrics : List[Callable]
        List of metric functions to evaluate forecasts against actual values.
    
    model_names : Optional[List[str]]
        List of model names to evaluate. If None, auto-detected from cv_df.
    
    hardness_reference : Optional[str]
        Reference model for identifying hard cases. Used in ModelRadarAcrossId.
    
    ratios_reference : Optional[str]
        Reference model for rope analysis comparison.
    
    rope : float, default=1.0
        Region of practical equivalence for rope analysis, as a percentage.
    
    cvar_quantile : float, default=0.95
        Quantile threshold for conditional value at risk calculation.
    
    hardness_quantile : float, default=0.9
        Quantile threshold for identifying hard cases.

    agg_func : str, default='mean'
        Aggregation function to use for error metrics. Either 'mean' or 'median'.

    train_df : Optional[pd.DataFrame], default=None
            Optional training DataFrame to pass to the evaluation function.
    
    id_col : str, default='unique_id'
        Column name for unique identifiers.
    
    time_col : str, default='ds'
        Column name for timestamps.
    
    target_col : str, default='y'
        Column name for target values.
    
    Attributes
    ----------
    uid_accuracy : ModelRadarAcrossId
        Instance for analyzing hard cases.
    
    rope : RopeAnalysis
        Instance for rope-based model comparison.
    
    model_order : list
        Ordered list of models based on overall evaluation results.
    """

    def __init__(self,
                 cv_df: pd.DataFrame,
                 metrics: List[Callable],
                 model_names: Optional[List[str]],
                 hardness_reference: Optional[str],
                 ratios_reference: Optional[str],
                 rope: float = 1.0,
                 cvar_quantile: float = 0.95,
                 hardness_quantile: float = 0.9,
                 train_df: Optional[pd.DataFrame] = None,
                 agg_func: str = 'mean',
                 id_col: str = 'unique_id',
                 time_col: str = 'ds',
                 target_col: str = 'y'):

        super().__init__(cv_df=cv_df,
                         metrics=metrics,
                         model_names=model_names,
                         agg_func=agg_func,
                         id_col=id_col,
                         time_col=time_col,
                         target_col=target_col)

        self.uid_accuracy = ModelRadarAcrossId(reference=hardness_reference,
                                               cvar_quantile=cvar_quantile,
                                               hardness_quantile=hardness_quantile,
                                               agg_func=agg_func)

        self.rope = RopeAnalysis(reference=ratios_reference, rope=rope)
        self.train_df = train_df
        self.model_order = self.evaluate().sort_values().index.tolist()

    def evaluate(self,
                 cv: Optional[pd.DataFrame] = None,
                 keep_uids: bool = False,
                 return_plot: bool = False,
                 **kwargs):
        
        """Evaluate models using specified metrics.
        
        Calculates error metrics for each model in the cross-validation DataFrame,
        either aggregated across all series or keeping results for each unique ID.
        
        Parameters
        ----------
        cv : Optional[pd.DataFrame], default=None
            Cross-validation DataFrame to evaluate. If None, uses self.cv_df.
        
        keep_uids : bool, default=False
            If True, returns metrics grouped by unique ID.
            If False, returns average metrics across all unique IDs.
        
        return_plot : bool, default=False
            If True and keep_uids is False, returns a bar plot of errors.
            If False, returns a DataFrame or Series of error values.
        
        **kwargs
            Additional keyword arguments passed to ModelRadarPlotter.error_barplot
            when return_plot is True.
        
        Returns
        -------
        pd.DataFrame, pd.Series, or plotnine.ggplot
            Evaluation results as DataFrame (if keep_uids is True),
            Series (if keep_uids is False and return_plot is False),
            or plot (if keep_uids is False and return_plot is True).
        """
 
        cv_ = self.cv_df if cv is None else cv

        scores_df = uf_evaluate(df=cv_,
                                models=self.models,
                                metrics=self.metrics,
                                train_df=self.train_df)

        if keep_uids:
            scores_df = scores_df.groupby(self.id_col).agg(self.agg_func, numeric_only=True)
        else:
            scores_df = scores_df.drop(columns=[self.id_col, self.COLUMNS.get('metric')]).agg(self.agg_func, numeric_only=True)
            scores_df.name = 'Overall'

            if return_plot:
                scores_df = scores_df.reset_index()
                scores_df.columns = ['Model', 'Error']

                plot = ModelRadarPlotter.error_barplot(data=scores_df,
                                                       x='Model',
                                                       y='Error', **kwargs)

                return plot

        return scores_df

    def evaluate_by_horizon(self,
                            cv: Optional[pd.DataFrame] = None,
                            group_by_freq: bool = False,
                            freq_col: str = 'Frequency',
                            return_plot: bool = False):
        """Evaluate models across different forecast horizons.
        
        Calculates error metrics for each model at each forecast horizon,
        optionally grouped by a frequency column.
        
        Parameters
        ----------
        cv : Optional[pd.DataFrame], default=None
            Cross-validation DataFrame to evaluate. If None, uses self.cv_df.
        
        group_by_freq : bool, default=False
            If True, evaluations are grouped by the freq_col.
            If False, all series are evaluated together.
        
        freq_col : str, default='Frequency'
            Column name to group by when group_by_freq is True.
        
        return_plot : bool, default=False
            If True, returns a line plot of errors across horizons.
            If False, returns a DataFrame of error values.
        
        Returns
        -------
        pd.DataFrame or plotnine.ggplot
            DataFrame of error metrics by horizon (and optionally frequency),
            or a line plot visualizing these errors if return_plot is True.
        """        

        cv_ = self.cv_df if cv is None else cv

        if group_by_freq:
            cv_groups = cv_.groupby(freq_col)
        else:
            cv_groups = [(None, cv_)]

        scores_by_group = {}
        for g, df in cv_groups:
            fh = df[self.COLUMNS.get('horizon')].sort_values().unique()
            eval_horizon = {}
            for h in fh:
                cv_fh = df.query(f'{self.COLUMNS.get("horizon")}<={h}')
                eval_horizon[h] = self.evaluate(cv_fh)

            results = pd.DataFrame(eval_horizon).T
            scores_by_group[g] = results

        scores_df = pd.concat(scores_by_group)

        if group_by_freq:
            scores_df = scores_df.reset_index()
        else:
            scores_df = scores_df.reset_index(drop=True)
            scores_df[self.COLUMNS.get("horizon")] = np.arange(1, scores_df.shape[0] + 1)

        if return_plot:
            plot = ModelRadarPlotter.error_by_horizon(data=scores_df, break_interval=2)

            return plot

        return scores_df

    def evaluate_by_horizon_bounds(self,
                                   cv: Optional[pd.DataFrame] = None,
                                   return_plot: bool = False,
                                   plot_model_cats: Optional[List[str]] = None,
                                   **kwargs) -> pd.DataFrame:
        """Compare model performance at first and last forecast horizons.
        
        Evaluates models on only the first horizon (immediate forecast) and
        last horizon (furthest forecast) for each unique ID, allowing comparison
        of model performance between short and long-term forecasts.
        
        Parameters
        ----------
        cv : Optional[pd.DataFrame], default=None
            Cross-validation DataFrame to evaluate. If None, uses self.cv_df.
        
        return_plot : bool, default=False
            If True, returns a faceted bar plot comparing first and last horizons.
            If False, returns a DataFrame with results.
        
        plot_model_cats : Optional[List[str]], default=None
            List of model categories for plot ordering. Required if return_plot is True.
        
        **kwargs
            Additional keyword arguments passed to ModelRadarPlotter.error_by_horizon_fl
            when return_plot is True.
        
        Returns
        -------
        pd.DataFrame or plotnine.ggplot
            DataFrame with 'First horizon' and 'Last horizon' columns for each model,
            or a plot comparing these values if return_plot is True.
        """

        cv_ = self.cv_df if cv is None else cv

        if self.COLUMNS.get('cutoff') in cv_:
            sorted_cv = cv_.sort_values([self.id_col,
                                         self.COLUMNS.get('cutoff'),
                                         self.COLUMNS.get('horizon')])
        else:
            sorted_cv = cv_.sort_values([self.id_col,
                                         self.COLUMNS.get('horizon')])

        first_horizon = sorted_cv.groupby(self.id_col).first().reset_index()
        last_horizon = sorted_cv.groupby(self.id_col).last().reset_index()

        errors_first = self.evaluate(first_horizon)
        errors_last = self.evaluate(last_horizon)

        errors_first_df = self._to_df_and_rename(errors_first)
        errors_first_df = errors_first_df.rename(columns={'Result': 'First horizon'})

        errors_last_df = self._to_df_and_rename(errors_last)
        errors_last_df = errors_last_df.rename(columns={'Result': 'Last horizon'})

        errors_combined = errors_first_df.merge(errors_last_df, on=self.DF_RESULT_COLUMNS[0])

        if return_plot:
            assert plot_model_cats is not None

            errors_combined_lg = errors_combined.melt('Model')
            column_mapper = {'variable': 'Horizon', 'value': 'Error'}
            errors_combined_lg = errors_combined_lg.rename(columns=column_mapper)

            plot = ModelRadarPlotter.error_by_horizon_fl(errors_combined_lg,
                                                         model_cats=plot_model_cats,
                                                         **kwargs)

            return plot

        errors_combined = errors_combined.set_index('Model')

        return errors_combined

    def evaluate_by_anomaly(self,
                            cv: Optional[pd.DataFrame] = None,
                            mode: str = 'observations',
                            anomaly_col: str = 'is_anomaly'):
        """Evaluate models on anomalous data points or series.
        
        Calculates error metrics for each model on series containing anomalies,
        either focusing only on the anomalous observations or on the entire series
        that contain anomalies.
        
        Parameters
        ----------
        cv : Optional[pd.DataFrame], default=None
            Cross-validation DataFrame to evaluate. If None, uses self.cv_df.
        
        mode : str, default='observations'
            Evaluation mode, either 'observations' (only anomalous points) or
            'series' (entire series containing anomalies).
        
        anomaly_col : str, default='is_anomaly'
            Column name indicating anomaly status (expected to contain 0/1 values).
        
        Returns
        -------
        pd.DataFrame or None
            DataFrame of error metrics for each series with anomalies,
            or None if no anomalies are found.
            
        Raises
        ------
        ValueError
            If the mode parameter is not one of 'observations' or 'series'.
        """

        cv_ = self.cv_df if cv is None else cv

        if mode not in ['observations', 'series']:
            raise ValueError("mode must be either 'observations' or 'series'")

        scores_uids = {}
        for uid, df_uid in cv_.groupby(self.id_col):
            # todo 0,1 numeric assumption
            has_anomalies = df_uid[anomaly_col].sum() > 0

            if has_anomalies:
                uid_ = df_uid.loc[df_uid[anomaly_col] > 0, :] if mode == 'observations' else df_uid

                if not uid_.empty:
                    scores_uid_ = self.evaluate(uid_)
                    scores_uids[uid] = scores_uid_

        if len(scores_uids) < 1:
            return None

        scores_df = pd.DataFrame(scores_uids).T

        return scores_df

    def evaluate_by_group(self,
                          group_col: str,
                          cv: Optional[pd.DataFrame] = None,
                          return_plot: bool = False,
                          plot_model_cats: Optional[List[str]] = None,
                          **kwargs) -> pd.DataFrame:
        """Evaluate models separately for each group in a categorical column.
        
        Calculates error metrics for each model within each category of the specified
        grouping column, allowing comparison of model performance across different
        data segments.
        
        Parameters
        ----------
        group_col : str
            Column name to group by for evaluation.
        
        cv : Optional[pd.DataFrame], default=None
            Cross-validation DataFrame to evaluate. If None, uses self.cv_df.
        
        return_plot : bool, default=False
            If True, returns a faceted bar plot comparing groups.
            If False, returns a DataFrame with results.
        
        plot_model_cats : Optional[List[str]], default=None
            List of model categories for plot ordering. Required if return_plot is True.
        
        **kwargs
            Additional keyword arguments passed to ModelRadarPlotter.error_by_group
            when return_plot is True.
        
        Returns
        -------
        pd.DataFrame or plotnine.ggplot
            DataFrame with groups as columns and models as rows,
            or a plot comparing model performance across groups if return_plot is True.
            
        Raises
        ------
        KeyError
            If group_col does not exist in the DataFrame.
        """

        cv_ = self.cv_df if cv is None else cv

        if group_col not in cv_.columns:
            raise KeyError(f"Column '{group_col}' not found in DataFrame")

        results_by_group = {}
        for group, group_df in cv_.groupby(group_col):
            results_by_group[group] = self.evaluate(group_df)

        results = pd.concat(results_by_group, axis=1)

        if return_plot:
            assert plot_model_cats is not None

            plot = ModelRadarPlotter.error_by_group(data=results,
                                                    model_cats=plot_model_cats,
                                                    **kwargs)

            return plot

        return results
