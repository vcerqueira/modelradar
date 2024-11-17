from typing import List, Optional

import numpy as np
import pandas as pd

from functools import partial
from utilsforecast.losses import mase, smape, rmae, mae
from utilsforecast.evaluation import evaluate


class BaseModelRadar:
    METADATA = ['unique_id', 'ds', 'cutoff', 'horizon', 'y']

    def __init__(self,
                 cv_df: pd.DataFrame,
                 freq: str,
                 id_col: str = 'unique_id',
                 time_col: str = 'ds',
                 target_col: str = 'y',
                 predictions_col: List[str] = 'y'):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.is_integer_valued = False

        self._assert_datatypes(cv_df, freq)

        self.cv_df = cv_df
        self.train_df = None

    @staticmethod
    def reset_on_uid(df: pd.DataFrame):
        if df.index.name == 'unique_id':
            df = df.reset_index()

        return df

    @staticmethod
    def _set_horizon_on_df(cv: pd.DataFrame) -> pd.DataFrame:
        cv_ = cv.copy()

        groups = ['unique_id', 'cutoff'] if 'cutoff' in cv_.columns else ['unique_id']
        dt_cols = ['ds', 'cutoff'] if 'cutoff' in cv_.columns else ['ds']

        for col in dt_cols:
            if not pd.api.types.is_datetime64_any_dtype(cv_[col]):
                cv_[col] = pd.to_datetime(cv_[col])

        cv_ = cv_.sort_values(groups + ['ds'])

        cv_['horizon'] = cv_.groupby(groups).cumcount() + 1

        return cv_

    @classmethod
    def get_model_names(cls, cv: pd.DataFrame):
        metadata = cv.columns.str.contains('|'.join(cls.METADATA))
        models = cv.loc[:, ~metadata].columns.tolist()

        return models


class ModelRadarAnalysis:

    def __init__(self,
                 reference: Optional[str],
                 hardness_quantile: float = 0.95,
                 cvar_quantile: float = 0.9)
        self.reference = reference
        self.hardness_quantile = hardness_quantile
        self.cvar_quantile = cvar_quantile
        self.hardness_threshold = None
        self.hard_uid: List[str] = []

    def get_hard_uids(self, err_df: pd.DataFrame, return_df: bool = True):
        assert self.reference in err_df.columns

        self.hardness_threshold = err_df[self.reference].quantile(self.hardness_quantile)
        self.hard_uid = err_df.loc[err_df[self.reference] > self.hardness_threshold, :].index.tolist()

        if return_df:
            err_df_h = err_df.loc[self.hard_uid, :]
            return err_df_h

    def calc_expected_shortfall(self, err_df: pd.DataFrame):
        shortfall = err_df.apply(lambda x: x[x > x.quantile(self.cvar_quantile)].mean())

        return shortfall


class ModelRadar(BaseModelRadar):

    def __init__(self,
                 cv_df: pd.DataFrame,
                 freq: str,
                 id_col: str = 'unique_id',
                 time_col: str = 'ds',
                 target_col: str = 'y',
                 predictions_col: List[str] = 'y'):
        super().__init__(cv_df=cv_df,
                         freq=freq,
                         id_col=id_col,
                         time_col=time_col,
                         target_col=target_col,
                         predictions_col=predictions_col)

    def evaluate(self, period: int, train_df: pd.DataFrame):
        pass




class EvaluationWorkflow:
    # todo get metadata from index

    ALL_METADATA = ['unique_id', 'ds', 'cutoff', 'horizon',
                    'hi', 'lo', 'freq', 'y', 'is_anomaly', 'dataset', 'group']
    ORIGINAL_FEATURES = ['is_anomaly', 'horizon', 'unique_id', 'freq']

    def __init__(self,
                 cv: pd.DataFrame,
                 baseline: str,
                 reference: str):
        self.func = smape

        self.baseline = baseline
        self.reference = reference
        self.cv = cv
        self.hard_thr = -1
        self.hard_series = []
        self.hard_scores = pd.DataFrame()
        self.error_on_hard = pd.DataFrame()

        self.map_forecasting_horizon_col()
        self.models = self.get_model_names()

    def eval_by_horizon_full(self, cv_: typing.Optional[pd.DataFrame] = None):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_g = cv.groupby('freq')
        results_by_g = {}
        for g, df in cv_g:
            fh = df['horizon'].sort_values().unique()
            eval_fh = {}
            for h in fh:
                cv_fh = df.query(f'horizon<={h}')

                eval_fh[h] = self.run(cv_fh)

            results = pd.DataFrame(eval_fh).T
            results_by_g[g] = results

        results_df = pd.concat(results_by_g).reset_index()
        results_df = results_df.rename(columns={'level_0': 'Frequency', 'level_1': 'Horizon'})
        results_df = results_df.melt(['Frequency', 'Horizon'])
        results_df = results_df.rename(columns={'variable': 'Model', 'value': 'Error'})

        return results_df

    def eval_by_horizon_first_and_last(self):
        cv_grouped = self.cv.groupby('unique_id')

        first_horizon, last_horizon = [], []
        for g, df in cv_grouped:
            first_horizon.append(df.iloc[0, :])
            last_horizon.append(df.iloc[-1, :])

        first_h_df = pd.concat(first_horizon, axis=1).T
        last_h_df = pd.concat(last_horizon, axis=1).T

        errf_df = self.run(first_h_df, return_df=True)
        errl_df = self.run(last_h_df, return_df=True)
        err_df = errf_df.merge(errl_df, on='Model')
        err_df.columns = ['Model', 'First horizon', 'Last horizon']

        err_melted_df = err_df.melt('Model')
        err_melted_df.columns = ['Model', 'Horizon', 'Error']

        return err_melted_df

    def eval_by_series(self, cv_: typing.Optional[pd.DataFrame] = None):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_group = cv.groupby('unique_id')

        results_by_series = {}
        for g, df in cv_group:
            results_by_series[g] = self.run(df)

        results_df = pd.concat(results_by_series, axis=1).T

        return results_df

    def eval_by_anomalies(self, cv_: typing.Optional[pd.DataFrame] = None):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_group = cv.groupby('unique_id')

        results_by_series, cv_df = {}, []
        for g, df in cv_group:
            # print(g)
            df_ = df.loc[df['is_anomaly_95'] > 0, :]
            if df_.shape[0] > 0:
                cv_df.append(df_)
                results_by_series[g] = self.run(df_)

        cv_df = pd.concat(cv_df).reset_index(drop=True)
        result_all = self.run(cv_df)

        results_df = pd.concat(results_by_series, axis=1).T

        return results_df, result_all

    def eval_by_anomalous_series(self, cv_: typing.Optional[pd.DataFrame] = None):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_group = cv.groupby('unique_id')

        results_by_series, cv_df = {}, []
        for g, df in cv_group:
            # print(g)
            if df['is_anomaly_95'].sum() > 0:
                cv_df.append(df)
                results_by_series[g] = self.run(df)

        cv_df = pd.concat(cv_df).reset_index(drop=True)
        result_all = self.run(cv_df)
        results_df = pd.concat(results_by_series, axis=1).T

        return results_df, result_all


    def eval_by_frequency(self,
                          cv_: typing.Optional[pd.DataFrame] = None,
                          long_format: bool = False):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_group = cv.groupby('freq')

        results_by_freq = {}
        for g, df in cv_group:
            results_by_freq[g] = self.run(df)

        results_df = pd.concat(results_by_freq, axis=1)

        if long_format:
            results_df = results_df.reset_index().melt('index')
            results_df.columns = ['Model', 'Frequency', 'Error']

        return results_df

    def run(self, cv_: typing.Optional[pd.DataFrame] = None, return_df: bool = False):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        evaluation = {}
        for model in self.models:
            # evaluation[model] = self.func(y=cv['y'], y_hat1=cv[model], y_hat2=cv['SNaive'])
            evaluation[model] = self.func(y=cv['y'], y_hat=cv[model])

        evaluation = pd.Series(evaluation)

        if return_df:
            evaluation = evaluation.reset_index()
            evaluation.columns = ['Model', 'Error']

        return evaluation
