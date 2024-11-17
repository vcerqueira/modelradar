from typing import List, Optional

import numpy as np
import pandas as pd

from functools import partial
from utilsforecast.losses import mase, smape, rmae, mae
from utilsforecast.evaluation import evaluate


class BaseModelRadar:
    METADATA = ['unique_id', 'ds', 'cutoff', 'horizon', 'y']
    HORIZON_COL = 'horizon'

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
        self.freq = freq
        self.predictions_col = predictions_col
        self.cv_df = cv_df
        self.train_df = None

        self.models = self.get_model_names(cv_df)

        self.added_metadata = [col for col in cv_df.columns
                               if col not in self.models + self.METADATA]

    @staticmethod
    def reset_on_uid(df: pd.DataFrame):
        if df.index.name == 'unique_id':
            df = df.reset_index()

        return df

    @classmethod
    def _set_horizon_on_df(cls, cv: pd.DataFrame) -> pd.DataFrame:
        cv_ = cv.copy()

        groups = ['unique_id', 'cutoff'] if 'cutoff' in cv_.columns else ['unique_id']
        dt_cols = ['ds', 'cutoff'] if 'cutoff' in cv_.columns else ['ds']

        for col in dt_cols:
            if not pd.api.types.is_datetime64_any_dtype(cv_[col]):
                cv_[col] = pd.to_datetime(cv_[col])

        cv_ = cv_.sort_values(groups + ['ds'])

        cv_[cls.HORIZON_COL] = cv_.groupby(groups).cumcount() + 1

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
                 cvar_quantile: float = 0.9):
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

    def evaluate_by_horizon(self, cv: pd.DataFrame, group_by_freq: bool = True):

        if group_by_freq:
            cv_groups = cv.groupby('freq')
        else:
            cv_groups = [(None, cv)]

        scores_by_group = {}
        for g, df in cv_groups:
            fh = df[self.HORIZON_COL].sort_values().unique()
            eval_horizon = {}
            for h in fh:
                cv_fh = df.query(f'{self.HORIZON_COL}<={h}')
                eval_horizon[h] = self.run(cv_fh)

            results = pd.DataFrame(eval_horizon).T
            scores_by_group[g] = results

        scores_df = pd.concat(scores_by_group)

        if group_by_freq:
            scores_df = scores_df.reset_index()
        else:
            scores_df = scores_df.reset_index(drop=True)

        return scores_df

    def evaluate_by_horizon_bounds(self, cv: pd.DataFrame) -> pd.DataFrame:

        sorted_cv = cv.sort_values(['unique_id', 'horizon'])

        first_horizon = sorted_cv.groupby('unique_id').first().reset_index()
        last_horizon = sorted_cv.groupby('unique_id').last().reset_index()

        errors_first = self.run(first_horizon)
        errors_last = self.run(last_horizon)

        errors_combined = (
            errors_first.merge(errors_last, on='Model')
            .rename(columns={
                errors_first.columns[1]: 'First horizon',
                errors_last.columns[1]: 'Last horizon'
            })
        )

        return errors_combined

    def eval_by_anomaly(self,
                        cv: pd.DataFrame,
                        mode: str = 'observations',
                        anomaly_col: str = 'is_anomaly'):

        if mode not in ['observations', 'series']:
            raise ValueError("mode must be either 'observations' or 'series'")

        scores_uids = []

        for uid, df_uid in cv.groupby('unique_id'):
            has_anomalies = df_uid[anomaly_col].sum() > 0

            if has_anomalies:
                df_uid_ = df_uid.loc[df_uid[anomaly_col] > 0, :] if mode == 'observations' else df_uid

                if not df_uid_.empty:
                    scores_uid_ = self.run(df_uid_)
                    scores_uids.append(scores_uid_)

        if len(scores_uids) < 1:
            return None

        scores_df = pd.concat(scores_uids).reset_index(drop=True)

        return scores_df

    def eval_by_group(self, cv: pd.DataFrame, group_col: str) -> pd.DataFrame:

        if group_col not in cv.columns:
            raise KeyError(f"Column '{group_col}' not found in DataFrame")

        results_by_group = {}
        for group, group_df in cv.groupby(group_col):
            results_by_group[group] = self.run(group_df)

        results = pd.concat(results_by_group, axis=1)

        return results
