from typing import List, Optional, Callable

import numpy as np
import pandas as pd

from utilsforecast.evaluation import evaluate as uf_evaluate


class BaseModelRadar:
    COLUMNS = {
        'metric': 'metric',
        'horizon': 'horizon',
        'cutoff': 'cutoff',
    }

    DF_RESULT_COLUMNS = ['Model', 'Result']

    def __init__(self,
                 cv_df: pd.DataFrame,
                 freq: str,
                 metrics: List[Callable],
                 model_names: Optional[List[str]],
                 id_col: str = 'unique_id',
                 time_col: str = 'ds',
                 target_col: str = 'y'):
        """

        :param cv_df:
        :param freq:
        :param metrics: multiple metrics will be averaged
        :param model_names:
        :param id_col:
        :param time_col:
        :param target_col:
        """

        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.freq = freq
        self.metrics = metrics

        self.cv_df = self._reset_on_uid(cv_df)
        self.cv_df = self._set_horizon_on_df(self.cv_df)

        self.train_df = None
        self.meta_data_cols = []

        self.models = self._get_model_names(cv_df) if model_names is None else model_names

        self.added_metadata = [col for col in cv_df.columns
                               if col not in self.models + self.meta_data_cols]

    def _set_horizon_on_df(self, cv: pd.DataFrame) -> pd.DataFrame:
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
        self._set_meta_data()

        meta_cols_j = cv.columns.str.contains('|'.join(self.meta_data_cols))
        models = cv.loc[:, ~meta_cols_j].columns.tolist()

        return models

    def _set_meta_data(self):
        self.meta_data_cols = [self.id_col,
                               self.time_col,
                               self.COLUMNS.get('cutoff'),
                               self.COLUMNS.get('horizon'),
                               self.target_col,
                               'lo', 'hi']

    def _reset_on_uid(self, df: pd.DataFrame):
        if df.index.name == self.id_col:
            df = df.reset_index()

        return df

    @classmethod
    def _to_df_and_rename(cls, s: pd.Series):
        df = s.reset_index()
        df.columns = cls.DF_RESULT_COLUMNS

        return df


class ModelRadarAcrossId:
    """
    class for handling analysis post uid-based evaluations
    """

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

    def expected_shortfall(self, err_df: pd.DataFrame):
        shortfall = err_df.apply(lambda x: x[x > x.quantile(self.cvar_quantile)].mean())

        return shortfall


class ModelRadar(BaseModelRadar):

    def __init__(self,
                 cv_df: pd.DataFrame,
                 freq: str,
                 metrics: List[Callable],
                 model_names: Optional[List[str]],
                 id_col: str = 'unique_id',
                 time_col: str = 'ds',
                 target_col: str = 'y'):

        super().__init__(cv_df=cv_df,
                         freq=freq,
                         metrics=metrics,
                         model_names=model_names,
                         id_col=id_col,
                         time_col=time_col,
                         target_col=target_col)

    def evaluate(self,
                 cv: Optional[pd.DataFrame] = None,
                 keep_uids: bool = False,
                 train_df: Optional[pd.DataFrame] = None):

        cv_ = self.cv_df if cv is None else cv

        scores_df = uf_evaluate(df=cv_,
                                models=self.models,
                                metrics=self.metrics,
                                train_df=train_df)

        if keep_uids:
            scores_df = scores_df.groupby(self.id_col).mean(numeric_only=True)#.reset_index()
        else:
            scores_df = scores_df.drop(columns=[self.id_col, self.COLUMNS.get('metric')]).mean()

        return scores_df

    def evaluate_by_horizon(self,
                            cv: Optional[pd.DataFrame] = None,
                            group_by_freq: bool = False,
                            freq_col: str = 'freq'):

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

        return scores_df

    def evaluate_by_horizon_bounds(self, cv: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        cv_ = self.cv_df if cv is None else cv

        sorted_cv = cv_.sort_values([self.id_col,
                                     self.COLUMNS.get('cutoff'),
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

        return errors_combined

    def evaluate_by_anomaly(self,
                            cv: Optional[pd.DataFrame],
                            mode: str = 'observations',
                            anomaly_col: str = 'is_anomaly'):

        cv_ = self.cv_df if cv is None else cv

        if mode not in ['observations', 'series']:
            raise ValueError("mode must be either 'observations' or 'series'")

        scores_uids = []

        for uid, df_uid in cv_.groupby(self.id_col):
            has_anomalies = df_uid[anomaly_col].sum() > 0

            if has_anomalies:
                df_uid_ = df_uid.loc[df_uid[anomaly_col] > 0, :] if mode == 'observations' else df_uid

                if not df_uid_.empty:
                    scores_uid_ = self.evaluate(df_uid_)
                    scores_uids.append(scores_uid_)

        if len(scores_uids) < 1:
            return None

        scores_df = pd.concat(scores_uids).reset_index(drop=True)

        return scores_df

    def evaluate_by_group(self, cv: pd.DataFrame, group_col: str) -> pd.DataFrame:

        cv_ = self.cv_df if cv is None else cv

        if group_col not in cv_.columns:
            raise KeyError(f"Column '{group_col}' not found in DataFrame")

        results_by_group = {}
        for group, group_df in cv_.groupby(group_col):
            results_by_group[group] = self.evaluate(group_df)

        results = pd.concat(results_by_group, axis=1)

        return results