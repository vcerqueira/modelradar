import typing

import numpy as np
import pandas as pd


class LoadDataset:
    DATASET_PATH = 'assets/datasets'
    DATASET_NAME = ''

    horizons = []
    frequency = []
    horizons_map = {}
    frequency_map = {}
    context_length = {}
    frequency_pd = {}
    data_group = [*horizons_map]

    @classmethod
    def load_data(cls, group, min_n_instances: typing.Optional[int] = None):
        pass

    @classmethod
    def load_everything(cls,
                        group,
                        min_n_instances=None,
                        sample_n_uid=None):

        df = cls.load_data(group, min_n_instances)

        horizon = cls.horizons_map.get(group)
        n_lags = cls.context_length.get(group)
        freq_str = cls.frequency_pd.get(group)
        freq_int = cls.frequency_map.get(group)

        if sample_n_uid is not None:
            assert isinstance(df, pd.DataFrame)
            df = cls.sample_first_uids(df, sample_n_uid)

        return df, horizon, n_lags, freq_str, freq_int

    @staticmethod
    def prune_df_by_size(df: pd.DataFrame, min_n_instances: int):
        large_ts = df['unique_id'].value_counts() >= min_n_instances
        large_ts_uid = large_ts[large_ts].index.tolist()

        df = df.query('unique_id== @large_ts_uid').reset_index(drop=True)

        return df

    @staticmethod
    def sample_first_uids(df: pd.DataFrame, n_uid: int):
        uid_sample = df['unique_id'].unique()[:n_uid].tolist()
        df = df.query(f'unique_id==@uid_sample').reset_index(drop=True)

        return df

    @staticmethod
    def dummify_series(df):
        df_uid = df.copy().groupby('unique_id')

        dummied_l = []
        for g, uid_df in df_uid:
            uid_df['y'] = range(uid_df.shape[0])

            dummied_l.append(uid_df)

        dummy_df = pd.concat(dummied_l, axis=0).reset_index(drop=True)

        return dummy_df

    @staticmethod
    def get_uid_tails(df, tail_size: int):
        df_list = []
        for g, df_ in df.groupby('unique_id'):
            df_list.append(df_.tail(tail_size))

        tail_df = pd.concat(df_list, axis=0).reset_index(drop=True)

        return tail_df

    @staticmethod
    def difference_series(df):
        df_uid = df.copy().groupby('unique_id')

        diff_l = []
        for g, uid_df in df_uid:
            uid_df['y'] = uid_df['y'].diff()

            diff_l.append(uid_df.tail(-1))

        diff_df = pd.concat(diff_l, axis=0).reset_index(drop=True)

        return diff_df

    @staticmethod
    def train_test_split(df: pd.DataFrame, horizon: int):
        df_by_unq = df.groupby('unique_id')

        train_l, test_l = [], []
        for g, df_ in df_by_unq:
            df_ = df_.sort_values('ds')

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df
