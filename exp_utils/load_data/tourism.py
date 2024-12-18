import os

import pandas as pd
import numpy as np

from exp_utils.load_data.base import LoadDataset


class TourismDataset(LoadDataset):
    DATASET_PATH = 'assets/datasets/tourism/'
    DATASET_NAME = 'T'

    horizons_map = {
        'Quarterly': 8,
        'Monthly': 12
    }

    frequency_map = {
        'Quarterly': 4,
        'Monthly': 12
    }

    context_length = {
        'Quarterly': 8,
        'Monthly': 24
    }

    min_samples = {
        'Quarterly': 36,
        'Monthly': 100,
        'Yearly': 10,
    }

    frequency_pd = {
        'Quarterly': 'QS',
        'Monthly': 'MS'
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group, min_n_instances=None):

        assert group in cls.data_group

        ds = {}
        train = pd.read_csv(os.path.join(cls.DATASET_PATH, f'{group.lower()}_in.csv'),
                            header=0, delimiter=",")
        test = pd.read_csv(os.path.join(cls.DATASET_PATH, f'{group.lower()}_oos.csv'),
                           header=0, delimiter=",")

        if group == 'Yearly':
            train_meta = train[:2]
            meta_length = train_meta.iloc[0].astype(int)
            test = test[2:].reset_index(drop=True).T
            train = train[2:].reset_index(drop=True).T
        else:
            train_meta = train[:3]
            meta_length = train_meta.iloc[0].astype(int)
            test = test[3:].reset_index(drop=True).T
            train = train[3:].reset_index(drop=True).T

        train_set = [ts[:ts_length] for ts, ts_length in zip(train.values, meta_length)]
        test_set = [ts[:ts_length] for ts, ts_length in zip(test.values, meta_length)]

        for i, idx in enumerate(train.index):
            ds[idx] = np.concatenate([train_set[i], test_set[i]])

        max_len = np.max([len(x) for k, x in ds.items()])
        idx = pd.date_range(end=pd.Timestamp('2023-11-01'),
                            periods=max_len,
                            freq=cls.frequency_pd[group])

        ds = {k: pd.Series(series, index=idx[-len(series):]) for k, series in ds.items()}
        df = pd.concat(ds, axis=1)
        df = df.reset_index().melt('index').dropna().reset_index(drop=True)
        df.columns = ['ds', 'unique_id', 'y']

        if min_n_instances is not None:
            df = cls.prune_df_by_size(df, min_n_instances)

        return df
