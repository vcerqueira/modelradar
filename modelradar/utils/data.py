import pandas as pd
import numpy as np


class LogTransformation:

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
