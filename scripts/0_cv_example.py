import pandas as pd
from datasetsforecast.m3 import M3
from neuralforecast.models import NHITS, MLP
from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast

from modelradar.pipelines.data_splits import train_test_split

ds, *_ = M3.load('assets/datasets', group='Monthly')

train_df, test_df = train_test_split(ds, horizon=12)

models = [NHITS(h=12, input_size=12, max_steps=5),
          MLP(h=12, input_size=12, max_steps=5),
          MLP(h=12, input_size=12, max_steps=5, num_layers=3)]

nf = NeuralForecast(models=models, freq='ME')

cv_nf = nf.cross_validation(df=train_df, n_windows=2)

stats_models = [SeasonalNaive(season_length=12)]
sf = StatsForecast(models=stats_models, freq='ME', n_jobs=1)

cv_sf = sf.cross_validation(df=train_df, h=12, level=[99])

cv = cv_nf.merge(cv_sf.drop(columns='y'), on=['unique_id', 'ds', 'cutoff'])


def pi_anomalies(cv: pd.DataFrame):
    is_outside_pi = (cv['y'] >= cv['SeasonalNaive-hi-99']) | (cv['y'] <= cv['SeasonalNaive-lo-99'])
    is_outside_pi = is_outside_pi.astype(int)
    is_anomaly_int = is_outside_pi.astype(int)

    return is_anomaly_int


cv['is_anomaly'] = pi_anomalies(cv)

cv.to_csv('assets/cv.csv')
