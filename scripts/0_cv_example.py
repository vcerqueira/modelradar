import pandas as pd
from datasetsforecast.m3 import M3
from neuralforecast.models import NHITS, MLP, KAN
from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast

from modelradar.pipelines.data_splits import train_test_split
from modelradar.pipelines.utils import DecompositionSTL

ds, *_ = M3.load('assets/datasets', group='Monthly')

train_df, test_df = train_test_split(ds, horizon=12)

strength_df = train_df.groupby('unique_id').apply(lambda x: DecompositionSTL.get_strengths(x, period=12))
strength_df = pd.DataFrame.from_records(strength_df, index=strength_df.index)
strength_df['trend_str'] = (strength_df['trend_str'] > 0.6).map({False: 'No trend', True: 'With trend'})
strength_df['seasonal_str'] = (strength_df['seasonal_str'] > 0.6).map(
    {False: 'No seasonality', True: 'With seasonality'})

models = [NHITS(h=12, input_size=12, max_steps=1000, accelerator='mps'),
          KAN(h=12, input_size=12, max_steps=1000, accelerator='mps'),
          MLP(h=12, input_size=12, max_steps=1000, accelerator='mps'),
          MLP(h=12, input_size=12, max_steps=1000, accelerator='mps', num_layers=3)]

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

cv = cv.merge(strength_df, on='unique_id', how='left')

cv.to_csv('assets/cv.csv')
