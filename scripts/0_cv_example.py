import pandas as pd
from datasetsforecast.m3 import M3
from neuralforecast.models import NHITS, MLP, KAN
from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast

from modelradar.utils.data import train_test_split_horizon
from cardtale.analytics.testing.card.trend import DifferencingTests

ds, *_ = M3.load('assets/datasets', group='Monthly')

train_df, test_df = train_test_split_horizon(ds, horizon=12)

features_l = []
for uid, uid_df in train_df.groupby('unique_id'):
    try:
        trend = DifferencingTests.ndiffs(uid_df['y'], test='kpss', test_type='level')
    except OverflowError:
        trend = 0

    seas = DifferencingTests.nsdiffs(uid_df['y'], test='seas', period=12)

    trend_str = 'Non-stationary' if trend > 0 else 'Stationary'
    seas_str = 'Seasonal' if seas > 0 else 'Non-seasonal'

    features_l.append({
        'unique_id': uid,
        'trend_str': trend_str,
        'seas_str': seas_str,
    })

features_df = pd.DataFrame(features_l)

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

is_outside_pi = (cv['y'] >= cv['SeasonalNaive-hi-99']) | (cv['y'] <= cv['SeasonalNaive-lo-99'])

cv['is_anomaly'] = is_outside_pi.astype(int)

cv = cv.merge(features_df, on='unique_id', how='left')

cv.to_csv('assets/cv.csv')
