import pandas as pd
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from mlforecast import MLForecast
from mlforecast.auto import AutoMLForecast

from exp_utils.load_data.config import DATASETS
from exp_utils.models_config import ModelsConfig

import warnings

warnings.filterwarnings("ignore")

# ---- data loading and partitioning
data_name, group = 'Gluonts', 'm1_monthly'
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=5)
df = data_loader.get_uid_tails(df, tail_size=100)
df = data_loader.dummify_series(df)


train, test = data_loader.train_test_split(df, horizon=horizon)

# ---- model setup
print('...Model setup')


sf = StatsForecast(
    models=ModelsConfig.get_sf_models(season_len=freq_int, input_size=n_lags),
    freq=freq_str,
    n_jobs=1,
)

nf = NeuralForecast(models=ModelsConfig.get_nf_models(horizon=horizon),
                    freq=freq_str)

mlf = MLForecast(
    models=ModelsConfig.get_mlf_models(),
    freq=freq_str,
    lags=range(1, n_lags + 1),
)

auto_mlf = AutoMLForecast(
    models=ModelsConfig.get_amlf_models(),
    freq=freq_str,
    season_length=freq_int
)


# ---- model fitting
sf.fit(df=train)
# sf.forecast(fitted=True, h=1)

print('......ML')
nf.fit(df=train)

mlf.fit(df=train)
mlf.predict(h=horizon)
auto_mlf.fit(df=train, n_windows=2, h=horizon,num_samples=2)

auto_mlf.predict(h=horizon)

# ---- insample forecasts
print('...getting insample predictions')

# fcst_insample_sf = sf.forecast_fitted_values()

n_windows = train['unique_id'].value_counts().min() - n_lags - 1

fcst_cv_sf = sf.cross_validation(df=train,
                                 n_windows=n_windows,
                                 step_size=1,
                                 h=2)
fcst_cv_sf = fcst_cv_sf.reset_index()
fcst_cv_sf = fcst_cv_sf.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff')

# nf.predict_insample(step_size=1)
# predict_insample() not working for nf
fcst_cv_nf = nf.cross_validation(df=train,
                                 n_windows=n_windows,
                                 step_size=1)
fcst_cv_nf = fcst_cv_nf.reset_index()
fcst_cv_nf = fcst_cv_nf.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff')

fcst_cv = fcst_cv_nf.merge(fcst_cv_sf.drop(columns='y'),
                           on=['unique_id', 'ds'])
