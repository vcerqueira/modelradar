import pandas as pd
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast

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


# ---- model fitting
sf.fit(df=train)
# sf.forecast(fitted=True, h=1)

print('......ML')
nf.fit(df=train)

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

# ---- fitting ensembles
print('...fitting ensembles')
trim = 0.5

combiners_by_uid = {
    'MLpol': MLpol(loss_type='square', gradient=True, trim_ratio=trim, weight_by_uid=True),
    'MLpol2': MLpol(loss_type='square', gradient=False, trim_ratio=trim, weight_by_uid=True),
    'MLewa': MLewa(loss_type='square', gradient=True, trim_ratio=trim, weight_by_uid=True),
    'MLewa2': MLewa(loss_type='square', gradient=False, trim_ratio=trim, weight_by_uid=True),
    'ADE': ADE(freq=freq_str, meta_lags=list(range(1, ADE_LAGS)), trim_ratio=trim, trim_by_uid=True),
    'LossOnTrain': LossOnTrain(trim_ratio=trim, weight_by_uid=True),
    'BestOnTrain': BestOnTrain(select_by_uid=True),
    'EqAverage': EqAverage(select_by_uid=True, trim_ratio=trim),
    'Windowing': Windowing(freq=freq_str, trim_ratio=trim, select_best=False, weight_by_uid=True),
    'BLAST': Windowing(freq=freq_str, trim_ratio=trim, select_best=True, weight_by_uid=True),
}

# combiners_uncond = {
#     'MLpol': MLpol(loss_type='square', gradient=True, trim_ratio=trim, weight_by_uid=False),
#     'MLpol2': MLpol(loss_type='square', gradient=False, trim_ratio=trim, weight_by_uid=True),
#     'MLewa': MLewa(loss_type='square', gradient=True, trim_ratio=trim, weight_by_uid=False),
#     'ADE': ADE(freq=freq_str, meta_lags=list(range(1, ADE_LAGS)), trim_ratio=trim, trim_by_uid=False),
#     'LossOnTrain': LossOnTrain(trim_ratio=trim, weight_by_uid=False),
#     'BestOnTrain': BestOnTrain(select_by_uid=False),
#     'EqAverage': EqAverage(select_by_uid=False, trim_ratio=trim),
#     'Windowing': Windowing(freq=freq_str, trim_ratio=trim, select_best=False, weight_by_uid=False),
#     'BLAST': Windowing(freq=freq_str, trim_ratio=trim, select_best=True, weight_by_uid=False),
# }

for k in combiners_by_uid:
    print(k)
    # combiners_uncond[k].fit(fcst_cv)
    print(k)
    combiners_by_uid[k].fit(fcst_cv)

# ---- test forecasts
print('...test forecasts')

fcst_sf = sf.predict(h=horizon)
fcst_ml = nf.predict()

fcst = fcst_ml.merge(fcst_sf, on=['unique_id', 'ds']).reset_index()

print('...ensemble forecasts')

ensembles = {}
for k in combiners_by_uid:
    print(k)
    if k == 'ADE':
        fc_uid = combiners_by_uid[k].predict(fcst, train=train, h=horizon)
        # fc = combiners_uncond[k].predict(fcst, train=train, h=horizon)
    else:
        fc_uid = combiners_by_uid[k].predict(fcst)
        # fc = combiners_uncond[k].predict(fcst)

    # ensembles[k] = fc
    ensembles[f'{k}(uid)'] = fc_uid

ensembles_df = pd.DataFrame(ensembles)

fcst_df = pd.concat([fcst, ensembles_df], axis=1)
fcst_df = fcst_df.merge(test, on=['unique_id', 'ds'])

print('...saving results')
# ---- saving results

fcst_df.to_csv(f'scripts/experiments/ensembles/results/{data_name}_{group}.csv', index=False)
