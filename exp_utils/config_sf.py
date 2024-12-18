import os

import pandas as pd
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from mlforecast import MLForecast
from statsforecast.models import (
    SeasonalNaive,
    AutoETS,
    AutoARIMA,
    RandomWalkWithDrift,
    AutoTheta,
    SimpleExponentialSmoothingOptimized,
    CrostonOptimized,
    SeasonalWindowAverage,
    WindowAverage
)

season_len = 12
input_size = 12

cls_models = [
    RandomWalkWithDrift(),
    SeasonalNaive(season_length=season_len),
    AutoETS(season_length=season_len),
    AutoARIMA(season_length=season_len, max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1, nmodels=5),
    AutoTheta(season_length=season_len),
    SimpleExponentialSmoothingOptimized(),
    CrostonOptimized(),
    SeasonalWindowAverage(season_length=season_len, window_size=input_size),
    WindowAverage(window_size=input_size),
]

#
# # anomalies
# an_sf = StatsForecast(
#     models=[SeasonalNaive(season_length=season_len)],
#     freq=freq,
#     n_jobs=1,
# )
#
# cv_an = \
#     an_sf.cross_validation(df=df,
#                            h=h,
#                            # test_size=h * N_HORIZONS,
#                            test_size=h,
#                            n_windows=None,
#                            level=[95, 99])
#
