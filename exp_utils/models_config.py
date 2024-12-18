import os

import pandas as pd
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from mlforecast import MLForecast
from mlforecast import MLForecast
from mlforecast.auto import AutoMLForecast

from neuralforecast.models import (NHITS,
                                   LSTM,
                                   GRU,
                                   NBEATS,
                                   DilatedRNN,
                                   MLP,
                                   TCN,
                                   TiDE)

from neuralforecast.auto import (AutoGRU,
                                 AutoKAN,
                                 AutoMLP,
                                 AutoLSTM,
                                 AutoDLinear,
                                 AutoNHITS,
                                 AutoiTransformer,
                                 AutoInformer,
                                 AutoTCN,
                                 AutoDilatedRNN)

import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from mlforecast.auto import (AutoLasso,
                             AutoRidge,
                             AutoXGBoost,
                             AutoLightGBM,
                             AutoElasticNet)

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


class ModelsConfig:

    @staticmethod
    def get_sf_models(season_len: int, input_size: int):
        models = [
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

        return models

    @staticmethod
    def get_mlf_models():
        models_ml = {
            'DT': DecisionTreeRegressor(max_depth=5),
            'RF': xgb.XGBRFRegressor(n_estimators=25),
            'KNN': KNeighborsRegressor(n_neighbors=50),
            'LGBl': lgb.LGBMRegressor(verbosity=-1, n_jobs=1, linear_tree=True),
        }

        return models_ml

    @staticmethod
    def get_amlf_models():
        auto_models_ml = {
            'Ridge': AutoRidge(),
            'Lasso': AutoLasso(),
            'Elastic-net': AutoElasticNet(),
            'XGB': AutoXGBoost(),
            'LGB': AutoLightGBM(),
        }

        return auto_models_ml

    @staticmethod
    def get_nf_models(horizon):
        models = [
            AutoKAN(h=horizon),
            AutoMLP(h=horizon),
        ]
        return models
