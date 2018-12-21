import lightgbm as lgb
import numpy as np
from sklearn import model_selection


def get_model(x, y, optimize=False):
    model = lgb.LGBMRegressor(objective='regression',
                              num_leaves=5,
                              learning_rate=0.01,
                              num_iterations=10000,
                              n_estimators=700,
                              max_bin=60,
                              bagging_fraction=0.7,
                              bagging_freq=5,
                              feature_fraction=0.25,
                              feature_fraction_seed=9,
                              bagging_seed=9,
                              min_data_in_leaf=12,
                              min_sum_hessian_in_leaf=11,
                              metric='mse')

    if optimize:
        params = [
            {
                'boosting_type': ["gbdt"],
                'num_leaves': [5, 20, 31],
                'max_depth': [-1, 5, 9],
                'learning_rate': [0.01],
                'n_estimators': [700],
                'num_iterations': [10000],
                'class_weight': [None, 'balanced'],
                'max_bin': [10, 55, 100],
                'bagging_fraction': [0.2, 0.5, 0.8],
                'bagging_freq': [1, 5, 9, 13],
                'feature_fraction': [0.1, 0.2319, 0.5],
                'min_data_in_leaf': [2, 6, 10],
                'min_sum_hessian_in_leaf': [9, 11, 13],
                'feature_fraction_seed': [9],
                'bagging_seed':[9],
                'silent': [False],
            }
        ]
        model = model_selection.GridSearchCV(lgb.LGBMRegressor(),
                                             params,
                                             scoring='neg_mean_squared_error',
                                             n_jobs=-1)
    model.fit(x, np.ravel(y))
    return model
