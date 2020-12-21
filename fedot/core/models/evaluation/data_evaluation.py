from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from fedot.core.algorithms.time_series.scale import estimate_period, split_ts_to_components
from fedot.core.data.data import InputData
from fedot.core.models.evaluation.evaluation import EvaluationStrategy

DEFAULT_EXPLAINED_VARIANCE_THR = 0.9
DEFAULT_MIN_EXPLAINED_VARIANCE = 0.01


def get_data(trained_model, predict_data: InputData):
    return predict_data.features


def get_difference(trained_model, predict_data: InputData):
    number_of_inputs = predict_data.features.shape[1]
    if number_of_inputs != 1:
        raise ValueError('Too many inputs for the differential model')
    return predict_data.features[:, 0] - predict_data.target


def fit_decomposition(train_data: InputData, params: Optional[dict]):
    target = train_data.target
    period = estimate_period(target)
    return period


def get_residual(trained_model, predict_data: InputData):
    _, target_residual = split_ts_to_components(trained_model, predict_data)
    return target_residual


def get_trend(trained_model, predict_data: InputData):
    target_trend, _ = split_ts_to_components(trained_model, predict_data)
    return target_trend


def fit_pca(train_data: InputData, params: Optional[dict]):
    if not params:
        pca = PCA(svd_solver='randomized', iterated_power='auto')
    else:
        pca_params = {k: params[k] for k in ['svd_solver', 'iterated_power']}
        pca = PCA(**pca_params)

    pca.fit(train_data.features)

    cum_variance = np.cumsum(pca.explained_variance_ratio_)

    explained_variance_thr = DEFAULT_EXPLAINED_VARIANCE_THR
    min_explained_variance = DEFAULT_MIN_EXPLAINED_VARIANCE

    if params:
        explained_variance_thr = params.get('dim_reduction_expl_thr', explained_variance_thr)
        min_explained_variance = params.get('dim_reduction_min_expl', min_explained_variance)

    components_before_thr = np.where(cum_variance > explained_variance_thr)[0]

    last_ind_cum = min(components_before_thr) if len(components_before_thr) > 0 else 1

    significant_components = \
        np.where(pca.explained_variance_ratio_ < min_explained_variance)[0]

    last_ind = min(significant_components) if len(significant_components) > 0 else 1

    pca.last_component_ind = min(last_ind, last_ind_cum)

    return pca


def predict_pca(pca_model, predict_data: InputData):
    return pca_model.transform(predict_data.features)[:, :(pca_model.last_component_ind + 1)]


class DataModellingStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        'direct_data_model': (None, get_data),
        'trend_data_model': (fit_decomposition, get_trend),
        'residual_data_model': (fit_decomposition, get_residual),
        'pca_data_model': (fit_pca, predict_pca)
    }

    def __init__(self, model_type: str, params: Optional[dict] = None):
        self._model_specific_fit = self._model_functions_by_type[model_type][0]
        self._model_specific_predict = self._model_functions_by_type[model_type][1]
        super().__init__(model_type, params)

    def fit(self, train_data: InputData):
        if not self._model_specific_fit:
            return None
        else:
            return self._model_specific_fit(train_data, self.params_for_fit)

    def predict(self, trained_model, predict_data: InputData):
        return self._model_specific_predict(trained_model, predict_data)

    def fit_tuned(self, **args):
        return None, None
