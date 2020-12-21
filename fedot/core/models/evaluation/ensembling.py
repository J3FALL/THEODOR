from copy import copy
from typing import Optional

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.models.evaluation.evaluation import EvaluationStrategy


def _most_frequent(data):
    return max(set(data), key=list(data).count)


def select_major(_, data: InputData):
    final_prediction = copy(data.features[:, 0])
    for item_ind in range(data.features.shape[0]):
        final_prediction[item_ind] = _most_frequent(data.features[item_ind, :])
    return final_prediction


def mix_equal(model, data: InputData):
    final_prediction = copy(data.features[:, 0])
    for item_ind in range(data.features.shape[0]):
        final_prediction[item_ind] = np.mean(data.features[item_ind, :])
    return final_prediction


class EnsemblingStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        'voting_ensembler': (None, select_major),
        'equal_ensembler': (None, mix_equal)
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
