from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.models.evaluation.custom_models.consensus_clustering \
    import ConsensusClusterer
from fedot.core.models.evaluation.evaluation import EvaluationStrategy


def _most_frequent(data):
    return max(set(data), key=list(data).count)


def consensus_fit(data: InputData, params):
    ensembler = ConsensusClusterer(data)
    ensembler.fit(data)
    return ensembler


def consensus_predict(model, data: InputData):
    return model.predict(data)


class EnsemblingStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        'consensus_ensembler': (consensus_fit, consensus_predict),
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
