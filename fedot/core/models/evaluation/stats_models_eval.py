from datetime import timedelta
from typing import Optional

import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA

from fedot.core.data.data import InputData, OutputData
from fedot.core.models.evaluation.evaluation import EvaluationStrategy
from fedot.core.models.tuning.tuners import ForecastingCustomRandomTuner


def fit_ar(train_data: InputData, params):
    return AutoReg(train_data.target, **params,
                   exog=train_data.features).fit()


def fit_arima(train_data: InputData, params):
    return ARIMA(train_data.target, **params).fit(disp=0)


def predict_ar(trained_model, predict_data: InputData) -> OutputData:
    prediction = None

    for ts_step_id, target in enumerate(predict_data.features):
        start, end = trained_model.nobs + ts_step_id, \
                     trained_model.nobs + ts_step_id + predict_data.task.task_params.forecast_length
        exog, exog_oos = None, predict_data.features[start:end]

        if trained_model.data.endog is predict_data.target:
            # if train sample used
            start, end = 0 + ts_step_id, predict_data.task.task_params.forecast_length + ts_step_id
            exog, exog_oos = (predict_data.features[start:end],
                              predict_data.features[start:end])

        prediction = trained_model.predict(start=start, end=end,
                                           exog=exog, exog_oos=exog_oos)

    return prediction


def predict_arima(trained_model, predict_data: InputData):
    prediction = []

    forecast_length = predict_data.task.task_params.forecast_length
    prediction_steps = len(predict_data.features) + 1
    for pred_step_id in range(prediction_steps):

        start, end = trained_model.nobs + pred_step_id, \
                     trained_model.nobs + pred_step_id + forecast_length - 1

        if trained_model.data.endog is predict_data.target:
            # if train sample used
            start, end = 0 + pred_step_id, forecast_length + pred_step_id

        # TODO take exog into account
        prediction_for_step = trained_model.predict(start=start, end=end)
        prediction_for_step = prediction_for_step[-forecast_length:]
        prediction.append(prediction_for_step)

    return np.stack(prediction)


class StatsModelsForecastingStrategy(EvaluationStrategy):
    __model_functions_by_types = {
        'arima': (fit_arima, predict_arima),
        'ar': (fit_ar, predict_ar)
    }

    __model_description_by_func = {
        fit_arima: 'statsmodels.tsa.arima_model.ARIMA',
        fit_ar: 'statsmodels.tsa.ar_model import AutoReg'
    }

    __default_params_by_model = {
        'arima': {'order': (2, 0, 0)},
        'ar': {'lags': (1, 2, 6, 12, 24)}
    }
    __params_range_by_model = {
        'arima': {'order': ((2, 0, 0), (5, 0, 5))},
        'ar': {'lags': (range(1, 6), range(6, 96, 6))}
    }

    def __init__(self, model_type: str, params: Optional[dict] = None):
        self._model_specific_fit, self._model_specific_predict = self._init_stats_model_functions(model_type)
        self._params_range = self.__params_range_by_model[model_type]
        self._default_params = self.__default_params_by_model[model_type]
        super().__init__(model_type, params)

    def _init_stats_model_functions(self, model_type: str):
        if model_type in self.__model_functions_by_types.keys():
            return self.__model_functions_by_types[model_type]
        else:
            raise ValueError(f'Impossible to obtain Stats strategy for {model_type}')

    def fit(self, train_data: InputData):
        stats_model = self._model_specific_fit(train_data, self._default_params)
        self.params_for_fit = self._default_params
        return stats_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        return self._model_specific_predict(trained_model, predict_data)

    def fit_tuned(self, train_data: InputData, iterations: int = 10,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        tuned_params = ForecastingCustomRandomTuner().tune(fit=self._model_specific_fit,
                                                           predict=self._model_specific_predict,
                                                           tune_data=train_data,
                                                           params_range=self._params_range,
                                                           default_params=self._default_params,
                                                           iterations=iterations)

        stats_model = self._model_specific_fit(train_data, tuned_params)
        self.params_for_fit = tuned_params

        return stats_model, tuned_params

    @property
    def implementation_info(self) -> str:
        return self.__model_description_by_func[self._model_specific_fit]
