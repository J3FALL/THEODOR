from copy import copy

import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import TaskTypesEnum


class TsForecastingChain(Chain):

    def forecast(self, initial_data: InputData, supplementary_data: InputData) -> OutputData:
        """Generates the time series forecast with a sliding window using pre-fitted chain.
        :param initial_data: the initial condition for the forecasting (should be greater or equals to max_window_size)
        :param supplementary_data: the data that should be available during the forecast:
            idx for the forecasted steps and optional exogenous variables
            (variables that are received from an external source instead of forecasting in place and
            used as features of the forecasting model to increase the quality of of forecast)
        :return: forecasted time series
        """

        if not self.is_all_cache_actual():
            raise ValueError('Chain for the time series forecasting was not fitted yet.')

        if supplementary_data.task.task_type is not TaskTypesEnum.ts_forecasting:
            raise ValueError('TsForecastingChain can be used for the ts_forecasting task only.')

        supplementary_data_for_forecast = copy(supplementary_data)
        supplementary_data_for_forecast.task.task_params.make_future_prediction = True

        initial_data_for_forecast = copy(initial_data)
        initial_data_for_forecast.task.task_params.make_future_prediction = True

        forecast_length = supplementary_data_for_forecast.task.task_params.forecast_length

        # check if predict features contains additional (exogenous) variables
        with_exog = supplementary_data_for_forecast.features is not None

        # initial data for the first prediction
        pre_history_start = (len(initial_data_for_forecast.idx) -
                             initial_data_for_forecast.task.task_params.max_window_size)
        pre_history_end = len(initial_data_for_forecast.idx)

        data_for_forecast = initial_data_for_forecast.subset(start=pre_history_start, end=pre_history_end)

        full_prediction = []
        forecast_steps_num = int(np.ceil(len(supplementary_data_for_forecast.idx) / forecast_length))
        for forecast_step in range(forecast_steps_num):
            stepwise_prediction = self.predict(data_for_forecast).predict
            if len(stepwise_prediction.shape) > 1:
                # multi-dim prediction
                stepwise_prediction = stepwise_prediction[-1, :-forecast_length]
                full_prediction.extend(stepwise_prediction)
            else:
                # single-dim prediction
                stepwise_prediction = list(stepwise_prediction[-forecast_length:])
                full_prediction.extend(stepwise_prediction)

            # add additional variable from external source
            if with_exog:
                data_for_forecast = _prepare_exog_features(data_for_forecast, supplementary_data_for_forecast,
                                                           stepwise_prediction,
                                                           forecast_step, forecast_length)
            else:
                predicted_ts = np.append(data_for_forecast.target, stepwise_prediction)
                data_for_forecast.target = np.stack(predicted_ts)
                data_for_forecast.features = data_for_forecast.target

            data_for_forecast.idx = _extend_idx_for_prediction(data_for_forecast.idx, forecast_length)

        full_prediction = full_prediction[0:len(supplementary_data_for_forecast.idx)]

        output_data = OutputData(idx=supplementary_data_for_forecast.idx,
                                 features=supplementary_data_for_forecast.features,
                                 predict=np.asarray(full_prediction), task=supplementary_data_for_forecast.task,
                                 data_type=supplementary_data_for_forecast.data_type)

        return output_data


def _extend_idx_for_prediction(exiting_idx, forecast_length):
    if forecast_length > 1:
        indices_for_forecast = (exiting_idx[-1] +
                                list(range(1, forecast_length + 1)))

    else:
        indices_for_forecast = exiting_idx[-1] + 1

    new_idx = np.append(exiting_idx, indices_for_forecast)
    return new_idx


def _prepare_exog_features(data_for_prediction: InputData,
                           exog_data: InputData,
                           last_prediction: np.array,
                           forecast_step: int, forecast_length: int) -> InputData:
    new_features = []
    if len(data_for_prediction.features.shape) == 1:
        # if one exog feature
        exog_features_num = 1
    else:
        # if several exog features
        exog_features_num = data_for_prediction.features.shape[1]

    new_part_len = 0
    if exog_features_num > 1:
        for exog_feat_id in range(exog_features_num):
            exog_feature = data_for_prediction.features[:, exog_feat_id]
            new_exog_values = \
                exog_data.features[forecast_step * forecast_length:
                                   ((forecast_step + 1) * forecast_length), exog_feat_id]
            new_feature = np.append(exog_feature, new_exog_values)
            new_features.append(new_feature)
            new_part_len = len(new_features[0])
    else:
        exog_feature = data_for_prediction.features
        new_exog_values = \
            exog_data.features[forecast_step * forecast_length:
                               ((forecast_step + 1) * forecast_length)]
        new_features = np.append(exog_feature, new_exog_values)
        new_part_len = len(new_features)

    # add predicted time series to features for next prediction
    predicted_ts = np.append(data_for_prediction.target,
                             last_prediction)

    # cut the prediction if it's too long (actual for the last forecast step)
    predicted_ts = predicted_ts[0: new_part_len]
    # new_features.append(predicted_ts)
    data_for_prediction.target = predicted_ts
    data_for_prediction.features = np.stack(np.asarray(new_features)).T

    return data_for_prediction


def convert_to_tschain(chain):
    """
    The method convert base chain object into TsChain
    """

    chain.__class__ = TsForecastingChain
    return chain
