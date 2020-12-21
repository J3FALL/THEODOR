import numpy as np
from numpy import nan

from fedot.core.data.data import InputData
from fedot.core.data.transformation import direct, ts_to_lagged_table
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

max_window_size = 4
forecast_length = 2


def synthetic_forecasting_problem():
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size))
    ts_len = 10
    ts = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    exog_variable = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0,
                                15.0, 16.0, 17.0, 18.0, 19.0]).reshape(-1, 1)

    ts_data = InputData(idx=range(len(ts)), features=exog_variable, target=ts,
                        task=task, data_type=DataTypesEnum.table)

    # shape is (11, 4, 1)
    exog_variable_3d = np.asarray([[nan, nan, nan, nan],
                                   [10.0, nan, nan, nan],
                                   [11.0, 10.0, nan, nan],
                                   [12.0, 11.0, 10.0, nan],
                                   [13.0, 12.0, 11.0, 10.0],
                                   [14.0, 13.0, 12.0, 11.0],
                                   [15.0, 14.0, 13.0, 12.0],
                                   [16.0, 15.0, 14.0, 13.0],
                                   [17.0, 16.0, 15.0, 14.0],
                                   [18.0, 17.0, 16.0, 15.0],
                                   [19.0, 18.0, 17.0, 16.0]])

    lagged_target_3d_as_feature = np.asarray([[nan, nan, nan, nan],
                                              [0.0, nan, nan, nan],
                                              [1.0, 0.0, nan, nan],
                                              [2.0, 1.0, 0.0, nan],
                                              [3.0, 2.0, 1.0, 0.0],
                                              [4.0, 3.0, 2.0, 1.0],
                                              [5.0, 4.0, 3.0, 2.0],
                                              [6.0, 5.0, 4.0, 3.0],
                                              [7.0, 6.0, 5.0, 4.0],
                                              [8.0, 7.0, 6.0, 5.0],
                                              [9.0, 8.0, 7.0, 6.0]])
    # [[6.0], [7.0], [8.0], [9.0]]
    # [[7.0], [8.0], [9.0], [nan]],
    # [[8.0], [9.0], [nan], [nan]],
    # [[9.0], [nan], [nan], [nan]],
    # [[nan], [nan], [nan], [nan]]])

    # now we concat exog lagged variables as well as target lagged
    # so features now is (11, 4, 2), i.e.
    # (n+max_window_size+forecast_length, max_window_size, amount_exog_features+target_shape)
    feature_3d = np.concatenate((lagged_target_3d_as_feature, exog_variable_3d), axis=1)

    # lagged format contains only the values to forecast (future values) in the target
    # this format convinient to use with classic regression modules
    # shape is (5, 2), i.e. (n-max_window_size-forecast_length+1, forecast_length * target_shape)
    target_lagged = np.asarray([[0.0, 1.0],
                                [1.0, 2.0],
                                [2.0, 3.0],
                                [3.0, 4.0],
                                [4.0, 5.0],
                                [5.0, 6.0],
                                [6.0, 7.0],
                                [7.0, 8.0],
                                [8.0, 9.0],
                                [9.0, nan],
                                [nan, nan]])

    # in lagged format all features are as feature_3d, but in 2d format
    # with shape (n, max_window_size * (amount_exog_features + target_shape))
    features_lagged = feature_3d.reshape(feature_3d.shape[0], -1)

    ts_data_lagged = InputData(idx=range(len(ts)), features=features_lagged, target=target_lagged,
                               task=task, data_type=DataTypesEnum.ts_lagged_table)

    return task, ts_len, ts_data, ts_data_lagged


def test_ts_to_lagged_table():
    task, ts_len, ts_data, ts_data_lagged = \
        synthetic_forecasting_problem()

    transformed_data = ts_to_lagged_table(ts_data)

    assert np.equal(transformed_data.target.shape,
                    [ts_len + 1,
                     forecast_length]).all()
    assert np.allclose(transformed_data.target, ts_data_lagged.target, equal_nan=True)

    # 2 is equal to amount_exog_features + target_shape
    assert np.equal(transformed_data.features.shape,
                    [ts_len + 1,
                     max_window_size * 2]).all()

    assert np.allclose(transformed_data.features,
                       ts_data_lagged.features, equal_nan=True)


def test_direct():
    task, ts_len, ts_data, ts_data_lagged = \
        synthetic_forecasting_problem()

    transformed_data = direct(ts_data)

    print(transformed_data.target.shape)
    assert np.equal(transformed_data.target.shape,
                    [ts_len]).all()
    assert np.equal(transformed_data.target, ts_data.target).all()
    print(transformed_data.features.shape)
