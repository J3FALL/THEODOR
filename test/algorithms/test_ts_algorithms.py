import numpy as np

from fedot.core.algorithms.time_series.lagged_features import prepare_lagged_ts_for_prediction
from fedot.core.data.transformation import ts_to_lagged_table
from test.data_operations.test_transform import synthetic_forecasting_problem


def test_lagged_features_preparation():
    task, ts_len, ts_data, ts_data_lagged = \
        synthetic_forecasting_problem()

    lagged_data = ts_to_lagged_table(ts_data)
    lagged_data.features = np.asarray(lagged_data.features)

    prepared_lagged_data_fit = prepare_lagged_ts_for_prediction(lagged_data, is_for_fit=True)

    assert len(prepared_lagged_data_fit.features) < len(lagged_data.features)
    assert len(prepared_lagged_data_fit.target) < len(lagged_data.target)

    prepared_lagged_data_predict = prepare_lagged_ts_for_prediction(lagged_data, is_for_fit=False)

    assert (len(lagged_data.target) >
            len(prepared_lagged_data_predict.features) >
            len(prepared_lagged_data_fit.features) > 0)
