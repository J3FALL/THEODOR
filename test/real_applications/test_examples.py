import os
from datetime import timedelta

import numpy as np
from sklearn.metrics import mean_squared_error

from examples.clustering import run_clustering_example
from examples.forecasting_model_composing import run_metocean_forecasting_problem
from examples.multiclass_prediction import get_model
from examples.time_series_forecasting import (run_multistep_composite_example, run_multistep_linear_example,
                                              run_multistep_lstm_example, run_multistep_multiscale_example,
                                              run_onestep_linear_example)
from examples.time_series_gapfilling_example import run_gapfilling_example


def test_forecasting_model_composing_example():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, '../data/simple_time_series.csv')
    file_path_test = file_path_train

    rmse = run_metocean_forecasting_problem(file_path_train, file_path_test, with_visualisation=False)
    assert rmse > 0


def test_ts_forecasting_example():
    data_length = 700
    data_length_onestep = 64
    run_onestep_linear_example(n_steps=data_length_onestep, is_visualise=False)
    run_multistep_linear_example(n_steps=data_length, is_visualise=False)
    run_multistep_multiscale_example(n_steps=data_length, is_visualise=False)
    run_multistep_composite_example(n_steps=data_length, is_visualise=False)
    run_multistep_lstm_example(n_steps=data_length, is_visualise=False)


def test_multiclass_example():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, '../data/multiclass_classification.csv')

    chain = get_model(file_path_train, cur_lead_time=timedelta(seconds=1))
    assert chain is not None


def test_gapfilling_example():
    arrays_dict, gap_data, real_data = run_gapfilling_example()

    gap_ids = np.ravel(np.argwhere(gap_data == -100.0))
    for key in arrays_dict.keys():
        arr_without_gaps = arrays_dict.get(key)

        # Get only values in the gap
        predicted_values = arr_without_gaps[gap_ids]
        true_values = real_data[gap_ids]

        model_rmse = mean_squared_error(true_values, predicted_values, squared=False)
        assert model_rmse < 0.5


def test_clustering_example():
    metrics = run_clustering_example(is_fast=True)
    assert all([_ >= 0 for _ in metrics])
