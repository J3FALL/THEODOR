import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root


def get_composite_chain():
    chain = TsForecastingChain()
    node_trend = PrimaryNode('trend_data_model')
    node_model_trend = SecondaryNode('linear', nodes_from=[node_trend])

    node_residual = PrimaryNode('residual_data_model')
    node_model_residual = SecondaryNode('linear', nodes_from=[node_residual])

    node_final = SecondaryNode('linear',
                               nodes_from=[node_model_residual, node_model_trend])
    chain.add_node(node_final)
    return chain


def calculate_validation_metric(pred: OutputData, valid: InputData,
                                name: str, is_visualise=False) -> float:
    forecast_length = valid.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict[~np.isnan(pred.predict)]
    real = valid.target[~np.isnan(pred.predict)]

    # plot results
    if is_visualise:
        compare_plot(predicted, real,
                     forecast_length=forecast_length,
                     model_name=name)

    rmse = mse(y_true=real,
               y_pred=predicted,
               squared=False)

    return rmse


def compare_plot(predicted, real, forecast_length, model_name):
    plt.clf()
    _, ax = plt.subplots()
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    ax.legend()
    plt.xlabel('Time, h')
    plt.ylabel('SSH, cm')
    plt.title(f'Sea surface height forecast for {forecast_length} hours with {model_name}')
    plt.show()


def run_metocean_forecasting_problem(train_file_path, test_file_path,
                                     forecast_length=1, max_window_size=32,
                                     is_visualise=False):
    # specify the task to solve
    task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                         TsForecastingParams(forecast_length=forecast_length,
                                             max_window_size=max_window_size))

    full_path_train = os.path.join(str(project_root()), train_file_path)
    dataset_to_train = InputData.from_csv(
        full_path_train, task=task_to_solve, data_type=DataTypesEnum.ts)

    # a dataset for a final validation of the composed model
    full_path_test = os.path.join(str(project_root()), test_file_path)
    dataset_to_validate = InputData.from_csv(
        full_path_test, task=task_to_solve, data_type=DataTypesEnum.ts)

    chain_simple = TsForecastingChain(PrimaryNode('linear'))
    chain_simple.fit(input_data=dataset_to_train, verbose=False)
    rmse_on_valid_simple = calculate_validation_metric(
        chain_simple.predict(dataset_to_validate), dataset_to_validate,
        f'full-simple_{forecast_length}',
        is_visualise=is_visualise)
    print(f'RMSE simple: {rmse_on_valid_simple}')

    chain_composite_lstm = get_composite_chain()
    chain_composite_lstm.fit(input_data=dataset_to_train, verbose=False)
    rmse_on_valid_lstm_only = calculate_validation_metric(
        chain_composite_lstm.predict(dataset_to_validate), dataset_to_validate,
        f'full-lstm-only_{forecast_length}',
        is_visualise=is_visualise)
    print(f'RMSE LSTM composite: {rmse_on_valid_lstm_only}')

    return rmse_on_valid_simple


if __name__ == '__main__':
    # the dataset was obtained from NEMO model simulation for sea surface height

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)

    run_metocean_forecasting_problem(full_path_train, full_path_test,
                                     forecast_length=72, max_window_size=72, is_visualise=True)
