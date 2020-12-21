import datetime
import os

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.fixed_structure_composer import FixedStructureComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root


def get_composite_lstm_chain():
    chain = Chain()
    node_trend = PrimaryNode('trend_data_model')
    node_trend.labels = ["fixed"]
    node_lstm_trend = SecondaryNode('linear', nodes_from=[node_trend])
    node_trend.labels = ["fixed"]
    node_residual = PrimaryNode('residual_data_model')
    node_ridge_residual = SecondaryNode('linear', nodes_from=[node_residual])

    node_final = SecondaryNode('linear',
                               nodes_from=[node_ridge_residual, node_lstm_trend])
    node_final.labels = ["fixed"]
    chain.add_node(node_final)
    return chain


def calculate_validation_metric(pred: OutputData, valid: InputData,
                                name: str, is_visualise: bool) -> float:
    window_size = valid.task.task_params.max_window_size
    forecast_length = valid.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict[window_size + forecast_length:]
    real = valid.target[window_size + forecast_length:]

    # plot results
    if is_visualise:
        compare_plot(predicted, real,
                     forecast_length=forecast_length,
                     model_name=name)

    # the quality assessment for the simulation results
    rmse = mse(y_true=real, y_pred=predicted, squared=False)

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


def run_metocean_forecasting_problem(train_file_path, test_file_path, forecast_length=1,
                                     max_window_size=64, with_visualisation=True):
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

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    ref_chain = get_composite_lstm_chain()

    available_model_types_primary = ['trend_data_model',
                                     'residual_data_model']

    available_model_types_secondary = ['rfr', 'linear',
                                       'ridge', 'lasso']

    composer_requirements = GPComposerRequirements(
        primary=available_model_types_primary,
        secondary=available_model_types_secondary, max_arity=2,
        max_depth=4, pop_size=10, num_of_generations=10,
        crossover_prob=0, mutation_prob=0.8,
        max_lead_time=datetime.timedelta(minutes=20))

    builder = FixedStructureComposerBuilder(task=task_to_solve).with_requirements(composer_requirements).with_metrics(
        metric_function).with_initial_chain(ref_chain)
    composer = builder.build()

    chain = composer.compose_chain(data=dataset_to_train,
                                   is_visualise=False)

    if with_visualisation:
        ComposerVisualiser.visualise(chain)

    chain.fit(input_data=dataset_to_train, verbose=False)
    rmse_on_valid = calculate_validation_metric(
        chain.predict(dataset_to_validate), dataset_to_validate,
        f'full-composite_{forecast_length}',
        is_visualise=with_visualisation)

    print(f'RMSE composite: {rmse_on_valid}')

    return rmse_on_valid


if __name__ == '__main__':
    # the dataset was obtained from NEMO model simulation for sea surface height

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)

    run_metocean_forecasting_problem(full_path_train, full_path_test, forecast_length=1)
