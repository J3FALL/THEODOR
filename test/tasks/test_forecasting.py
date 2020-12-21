import numpy as np
import pytest
from statsmodels.tsa.arima_process import ArmaProcess

from fedot.core.algorithms.time_series.prediction import ts_mse
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def get_synthetic_ts_data_period(n_steps=1000, forecast_length=1, max_window_size=50):
    simulated_data = ArmaProcess().generate_sample(nsample=n_steps)
    x1 = np.arange(0, n_steps)
    x2 = np.arange(0, n_steps) + 1

    simulated_data = simulated_data + x1 * 0.0005 - x2 * 0.0001

    periodicity = np.sin(x1 / 50)

    simulated_data = simulated_data + periodicity

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size,
                                    return_all_steps=False))

    data = InputData(idx=np.arange(0, n_steps),
                     features=np.asarray([x1, x2]).T,
                     target=simulated_data,
                     task=task,
                     data_type=DataTypesEnum.ts)

    return train_test_data_setup(data)


def get_synthetic_ts_data_linear(n_steps=1000, forecast_length=1, max_window_size=50):
    simulated_data = np.asarray([float(_) for _ in (np.arange(0, n_steps))])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size,
                                    return_all_steps=False,
                                    make_future_prediction=False))

    data = InputData(idx=np.arange(0, n_steps),
                     features=simulated_data,
                     target=simulated_data,
                     task=task,
                     data_type=DataTypesEnum.ts)

    return train_test_data_setup(data, shuffle_flag=False)


def get_rmse_value(chain: Chain, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = chain.predict(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)

    rmse_value_test = ts_mse(obs=test_data.target, pred=test_pred.predict)
    rmse_value_train = ts_mse(obs=train_data.target, pred=train_pred.predict)

    return rmse_value_train, rmse_value_test, train_pred, test_pred


def get_multiscale_chain(model_trend='lstm', model_residual='ridge'):
    node_trend = PrimaryNode('trend_data_model')
    node_first_trend = SecondaryNode(model_trend,
                                     nodes_from=[node_trend])

    if model_trend == 'lstm':
        # decrease the number of epochs to fit
        node_first_trend.model.params = {'epochs': 1}

    node_residual = PrimaryNode('residual_data_model')
    node_model_residual = SecondaryNode(model_residual,
                                        nodes_from=[node_residual])

    node_final = SecondaryNode('linear', nodes_from=[node_model_residual,
                                                     node_first_trend])

    chain = Chain(node_final)

    return chain


def get_multilinear_chain():
    node_trend = PrimaryNode('trend_data_model')
    node_first_trend = SecondaryNode('linear',
                                     nodes_from=[node_trend])

    node_residual = PrimaryNode('residual_data_model')
    node_model_residual = SecondaryNode('linear',
                                        nodes_from=[node_residual])

    node_final = SecondaryNode('linear', nodes_from=[node_model_residual,
                                                     node_first_trend])

    chain = Chain(node_final)

    return chain


def get_composite_chain(model_first='lasso', model_second='ridge'):
    node_first = PrimaryNode(model_first)
    node_second = PrimaryNode(model_second)

    node_final = SecondaryNode('linear', nodes_from=[node_first,
                                                     node_second])

    chain = Chain(node_final)
    return chain


@pytest.mark.skip("Skipped due to the unstable SVD did not converge error")
def test_arima_chain_fit_correct():
    train_data, test_data = get_synthetic_ts_data_linear(forecast_length=12)

    chain = Chain(PrimaryNode('arima'))

    chain.fit(input_data=train_data)
    rmse_on_train, rmse_on_test, _, _ = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_train < rmse_threshold
    # TODO investigate arima performance of test
    # assert rmse_on_test < rmse_threshold


def test_regression_chain_forecast_onestep_correct():
    train_data, test_data = get_synthetic_ts_data_linear(forecast_length=1, max_window_size=10)

    chain = Chain(PrimaryNode('ridge'))

    chain.fit(input_data=train_data)
    rmse_on_train, rmse_on_test, _, _ = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_train < rmse_threshold
    assert rmse_on_test < rmse_threshold


def test_regression_chain_forecast_multistep_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=20, max_window_size=30)

    chain = Chain(PrimaryNode('ridge'))

    chain.fit(input_data=train_data)
    _, rmse_on_test, _, _ = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_test < rmse_threshold


def test_regression_chain_linear_forecast_multistep_correct():
    train_data, test_data = get_synthetic_ts_data_linear(forecast_length=2, max_window_size=3)

    chain = Chain(PrimaryNode('linear'))

    chain.fit(input_data=train_data)
    rmse_on_train, rmse_on_test, _, _ = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = 0.01
    assert rmse_on_train < rmse_threshold
    assert rmse_on_test < rmse_threshold


def test_regression_chain_period_exog_forecast_multistep_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=2, max_window_size=3)

    chain = Chain(PrimaryNode('linear'))

    chain.fit(input_data=train_data)
    rmse_on_train, rmse_on_test, _, _ = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = 1.5
    assert rmse_on_train < rmse_threshold
    assert rmse_on_test < rmse_threshold


def test_forecasting_regression_composite_fit_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=2, max_window_size=3)

    chain = get_composite_chain(model_first='ridge',
                                model_second='lasso')

    chain.fit(input_data=train_data)
    _, rmse_on_test, _, _ = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target) * 1.3

    assert rmse_on_test < rmse_threshold


def test_forecasting_regression_multiscale_fit_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=2, max_window_size=3)

    chain = get_multiscale_chain(model_trend='ridge',
                                 model_residual='lasso')

    chain.fit(input_data=train_data)
    _, rmse_on_test, _, _ = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_test < rmse_threshold


def test_forecasting_composite_lstm_chain_fit_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=10, max_window_size=10)

    chain = get_multiscale_chain()

    chain.fit(input_data=train_data)
    _, rmse_on_test, _, _ = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)
    assert rmse_on_test < rmse_threshold


def test_forecasting_multilinear_chain_fit_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=10, max_window_size=10)

    chain = get_multilinear_chain()

    chain.fit(input_data=train_data)
    _, rmse_on_test, _, test_prediction = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)
    assert rmse_on_test < rmse_threshold
    assert test_prediction.predict[0] != test_prediction.predict[1]
