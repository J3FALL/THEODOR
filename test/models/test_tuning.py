import os
from datetime import timedelta

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc_auc

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.data.preprocessing import Scaling
from fedot.core.models.model import Model
from fedot.core.models.tuning.tuners import get_random_params
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.tasks.test_forecasting import get_synthetic_ts_data_period


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = '../data/advanced_classification.csv'
    return InputData.from_csv(os.path.join(test_file_path, file))


@pytest.fixture()
def regression_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = '../data/advanced_regression.csv'
    data = InputData.from_csv(os.path.join(test_file_path, file))
    data.task = Task(TaskTypesEnum.regression)
    return data


@pytest.fixture()
def scoring_dataset():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    return train_data, test_data


@pytest.mark.skip('depends on random values generation')
@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_knn_classification_tune_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    knn = Model(model_type='knn')
    model, _ = knn.fit(data=train_data)
    test_predicted = knn.predict(fitted_model=model, data=test_data)

    roc_on_test = roc_auc(y_true=test_data.target,
                          y_score=test_predicted)

    roc_on_test_tuned_list = []
    for _ in range(3):
        knn_for_tune = Model(model_type='knn')
        model, _ = knn_for_tune.fine_tune(data=train_data, iterations=10,
                                          max_lead_time=timedelta(minutes=1))

        test_predicted_tuned = knn_for_tune.predict(fitted_model=model, data=test_data)

        roc_on_test_tuned = roc_auc(y_true=test_data.target,
                                    y_score=test_predicted_tuned)

        roc_on_test_tuned_list.append(roc_on_test_tuned)

    roc_threshold = 0.6
    assert np.array(roc_on_test_tuned_list).any() >= roc_on_test > roc_threshold


@pytest.mark.skip("Investigate performance")
def test_arima_tune_correct():
    data = get_synthetic_ts_data_period()
    train_data, test_data = train_test_data_setup(data=data)

    arima_for_tune = Model(model_type='arima')
    model, _ = arima_for_tune.fine_tune(data=train_data, iterations=5,
                                        max_lead_time=timedelta(minutes=0.1))

    test_predicted_tuned = arima_for_tune.predict(fitted_model=model, data=test_data)

    rmse_on_test_tuned = mse(y_true=test_data.target,
                             y_pred=test_predicted_tuned, squared=False)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_test_tuned < rmse_threshold


@pytest.mark.skip('depends on random values generation, AssertionError')
@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_rf_class_tune_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    rf = Model(model_type='rf')

    model, _ = rf.fit(train_data)
    test_predicted = rf.predict(fitted_model=model, data=test_data)

    test_roc_auc = roc_auc(y_true=test_data.target,
                           y_score=test_predicted)

    model_tuned, _ = rf.fine_tune(data=train_data, iterations=12,
                                  max_lead_time=timedelta(minutes=0.1))
    test_predicted_tuned = rf.predict(fitted_model=model_tuned, data=test_data)

    test_roc_auc_tuned = roc_auc(y_true=test_data.target,
                                 y_score=test_predicted_tuned)
    roc_threshold = 0.7

    assert test_roc_auc_tuned >= test_roc_auc
    assert test_roc_auc_tuned > roc_threshold


@pytest.mark.parametrize('data_fixture', ['scoring_dataset'])
def test_scoring_logreg_tune_correct(data_fixture, request):
    train_data, test_data = request.getfixturevalue(data_fixture)

    train_data.features = Scaling().fit(train_data.features).apply(train_data.features)
    test_data.features = Scaling().fit(test_data.features).apply(test_data.features)

    logreg = Model(model_type='logit')

    model, _ = logreg.fit(train_data)
    test_predicted = logreg.predict(fitted_model=model, data=test_data)

    test_roc_auc = roc_auc(y_true=test_data.target,
                           y_score=test_predicted)

    logreg_for_tune = Model(model_type='logit')

    model_tuned, _ = logreg_for_tune.fine_tune(data=train_data, iterations=50,
                                               max_lead_time=timedelta(minutes=0.1))
    test_predicted_tuned = logreg_for_tune.predict(fitted_model=model_tuned, data=test_data)

    test_roc_auc_tuned = roc_auc(y_true=test_data.target,
                                 y_score=test_predicted_tuned)

    roc_threshold = 0.6

    assert round(test_roc_auc_tuned, 2) >= round(test_roc_auc, 2) > roc_threshold


def test_get_random_params_constant_length():
    test_param_range = {'param': ((1, 2, 3), (4, 5, 6))}
    random_param_range = get_random_params(test_param_range)
    assert len(random_param_range['param']) == len(test_param_range['param'][0])


def test_get_random_params_varied_length():
    test_param_range = {'param': (list((1, 2, 3)), list((4, 5, 6)))}
    random_param_range = get_random_params(test_param_range)
    assert len(random_param_range['param']) != len(test_param_range['param'][0])


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_classification_manual_tuning_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    knn = Model(model_type='knn')
    model, _ = knn.fit(data=train_data)
    test_predicted = knn.predict(fitted_model=model, data=test_data)

    knn_for_tune = Model(model_type='knn')
    knn_for_tune.params = {'n_neighbors': 1}
    model, _ = knn_for_tune.fit(data=train_data)

    test_predicted_tuned = knn_for_tune.predict(fitted_model=model, data=test_data)

    assert not np.array_equal(test_predicted, test_predicted_tuned)


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_pca_manual_tuning_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    pca = Model(model_type='pca_data_model')
    model, _ = pca.fit(data=train_data)
    test_predicted = pca.predict(fitted_model=model, data=test_data)

    pca_for_tune = Model(model_type='pca_data_model')

    pca_for_tune.params = {'svd_solver': 'randomized',
                           'iterated_power': 'auto',
                           'dim_reduction_expl_thr': 0.7,
                           'dim_reduction_min_expl': 0.001}

    model, _ = pca_for_tune.fit(data=train_data)
    test_predicted_tuned = pca_for_tune.predict(fitted_model=model, data=test_data)

    assert not np.array_equal(test_predicted, test_predicted_tuned)
