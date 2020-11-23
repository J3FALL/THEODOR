import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.data.preprocessing import Scaling
from fedot.core.models.model import Model
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def get_roc_auc(train_data: InputData, train_predicted: list) -> float:
    n_classes = train_data.num_classes
    if n_classes > 2:
        additional_params = {'multi_class': 'ovo', 'average': 'macro'}
    else:
        additional_params = {}

    try:
        roc_on_train = round(roc_auc(y_score=train_predicted,
                                     y_true=train_data.target,
                                     **additional_params), 3)
    except Exception as ex:
        print(ex)
        roc_on_train = 0.5

    return roc_on_train


@pytest.fixture()
def classification_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = InputData(features=x, target=classes, idx=np.arange(0, len(x)),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)

    return data


def classification_dataset_with_redunant_features(
        n_samples=1000, n_features=100, n_informative=5) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples,
                                         n_features=n_features,
                                         n_informative=n_informative)

    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def test_log_regression_fit_correct(classification_dataset):
    data = classification_dataset
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    log_reg = Model(model_type='logit')

    _, train_predicted = log_reg.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_random_forest_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    random_forest = Model(model_type='rf')

    _, train_predicted = random_forest.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_decision_tree_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    decision_tree = Model(model_type='dt')

    decision_tree.fit(data=train_data)
    _, train_predicted = decision_tree.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_lda_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    lda = Model(model_type='lda')

    _, train_predicted = lda.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_qda_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    qda = Model(model_type='qda')

    _, train_predicted = qda.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_log_clustering_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    kmeans = Model(model_type='kmeans')

    _, train_predicted = kmeans.fit(data=train_data)

    assert all(np.unique(train_predicted) == [0, 1, 2])


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_svc_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    svc = Model(model_type='svc')

    _, train_predicted = svc.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


def test_pca_model_removes_redunant_features_correct():
    n_informative = 5
    data = classification_dataset_with_redunant_features(n_samples=1000, n_features=100,
                                                         n_informative=n_informative)
    train_data, test_data = train_test_data_setup(data=data)

    pca = Model(model_type='pca_data_model')
    _, train_predicted = pca.fit(data=train_data)

    assert train_predicted.shape[1] < data.features.shape[1]
