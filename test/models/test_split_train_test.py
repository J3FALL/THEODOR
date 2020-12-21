import random
from copy import deepcopy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

np.random.seed(1)
random.seed(1)
N_SAMPLES = 10000
N_FEATURES = 10
CORRECT_MODEL_AUC_THR = 0.25


def generate_chain() -> Chain:
    chain = Chain(PrimaryNode('logit'))
    return chain


def get_roc_auc_value(chain: Chain, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = chain.predict(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)
    roc_auc_value_test = roc_auc(y_true=test_data.target, y_score=test_pred.predict)
    roc_auc_value_train = roc_auc(y_true=train_data.target, y_score=train_pred.predict)

    return roc_auc_value_train, roc_auc_value_test


def get_synthetic_input_data(n_samples=10000, n_features=10, random_state=None) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples,
                                         n_features=n_features, random_state=random_state)
    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def get_random_target_data(data: InputData) -> InputData:
    data_copy = deepcopy(data)
    data_copy.target = np.array([random.choice((0, 1)) for _ in range(len(data.target))])

    return data_copy


def get_auc_threshold(roc_auc_value: float) -> float:
    return abs(roc_auc_value - 0.5)


def test_model_fit_and_predict_correctly():
    """Checks whether the model fits and predict correctly on the synthetic dataset"""
    data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=1)

    chain = generate_chain()
    train_data, test_data = train_test_data_setup(data)

    chain.fit(input_data=train_data)
    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert train_auc_thr >= CORRECT_MODEL_AUC_THR
    assert test_auc_thr >= CORRECT_MODEL_AUC_THR


def test_model_fit_correctly_but_predict_incorrectly():
    """Check that the model can fit the train dataset but
    can't predict the test dataset. Train and test are supposed to be
    from different distributions."""
    train_data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=1)
    test_data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=2)
    test_data.features = deepcopy(train_data.features)

    chain = generate_chain()
    chain.fit(input_data=train_data)
    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr <= CORRECT_MODEL_AUC_THR
    assert train_auc_thr >= CORRECT_MODEL_AUC_THR


def test_model_fit_correctly_but_random_predictions_on_test():
    """Checks whether the model can fit train dataset correctly, but
    the roc_auc_score on the test dataset is close to 0.5 (predictions are random).
    Test data has not relations between features and target."""
    train_data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=1)
    test_data = get_random_target_data(train_data)

    chain = generate_chain()
    chain.fit(input_data=train_data)
    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr <= CORRECT_MODEL_AUC_THR
    assert train_auc_thr >= CORRECT_MODEL_AUC_THR


def test_model_predictions_on_train_test_random():
    """Checks that model can't predict correctly on random train and test datasets and
    the roc_auc_scores is close to 0.5.
    Both train and test data have no relations between features and target."""
    data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=1)
    data = get_random_target_data(data)

    train_data, test_data = train_test_data_setup(data)

    chain = generate_chain()
    chain.fit(input_data=train_data)
    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr <= CORRECT_MODEL_AUC_THR
    assert train_auc_thr <= CORRECT_MODEL_AUC_THR
