import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from fedot.core.chains.node import PrimaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.models.model import Model
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def data_setup() -> InputData:
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


def model_metrics_info(class_name, y_true, y_pred):
    print('\n', f'#test_eval_strategy_{class_name}')
    print(classification_report(y_true, y_pred))
    print('Test model accuracy: ', accuracy_score(y_true, y_pred))


def test_node_factory_log_reg_correct(data_setup):
    model_type = 'logit'
    node = PrimaryNode(model_type=model_type)

    expected_model = Model(model_type=model_type).__class__
    actual_model = node.model.__class__

    assert node.__class__ == PrimaryNode
    assert expected_model == actual_model


def test_eval_strategy_logreg(data_setup):
    data_set = data_setup
    train, test = train_test_data_setup(data=data_set)
    test_skl_model = LogisticRegression(C=10., random_state=1,
                                        solver='liblinear',
                                        max_iter=10000, verbose=0)
    test_skl_model.fit(train.features, train.target)
    expected_result = test_skl_model.predict(test.features)

    test_model_node = PrimaryNode(model_type='logit')
    test_model_node.fit(input_data=train)
    actual_result = test_model_node.predict(input_data=test)

    assert len(actual_result.predict) == len(expected_result)


def test_node_str():
    # given
    model_type = 'logit'
    test_model_node = PrimaryNode(model_type=model_type)
    expected_node_description = model_type

    # when
    actual_node_description = str(test_model_node)

    # then
    assert actual_node_description == expected_node_description


def test_node_repr():
    # given
    model_type = 'logit'
    test_model_node = PrimaryNode(model_type=model_type)
    expected_node_description = model_type

    # when
    actual_node_description = repr(test_model_node)

    # then
    assert actual_node_description == expected_node_description
