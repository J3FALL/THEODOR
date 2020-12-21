import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, split_train_test
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    (ClassificationMetricsEnum,
     ComplexityMetricsEnum,
     MetricsRepository,
     RegressionMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def data_setup():
    predictors, response = load_breast_cancer(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]
    train_data_x, test_data_x = split_train_test(predictors)
    train_data_y, test_data_y = split_train_test(response)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=np.arange(0, len(train_data_y)),
                           task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    test_data = InputData(features=test_data_x, target=test_data_y,
                          idx=np.arange(0, len(test_data_y)),
                          task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    return train_data, test_data


def default_valid_chain():
    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='logit', nodes_from=[first])
    third = SecondaryNode(model_type='logit', nodes_from=[first])
    final = SecondaryNode(model_type='logit', nodes_from=[second, third])

    chain = Chain(final)

    return chain


def test_structural_quality_correct():
    chain = default_valid_chain()
    metric_function = MetricsRepository().metric_by_id(ComplexityMetricsEnum.structural)
    expected_metric_value = 13
    actual_metric_value = metric_function(chain)
    assert actual_metric_value <= expected_metric_value


def test_classification_quality_metric(data_setup):
    train, _ = data_setup
    chain = default_valid_chain()
    chain.fit(input_data=train)

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
    metric_value = metric_function(chain=chain, reference_data=train)

    metric_function_with_penalty = \
        MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)
    metric_value_with_penalty = \
        metric_function_with_penalty(chain=chain, reference_data=train)

    assert 0.5 < abs(metric_value) < 1.0
    assert 0.5 < abs(metric_value_with_penalty) < 1.0
    assert metric_value < metric_value_with_penalty


def test_regression_quality_metric(data_setup):
    train, _ = data_setup
    chain = default_valid_chain()
    chain.fit(input_data=train)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
    metric_value = metric_function(chain=chain, reference_data=train)

    metric_function_with_penalty = \
        MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE_penalty)
    metric_value_with_penalty = \
        metric_function_with_penalty(chain=chain, reference_data=train)

    assert metric_value > 0
    assert metric_value_with_penalty > 0
    assert metric_value < metric_value_with_penalty
