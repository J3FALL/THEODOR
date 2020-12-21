import numpy as np
import pytest
from sklearn.datasets import load_iris

from fedot.core.data.data import InputData
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


def test_data_subset_correct(data_setup):
    subset_size = 50
    subset = data_setup.subset(0, subset_size - 1)

    assert len(subset.idx) == subset_size
    assert len(subset.features) == subset_size
    assert len(subset.target) == subset_size


def test_data_subset_incorrect(data_setup):
    subset_size = 105
    with pytest.raises(ValueError):
        assert data_setup.subset(0, subset_size)

    with pytest.raises(ValueError):
        assert data_setup.subset(-1, subset_size)
    with pytest.raises(ValueError):
        assert data_setup.subset(-1, -1)
