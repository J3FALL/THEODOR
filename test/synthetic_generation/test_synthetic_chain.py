import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.chain import \
    (chain_balanced_tree,
     chain_full_random,
     chain_with_random_links)


# TODO: get rid of duplicated code
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
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification))

    return data


def test_chain_with_random_links_correct(classification_dataset):
    depth = 3
    models_per_level = [3, 2, 1]
    used_models = ['logit']
    chain = chain_with_random_links(depth=depth, models_per_level=models_per_level,
                                    used_models=used_models)

    assert chain.depth == depth
    assert chain.length == sum(models_per_level)

    chain.fit_from_scratch(input_data=classification_dataset)


def test_chain_full_random_correct(classification_dataset):
    depth = 3
    max_lvl_size = 4
    used_models = ['logit']
    chain = chain_full_random(depth=depth, max_level_size=max_lvl_size,
                              used_models=used_models)

    assert chain.depth == depth

    chain.fit_from_scratch(input_data=classification_dataset)


def test_chain_balanced_tree_correct(classification_dataset):
    depth = 3
    models_per_level = [4, 2, 1]
    used_models = ['logit']
    chain = chain_balanced_tree(depth=depth, models_per_level=models_per_level,
                                used_models=used_models)

    assert chain.depth == depth
    assert chain.length == sum(models_per_level)

    chain.fit_from_scratch(input_data=classification_dataset)
