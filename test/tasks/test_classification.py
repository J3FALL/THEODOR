import random

import numpy as np
from numpy import random as np_random
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.chain import Chain
from fedot.core.composer.node import PrimaryNode, SecondaryNode
from fedot.core.models.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.models.test_model import classification_dataset_with_redunant_features

random.seed(42)
np_random.seed(42)


def chain_simple() -> Chain:
    node_first = PrimaryNode('svc')
    node_second = PrimaryNode('lda')
    node_final = SecondaryNode('rf', nodes_from=[node_first, node_second])

    chain = Chain(node_final)

    return chain


def chain_with_pca() -> Chain:
    node_first = PrimaryNode('pca_data_model')
    node_second = PrimaryNode('lda')
    node_final = SecondaryNode('rf', nodes_from=[node_first, node_second])

    chain = Chain(node_final)

    return chain


def get_iris_data() -> InputData:
    synthetic_data = load_iris()
    input_data = InputData(idx=np.arange(0, len(synthetic_data.target)),
                           features=synthetic_data.data,
                           target=synthetic_data.target,
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def test_multiclassification_chain_fit_correct():
    data = get_iris_data()
    chain = chain_simple()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    chain.fit(input_data=train_data)
    results = chain.predict(input_data=test_data)

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')

    assert roc_auc_on_test > 0.95


def test_classification_with_pca_chain_fit_correct():
    data = classification_dataset_with_redunant_features()
    chain_pca = chain_with_pca()
    chain = chain_simple()

    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    chain.fit(input_data=train_data)
    chain_pca.fit(input_data=train_data)

    results = chain.predict(input_data=test_data)
    results_pca = chain_pca.predict(input_data=test_data)

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')

    roc_auc_on_test_pca = roc_auc(y_true=test_data.target,
                                  y_score=results_pca.predict,
                                  multi_class='ovo',
                                  average='macro')

    assert roc_auc_on_test_pca > roc_auc_on_test > 0.5
