import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.models.test_model import classification_dataset_with_redunant_features


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


def get_binary_classification_data():
    test_file_path = str(os.path.dirname(__file__))
    file = '../data/simple_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
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


def test_output_mode_labels():
    data = get_iris_data()
    chain = chain_simple()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    chain.fit(input_data=train_data)
    results = chain.predict(input_data=test_data, output_mode='labels')
    results_probs = chain.predict(input_data=test_data)

    assert len(results.predict) == len(test_data.target)
    assert set(results.predict) == {0, 1, 2}

    assert not np.array_equal(results_probs.predict, results.predict)


def test_output_mode_full_probs():
    data = get_binary_classification_data()
    chain = chain_simple()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    chain.fit(input_data=train_data)
    results = chain.predict(input_data=test_data, output_mode='full_probs')
    results_default = chain.predict(input_data=test_data)
    results_probs = chain.predict(input_data=test_data, output_mode='probs')

    assert not np.array_equal(results_probs.predict, results.predict)
    assert np.array_equal(results_probs.predict, results_default.predict)
    assert results.predict.shape == (len(test_data.target), 2)
    assert results_probs.predict.shape == (len(test_data.target),)
