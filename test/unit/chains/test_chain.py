import os
from copy import deepcopy
from random import seed

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import probs_to_labels

seed(1)
np.random.seed(1)


@pytest.fixture()
def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


@pytest.mark.parametrize('data_fixture', ['data_setup', 'file_data_setup'])
def test_nodes_sequence_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train, _ = train_test_data_setup(data)

    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='lda', nodes_from=[first])
    third = SecondaryNode(model_type='qda', nodes_from=[first])
    final = SecondaryNode(model_type='knn', nodes_from=[second, third])

    train_predicted = final.fit(input_data=train)

    assert final.descriptive_id == (
        '((/n_logit_default_params;)/'
        'n_lda_default_params;;(/'
        'n_logit_default_params;)/'
        'n_qda_default_params;)/'
        'n_knn_default_params')

    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.cache.actual_cached_state is not None


def test_chain_hierarchy_fit_correct(data_setup):
    data = data_setup
    train, _ = train_test_data_setup(data)

    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='logit', nodes_from=[first])
    third = SecondaryNode(model_type='logit', nodes_from=[first])
    final = SecondaryNode(model_type='logit', nodes_from=[second, third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    train_predicted = chain.fit(input_data=train, use_cache=False)

    assert chain.root_node.descriptive_id == (
        '((/n_logit_default_params;)/'
        'n_logit_default_params;;(/'
        'n_logit_default_params;)/'
        'n_logit_default_params;)/'
        'n_logit_default_params')

    assert chain.length == 4
    assert chain.depth == 3
    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.cache.actual_cached_state is not None


def test_chain_sequential_fit_correct(data_setup):
    data = data_setup
    train, _ = train_test_data_setup(data)

    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='logit', nodes_from=[first])
    third = SecondaryNode(model_type='logit', nodes_from=[second])
    final = SecondaryNode(model_type='logit', nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    train_predicted = chain.fit(input_data=train, use_cache=False)

    assert chain.root_node.descriptive_id == (
        '(((/n_logit_default_params;)/'
        'n_logit_default_params;)/'
        'n_logit_default_params;)/'
        'n_logit_default_params')

    assert chain.length == 4
    assert chain.depth == 4
    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.cache.actual_cached_state is not None


def test_chain_with_datamodel_fit_correct(data_setup):
    data = data_setup
    train_data, test_data = train_test_data_setup(data)

    chain = Chain()

    node_data = PrimaryNode('direct_data_model')
    node_first = PrimaryNode('bernb')
    node_second = SecondaryNode('rf')

    node_second.nodes_from = [node_first, node_data]

    chain.add_node(node_data)
    chain.add_node(node_first)
    chain.add_node(node_second)

    chain.fit(train_data)
    results = np.asarray(probs_to_labels(chain.predict(test_data).predict))

    assert results.shape == test_data.target.shape


def test_secondary_nodes_is_invariant_to_inputs_order(data_setup):
    data = data_setup
    train, test = train_test_data_setup(data)

    first = PrimaryNode(model_type='logit')
    second = PrimaryNode(model_type='lda')
    third = PrimaryNode(model_type='knn')
    final = SecondaryNode(model_type='xgboost',
                          nodes_from=[first, second, third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    first = deepcopy(first)
    second = deepcopy(second)
    third = deepcopy(third)

    final_shuffled = SecondaryNode(model_type='xgboost',
                                   nodes_from=[third, first, second])

    chain_shuffled = Chain()
    # change order of nodes in list
    for node in [final_shuffled, third, first, second]:
        chain_shuffled.add_node(node)

    train_predicted = chain.fit(input_data=train)

    train_predicted_shuffled = chain_shuffled.fit(input_data=train)

    # train results should be invariant
    assert chain.root_node.descriptive_id == chain_shuffled.root_node.descriptive_id
    assert np.equal(train_predicted.predict, train_predicted_shuffled.predict).all()

    test_predicted = chain.predict(input_data=test)
    test_predicted_shuffled = chain_shuffled.predict(input_data=test)

    # predict results should be invariant
    assert np.equal(test_predicted.predict, test_predicted_shuffled.predict).all()

    # change parents order for the nodes fitted chain
    nodes_for_change = chain.nodes[3].nodes_from
    chain.nodes[3].nodes_from = [nodes_for_change[2], nodes_for_change[0], nodes_for_change[1]]
    chain.nodes[3].cache.clear()
    chain.fit(train)
    test_predicted_re_shuffled = chain.predict(input_data=test)

    # predict results should be invariant
    assert np.equal(test_predicted.predict, test_predicted_re_shuffled.predict).all()


def test_chain_with_custom_params_for_model(data_setup):
    data = data_setup
    custom_params = dict(n_neighbors=1,
                         weights='uniform',
                         p=1)

    first = PrimaryNode(model_type='logit')
    second = PrimaryNode(model_type='lda')
    final = SecondaryNode(model_type='knn', nodes_from=[first, second])

    chain = Chain()
    chain.add_node(final)
    chain_default_params = deepcopy(chain)

    chain.root_node.custom_params = custom_params

    chain_default_params.fit(data)
    chain.fit(data)

    custom_params_prediction = chain.predict(data).predict
    default_params_prediction = chain_default_params.predict(data).predict

    assert not np.array_equal(custom_params_prediction, default_params_prediction)


def test_chain_with_wrong_data():
    chain = Chain(PrimaryNode('linear'))
    data_seq = np.arange(0, 10)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=10,
                                    max_window_size=len(data_seq) + 1,
                                    return_all_steps=False))

    data = InputData(idx=data_seq, features=data_seq, target=data_seq,
                     data_type=DataTypesEnum.ts, task=task)

    with pytest.raises(ValueError):
        chain.fit(data)


def test_chain_str():
    # given
    first = PrimaryNode(model_type='logit')
    second = PrimaryNode(model_type='lda')
    third = PrimaryNode(model_type='knn')
    final = SecondaryNode(model_type='xgboost',
                          nodes_from=[first, second, third])
    chain = Chain()
    chain.add_node(final)

    expected_chain_description = "{'depth': 2, 'length': 4, 'nodes': [xgboost, logit, lda, knn]}"

    # when
    actual_chain_description = str(chain)

    # then
    assert actual_chain_description == expected_chain_description


def test_cahin_repr():
    first = PrimaryNode(model_type='logit')
    second = PrimaryNode(model_type='lda')
    third = PrimaryNode(model_type='knn')
    final = SecondaryNode(model_type='xgboost',
                          nodes_from=[first, second, third])
    chain = Chain()
    chain.add_node(final)

    expected_chain_description = "{'depth': 2, 'length': 4, 'nodes': [xgboost, logit, lda, knn]}"

    assert repr(chain) == expected_chain_description
