from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris

from fedot.core.chains.chain import Chain, SharedChain
from fedot.core.chains.node import FittedModelCache, \
    PrimaryNode, SecondaryNode, SharedCache
from fedot.core.data.data import InputData, split_train_test
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def data_setup():
    task = Task(TaskTypesEnum.classification)
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
                           task=task, data_type=DataTypesEnum.table)
    test_data = InputData(features=test_data_x, target=test_data_y,
                          idx=np.arange(0, len(test_data_y)),
                          task=task, data_type=DataTypesEnum.table)
    return train_data, test_data


@pytest.fixture()
def iris_data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


def chain_first():
    #    XG
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('xgboost', 'xgboost', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            chain.add_node(new_node)
        chain.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    chain.add_node(root_of_tree)
    return chain


def chain_second():
    #    XG
    #  |     \
    # DT      KNN
    # |  \    |  \
    # KNN KNN LR  LDA
    chain = chain_first()
    new_node = SecondaryNode('dt')
    for model_type in ('knn', 'knn'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0], new_node)
    return chain


def chain_third():
    #    QDA
    #  |     \
    # RF     RF
    chain = Chain()
    new_node = SecondaryNode('qda')
    for model_type in ('rf', 'rf'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    chain.add_node(new_node)
    [chain.add_node(node_from) for node_from in new_node.nodes_from]
    return chain


def chain_fourth():
    #          XG
    #      |         \
    #     XG          KNN
    #   |    \        |  \
    # QDA     KNN     LR  LDA
    # |  \    |    \
    # RF  RF  KNN KNN
    chain = chain_first()
    new_node = SecondaryNode('qda')
    for model_type in ('rf', 'rf'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[1], new_node)
    new_node = SecondaryNode('knn')
    for model_type in ('knn', 'knn'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[0], new_node)
    return chain


def chain_fifth():
    #    KNN
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA KNN  KNN
    chain = chain_first()
    new_node = SecondaryNode('knn')
    chain.update_node(chain.root_node, new_node)
    new_node = PrimaryNode('knn')
    chain.update_node(chain.root_node.nodes_from[1].nodes_from[0], new_node)
    chain.update_node(chain.root_node.nodes_from[1].nodes_from[1], new_node)

    return chain


def test_cache_actuality_after_model_change(data_setup):
    """The non-affected nodes has actual cache after changing the model"""
    chain = chain_first()
    train, _ = data_setup
    chain.fit(input_data=train)
    new_node = SecondaryNode(model_type='logit')
    chain.update_node(old_node=chain.root_node.nodes_from[0],
                      new_node=new_node)

    root_parent_first = chain.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [chain.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in chain.nodes if node not in nodes_with_non_actual_cache]

    # non-affected nodes are actual
    assert all([node.cache.actual_cached_state for node in nodes_with_actual_cache])
    # affected nodes and their childs has no any actual cache
    assert not any([node.cache.actual_cached_state for node in nodes_with_non_actual_cache])


def test_cache_actuality_after_subtree_change_to_identical(data_setup):
    """The non-affected nodes has actual cache after changing the subtree to other pre-fitted subtree"""
    train, _ = data_setup
    chain = chain_first()
    other_chain = chain_second()
    chain.fit(input_data=train)
    other_chain.fit(input_data=train)
    chain.replace_node_with_parents(chain.root_node.nodes_from[0],
                                    other_chain.root_node.nodes_from[0])

    nodes_with_actual_cache = [node for node in chain.nodes if node not in [chain.root_node]]

    # non-affected nodes of initial chain and fitted nodes of new subtree are actual
    assert all([node.cache.actual_cached_state for node in nodes_with_actual_cache])
    # affected root node has no any actual cache
    assert not chain.root_node.cache.actual_cached_state


def test_cache_actuality_after_primary_node_changed_to_subtree(data_setup):
    """ The non-affected nodes has actual cache after changing the primary node to pre-fitted subtree"""
    train, _ = data_setup
    chain = chain_first()
    other_chain = chain_second()
    chain.fit(input_data=train)
    other_chain.fit(input_data=train)
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[0],
                                    other_chain.root_node.nodes_from[0])

    root_parent_first = chain.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [chain.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in chain.nodes if node not in nodes_with_non_actual_cache]

    # non-affected nodes of initial chain and fitted nodes of new subtree are actual
    assert all([node.cache.actual_cached_state for node in nodes_with_actual_cache])
    # affected root nodes and their childs has no any actual cache
    assert not any([node.cache.actual_cached_state for node in nodes_with_non_actual_cache])


def test_cache_historical_state_using(data_setup):
    train, _ = data_setup
    chain = chain_first()

    # chain fitted, model goes to cache
    chain.fit(input_data=train)
    new_node = SecondaryNode(model_type='logit')
    old_node = chain.root_node.nodes_from[0]

    # change child node to new one
    chain.update_node(old_node=old_node,
                      new_node=new_node)
    # cache is not actual
    assert not chain.root_node.cache.actual_cached_state
    # fit modified chain
    chain.fit(input_data=train)
    # cache is actual now
    assert chain.root_node.cache.actual_cached_state

    # change node back
    chain.update_node(old_node=chain.root_node.nodes_from[0],
                      new_node=old_node)
    # cache is actual without new fitting,
    # because the cached model was saved after first fit
    assert chain.root_node.cache.actual_cached_state


def test_multi_chain_caching_with_shared_cache(data_setup):
    train, _ = data_setup
    shared_cache = {}

    main_chain = SharedChain(base_chain=chain_second(), shared_cache=shared_cache)
    other_chain = SharedChain(base_chain=chain_first(), shared_cache=shared_cache)

    # fit other_chain that contains the parts identical to main_chain
    other_chain.fit(input_data=train)

    nodes_with_non_actual_cache = [main_chain.root_node, main_chain.root_node.nodes_from[0]] + \
                                  [_ for _ in main_chain.root_node.nodes_from[0].nodes_from]
    nodes_with_actual_cache = [node for node in main_chain.nodes if node not in nodes_with_non_actual_cache]

    # check that using of SharedChain make identical of the main_chain fitted,
    # despite the main_chain.fit() was not called
    assert all([node.cache.actual_cached_state for node in nodes_with_actual_cache])
    # the non-identical parts are still not fitted
    assert not any([node.cache.actual_cached_state for node in nodes_with_non_actual_cache])

    # check the same case with another chains
    shared_cache = {}

    main_chain = SharedChain(base_chain=chain_fourth(), shared_cache=shared_cache)

    prev_chain_first = SharedChain(base_chain=chain_third(), shared_cache=shared_cache)
    prev_chain_second = SharedChain(base_chain=chain_fifth(), shared_cache=shared_cache)

    prev_chain_first.fit(input_data=train)
    prev_chain_second.fit(input_data=train)

    nodes_with_non_actual_cache = [main_chain.root_node, main_chain.root_node.nodes_from[1]]
    nodes_with_actual_cache = [child for child in main_chain.root_node.nodes_from[0].nodes_from]

    assert not any([node.cache.actual_cached_state for node in nodes_with_non_actual_cache])
    assert all([node.cache.actual_cached_state for node in nodes_with_actual_cache])


def test_multi_chain_caching_with_import(data_setup):
    train, _ = data_setup

    main_chain = chain_second()
    other_chain = chain_first()

    # fit other_chain that contains the parts identical to main_chain
    other_chain.fit(input_data=train)
    main_chain.import_cache(other_chain)

    nodes_with_non_actual_cache = [main_chain.root_node, main_chain.root_node.nodes_from[0]]
    nodes_with_non_actual_cache += main_chain.root_node.nodes_from[0].nodes_from

    nodes_with_actual_cache = [node for node in main_chain.nodes if node not in nodes_with_non_actual_cache]

    # check that using of SharedChain make identical of the main_chain fitted,
    # despite the main_chain.fit() was not called
    assert all([node.cache.actual_cached_state for node in nodes_with_actual_cache])
    # the non-identical parts are still not fitted
    assert not any([node.cache.actual_cached_state for node in nodes_with_non_actual_cache])

    # check the same case with another chains
    main_chain = chain_fourth()

    prev_chain_first = chain_third()
    prev_chain_second = chain_fifth()

    prev_chain_first.fit(input_data=train)
    prev_chain_second.fit(input_data=train)

    main_chain.import_cache(prev_chain_first)
    main_chain.import_cache(prev_chain_second)

    nodes_with_non_actual_cache = [main_chain.root_node, main_chain.root_node.nodes_from[1]]
    nodes_with_actual_cache = main_chain.root_node.nodes_from[0].nodes_from

    assert not any([node.cache.actual_cached_state for node in nodes_with_non_actual_cache])
    assert all([node.cache.actual_cached_state for node in nodes_with_actual_cache])


def test_multi_chain_caching_local_cache(data_setup):
    train, _ = data_setup

    main_chain = chain_second()
    other_chain = chain_first()

    other_chain.fit(input_data=train)
    # shared cache is not used, so the main_chain is not fitted at all
    assert not any([node.cache.actual_cached_state for node in main_chain.nodes])

    main_chain = chain_fourth()

    prev_chain_first = chain_third()
    prev_chain_second = chain_fifth()

    prev_chain_first.fit(input_data=train)
    prev_chain_second.fit(input_data=train)

    assert not any([node.cache.actual_cached_state for node in main_chain.nodes])


def test_chain_sharing_and_unsharing(data_setup):
    chain = chain_first()
    assert all([isinstance(node.cache, FittedModelCache) for node in chain.nodes])
    chain = SharedChain(chain, {})

    assert all([isinstance(node.cache, SharedCache) for node in chain.nodes])
    chain = chain.unshare()
    assert all([isinstance(node.cache, FittedModelCache) for node in chain.nodes])
    assert isinstance(chain, Chain)


def test_shared_cache(data_setup):
    train, _ = data_setup

    shared_cache = {}
    main_chain = SharedChain(chain_first(), shared_cache)
    other_chain = SharedChain(chain_first(), shared_cache)
    other_chain.fit(train)

    # test cache is shared
    assert isinstance(main_chain.root_node.cache, SharedCache)
    # test cache is actual
    assert main_chain.root_node.cache.actual_cached_state is not None

    saved_model = main_chain.root_node.cache.actual_cached_state
    main_chain.root_node.cache.clear()
    # test cache is still actual despite the clearing of local cache
    assert main_chain.root_node.cache.actual_cached_state is not None

    shared_cache.clear()
    # test cache is not actual after clearing shared cache
    assert main_chain.root_node.cache.actual_cached_state is None

    main_chain.root_node.cache.append(saved_model)
    # test cache is actual after manual appending of model
    assert main_chain.root_node.cache.actual_cached_state is not None
    assert shared_cache[main_chain.root_node.descriptive_id] == saved_model


def test_cache_changed_data(iris_data_setup):
    data_first = iris_data_setup
    data_second = deepcopy(data_first)
    chain = chain_third()
    chain.fit(data_first)
    root_cache = chain.root_node.cache
    first_child_cache = chain.root_node.nodes_from[0].cache
    chain.fit(data_second, use_cache=True)
    new_root_cache = chain.root_node.cache
    new_first_child_cache = chain.root_node.nodes_from[0].cache
    chain.fit(data_second, use_cache=True)
    eq_new_root_cache = chain.root_node.cache
    eq_new_first_child_cache = chain.root_node.nodes_from[0].cache
    assert root_cache != new_root_cache and first_child_cache != new_first_child_cache
    assert new_root_cache == eq_new_root_cache and new_first_child_cache == eq_new_first_child_cache
