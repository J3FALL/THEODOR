import pytest

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_validation import (has_correct_model_positions, has_no_cycle, has_no_isolated_components,
                                                has_no_isolated_nodes, has_no_self_cycled_nodes, has_primary_nodes,
                                                validate)
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.repository.tasks import Task, TaskTypesEnum

ERROR_PREFIX = 'Invalid chain configuration:'


def valid_chain():
    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(model_type='logit',
                          nodes_from=[second])
    last = SecondaryNode(model_type='logit',
                         nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, last]:
        chain.add_node(node)

    return chain


def chain_with_cycle():
    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(model_type='logit',
                          nodes_from=[second, first])
    second.nodes_from.append(third)
    chain = Chain()
    for node in [first, second, third]:
        chain.add_node(node)

    return chain


def chain_with_isolated_nodes():
    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(model_type='logit',
                          nodes_from=[second])
    isolated = SecondaryNode(model_type='logit',
                             nodes_from=[])
    chain = Chain()

    for node in [first, second, third, isolated]:
        chain.add_node(node)

    return chain


def chain_with_multiple_roots():
    first = PrimaryNode(model_type='logit')
    root_first = SecondaryNode(model_type='logit',
                               nodes_from=[first])
    root_second = SecondaryNode(model_type='logit',
                                nodes_from=[first])
    chain = Chain()

    for node in [first, root_first, root_second]:
        chain.add_node(node)

    return chain


def chain_with_secondary_nodes_only():
    first = SecondaryNode(model_type='logit',
                          nodes_from=[])
    second = SecondaryNode(model_type='logit',
                           nodes_from=[first])
    chain = Chain()
    chain.add_node(first)
    chain.add_node(second)

    return chain


def chain_with_self_cycle():
    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='logit',
                           nodes_from=[first])
    second.nodes_from.append(second)

    chain = Chain()
    chain.add_node(first)
    chain.add_node(second)

    return chain


def chain_with_isolated_components():
    first = PrimaryNode(model_type='logit')
    second = SecondaryNode(model_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(model_type='logit',
                          nodes_from=[])
    fourth = SecondaryNode(model_type='logit',
                           nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, fourth]:
        chain.add_node(node)

    return chain


def chain_with_incorrect_root_model():
    first = PrimaryNode(model_type='logit')
    second = PrimaryNode(model_type='logit')
    final = SecondaryNode(model_type='direct_data_model',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain


def chain_with_incorrect_task_type():
    first = PrimaryNode(model_type='linear')
    second = PrimaryNode(model_type='linear')
    final = SecondaryNode(model_type='kmeans',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain, Task(TaskTypesEnum.classification)


def chain_with_incorrect_decomposition_structure():
    first = PrimaryNode(model_type='trend_data_model')
    second = PrimaryNode(model_type='residual_data_model')
    final = SecondaryNode(model_type='trend_data_model',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain


def chain_with_correct_decomposition_structure():
    first = PrimaryNode(model_type='trend_data_model')
    second = PrimaryNode(model_type='residual_data_model')
    final = SecondaryNode(model_type='linear',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain


def test_chain_with_cycle_raise_exception():
    chain = chain_with_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_cycle(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has cycles'


def test_chain_without_cycles_correct():
    chain = valid_chain()

    assert has_no_cycle(chain)


def test_chain_with_isolated_nodes_raise_exception():
    chain = chain_with_isolated_nodes()
    with pytest.raises(ValueError) as exc:
        assert has_no_isolated_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has isolated nodes'


def test_multi_root_chain_raise_exception():
    chain = chain_with_multiple_roots()

    with pytest.raises(Exception) as exc:
        assert chain.root_node
    assert str(exc.value) == f'{ERROR_PREFIX} More than 1 root_nodes in chain'


def test_chain_with_primary_nodes_correct():
    chain = valid_chain()
    assert has_primary_nodes(chain)


def test_chain_without_primary_nodes_raise_exception():
    chain = chain_with_secondary_nodes_only()
    with pytest.raises(Exception) as exc:
        assert has_primary_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain does not have primary nodes'


def test_chain_with_self_cycled_nodes_raise_exception():
    chain = chain_with_self_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_self_cycled_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has self-cycled nodes'


def test_chain_validate_correct():
    chain = valid_chain()
    validate(chain)


def test_chain_with_isolated_components_raise_exception():
    chain = chain_with_isolated_components()
    with pytest.raises(Exception) as exc:
        assert has_no_isolated_components(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has isolated components'


def test_chain_with_incorrect_root_model_raise_exception():
    chain = chain_with_incorrect_root_model()
    with pytest.raises(Exception) as exc:
        assert has_correct_model_positions(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has incorrect models positions'


def test_chain_with_incorrect_decomposition_raise_exception():
    chain = chain_with_incorrect_decomposition_structure()
    with pytest.raises(Exception) as exc:
        assert has_correct_model_positions(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has incorrect models positions'


def test_chain_with_incorrect_task_type_raise_exception():
    chain, task = chain_with_incorrect_task_type()
    with pytest.raises(Exception) as exc:
        assert has_correct_model_positions(chain, task)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has incorrect models positions'


def test_chain_with_correct_decomposition_raise_exception():
    chain = chain_with_correct_decomposition_structure()
    assert has_correct_model_positions(chain)
