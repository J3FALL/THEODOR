import numpy as np
from copy import deepcopy
from random import choice, randint
from typing import (Any, List, Tuple, Callable)

from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.multi_objective_fitness import MultiObjFitness


def node_height(chain: Any, node: Any) -> int:
    def recursive_child_height(parent_node: Any) -> int:
        node_child = chain.node_childs(parent_node)
        if node_child:
            height = recursive_child_height(node_child[0]) + 1
            return height
        else:
            return 0

    height = recursive_child_height(node)
    return height


def node_depth(node: Any) -> int:
    if not node.nodes_from:
        return 0
    else:
        return 1 + max([node_depth(next_node) for next_node in node.nodes_from])


def nodes_from_height(chain: Any, selected_height: int) -> List[Any]:
    def get_nodes(node: Any, current_height):
        nodes = []
        if current_height == selected_height:
            nodes.append(node)
        else:
            if node.nodes_from:
                for child in node.nodes_from:
                    nodes += get_nodes(child, current_height + 1)
        return nodes

    nodes = get_nodes(chain.root_node, current_height=0)
    return nodes


def random_chain(chain_generation_params, requirements, max_depth=None) -> Any:
    secondary_node_func = chain_generation_params.secondary_node_func
    primary_node_func = chain_generation_params.primary_node_func
    chain_class = chain_generation_params.chain_class
    max_depth = max_depth if max_depth else requirements.max_depth

    def chain_growth(chain: Any, node_parent: Any):
        offspring_size = randint(requirements.min_arity, requirements.max_arity)
        for offspring_node in range(offspring_size):
            height = node_height(chain, node_parent)
            is_max_depth_exceeded = height >= max_depth - 1
            is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
            if is_max_depth_exceeded or is_primary_node_selected:
                primary_node = primary_node_func(model_type=choice(requirements.primary))
                node_parent.nodes_from.append(primary_node)
                chain.add_node(primary_node)
            else:
                secondary_node = secondary_node_func(model_type=choice(requirements.secondary))
                chain.add_node(secondary_node)
                node_parent.nodes_from.append(secondary_node)
                chain_growth(chain, secondary_node)

    is_correct_chain = False
    while not is_correct_chain:
        chain = chain_class()
        chain_root = secondary_node_func(model_type=choice(requirements.secondary))
        chain.add_node(chain_root)
        chain_growth(chain, chain_root)
        is_correct_chain = constraint_function(chain)
    return chain


def equivalent_subtree(chain_first: Any, chain_second: Any) -> List[Tuple[Any, Any]]:
    """Finds the similar subtree in two given trees"""

    def structural_equivalent_nodes(node_first, node_second):
        nodes = []
        is_same_type = type(node_first) == type(node_second)
        node_first_childs = node_first.nodes_from
        node_second_childs = node_second.nodes_from
        if is_same_type and ((not node_first.nodes_from) or len(node_first_childs) == len(node_second_childs)):
            nodes.append((node_first, node_second))
            if node_first.nodes_from:
                for node1_child, node2_child in zip(node_first.nodes_from, node_second.nodes_from):
                    nodes_set = structural_equivalent_nodes(node1_child, node2_child)
                    if nodes_set:
                        nodes += nodes_set
        return nodes

    pairs_set = structural_equivalent_nodes(chain_first.root_node, chain_second.root_node)
    assert isinstance(pairs_set, list)
    return pairs_set


def replace_subtrees(chain_first: Any, chain_second: Any, node_from_first: Any, node_from_second: Any,
                     layer_in_first: int, layer_in_second: int, max_depth: int):
    node_from_chain_first_copy = deepcopy(node_from_first)

    summary_depth = layer_in_first + node_depth(node_from_second)
    if summary_depth <= max_depth and summary_depth != 0:
        chain_first.replace_node_with_parents(node_from_first, node_from_second)

    summary_depth = layer_in_second + node_depth(node_from_first)
    if summary_depth <= max_depth and summary_depth != 0:
        chain_second.replace_node_with_parents(node_from_second, node_from_chain_first_copy)


def num_of_parents_in_crossover(num_of_final_inds: int) -> int:
    return num_of_final_inds if not num_of_final_inds % 2 else num_of_final_inds + 1


def evaluate_individuals(individuals_set, objective_function, is_multi_objective: bool):
    for ind in reversed(individuals_set):
        ind.fitness = calculate_objective(ind, objective_function, is_multi_objective)
        if ind.fitness is None:
            individuals_set.remove(ind)
    if len(individuals_set) == 0:
        raise AttributeError('List became empty after incorrect individuals removing.'
                             'It can occur because of too short model fitting time constraint')


def calculate_objective(ind: Any, objective_function: Callable, is_multi_objective: bool) -> Any:
    calculated_fitness = objective_function(ind)
    if calculated_fitness is None:
        return None
    else:
        if is_multi_objective:
            fitness = MultiObjFitness(values=calculated_fitness,
                                      weights=tuple([-1 for _ in range(len(calculated_fitness))]))
        else:
            fitness = calculated_fitness[0]
    return fitness


def is_equal_fitness(first_fitness, second_fitness, atol=1e-10, rtol=1e-10):
    return np.isclose(first_fitness, second_fitness, atol=atol, rtol=rtol)


def is_equal_archive(old_archive: Any, new_archive: Any) -> bool:
    if len(old_archive.items) != len(new_archive.items):
        fronts_coincidence = False
    else:
        are_inds_found = []
        for ind in new_archive:
            eq_inds = list(filter(lambda item: all(
                [is_equal_fitness(obj, ind.fitness.values[obj_num]) for obj_num, obj in
                 enumerate(item.fitness.values)]), old_archive.items))
            are_inds_found.append(len(eq_inds) > 0)
        fronts_coincidence = all(are_inds_found)

    return fronts_coincidence


def duplicates_filtration(archive, population):
    return list(filter(lambda x: not any([x.fitness == pop_ind.fitness for pop_ind in population]), archive.items))
