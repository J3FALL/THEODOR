from functools import partial

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import ChainGenerationParams, GPComposerRequirements
from fedot.core.composer.optimisers.gp_operators import random_chain
from fedot.core.composer.optimisers.selection import (
    SelectionTypesEnum,
    individuals_selection,
    random_selection,
    selection,
    tournament_selection
)
from fedot.core.debug.metrics import RandomMetric


def rand_population_gener_and_eval(pop_size=4):
    models_set = ['knn', 'logit', 'rf']
    requirements = GPComposerRequirements(primary=models_set,
                                          secondary=models_set, max_depth=1)
    secondary_node_func = SecondaryNode
    primary_node_func = PrimaryNode
    chain_gener_params = ChainGenerationParams(chain_class=Chain,
                                               secondary_node_func=secondary_node_func,
                                               primary_node_func=primary_node_func)
    random_chain_function = partial(random_chain, chain_generation_params=chain_gener_params,
                                    requirements=requirements)
    population = [random_chain_function() for _ in range(pop_size)]
    # evaluation
    for ind in population:
        ind.fitness = obj_function(ind)
    return population


def obj_function(chain: Chain) -> float:
    metric_function = RandomMetric.get_value
    return metric_function()


def test_tournament_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    selected_individuals = tournament_selection(individuals=population,
                                                pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_random_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    selected_individuals = random_selection(individuals=population,
                                            pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    selected_individuals = selection(types=[SelectionTypesEnum.tournament],
                                     population=population,
                                     pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_individuals_selection_random_individuals():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    types = [SelectionTypesEnum.tournament]
    selected_individuals = individuals_selection(types=types,
                                                 individuals=population,
                                                 pop_size=num_of_inds)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(set(selected_individuals_ref)) == len(selected_individuals) and
            len(selected_individuals) == num_of_inds)


def test_individuals_selection_equality_individuals():
    num_of_inds = 4
    population = rand_population_gener_and_eval(pop_size=1)
    types = [SelectionTypesEnum.tournament]
    population = [population[0] for _ in range(4)]
    selected_individuals = individuals_selection(types=types,
                                                 individuals=population,
                                                 pop_size=num_of_inds)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(selected_individuals) == num_of_inds and
            len(set(selected_individuals_ref)) == 1)
