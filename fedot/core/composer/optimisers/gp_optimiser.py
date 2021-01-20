import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (Any, Callable, List, Optional, Tuple)

import numpy as np

from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.crossover import CrossoverTypesEnum, crossover
from fedot.core.composer.optimisers.gp_operators import random_chain, num_of_parents_in_crossover, \
    evaluate_individuals, calculate_objective
from fedot.core.composer.optimisers.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.composer.optimisers.mutation import MutationTypesEnum, mutation
from fedot.core.composer.optimisers.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.composer.optimisers.selection import SelectionTypesEnum, selection
from fedot.core.composer.timer import CompositionTimer
from fedot.core.log import default_log, Log


@dataclass
class OptimiserHistory:
    individuals: List[Any] = None
    archive_history: List[Any] = None


class GPChainOptimiserParameters:
    """
        This class is for defining the parameters of optimiser

        :param selection_types: List of selection operators types
        :param crossover_types: List of crossover operators types
        :param mutation_types: List of mutation operators types
        :param regularization_type: type of regularization operator
        :param genetic_scheme_type: type of genetic evolutionary scheme
        :param with_auto_depth_configuration: flag to enable option of automated tree depth configuration during
        evolution. Default False.
        :param depth_increase_step: the step of depth increase in automated depth configuration
        :param multi_objective: flag used for of algorithm type definition (muti-objective if true or  single-objective
        if false). Value is defined in GPComposerBuilder. Default False.
    """

    def __init__(self, selection_types: List[SelectionTypesEnum] = None,
                 crossover_types: List[CrossoverTypesEnum] = None,
                 mutation_types: List[MutationTypesEnum] = None,
                 regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.none,
                 genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.generational,
                 with_auto_depth_configuration: bool = False, depth_increase_step: int = 3,
                 multi_objective: bool = False):

        self.selection_types = selection_types
        self.crossover_types = crossover_types
        self.mutation_types = mutation_types
        self.regularization_type = regularization_type
        self.genetic_scheme_type = genetic_scheme_type
        self.with_auto_depth_configuration = with_auto_depth_configuration
        self.depth_increase_step = depth_increase_step
        self.multi_objective = multi_objective
        self.set_default_params()

    def set_default_params(self):
        if not self.selection_types:
            self.selection_types = [SelectionTypesEnum.tournament]
        if not self.crossover_types:
            self.crossover_types = [CrossoverTypesEnum.subtree]
        if not self.mutation_types:
            self.mutation_types = [MutationTypesEnum.simple]


class GPChainOptimiser:
    """
    Base class of evolutionary chain optimiser

    :param initial_chain: chain which was initialized outside the optimiser
    :param requirements: composer requirements
    :param chain_generation_params: parameters for new chain generation
    :param parameters: parameters of chain optimiser
    :param log: optional parameter for log oject
    :param archive_type: type of archive with best individuals
    """

    def __init__(self, initial_chain, requirements, chain_generation_params,
                 parameters: Optional[GPChainOptimiserParameters] = None, log: Log = None, archive_type=None):
        self.chain_generation_params = chain_generation_params
        self.primary_node_func = self.chain_generation_params.primary_node_func
        self.secondary_node_func = self.chain_generation_params.secondary_node_func
        self.chain_class = self.chain_generation_params.chain_class
        self.requirements = requirements
        self.history = OptimiserHistory(individuals=[], archive_history=[])
        self.archive = archive_type
        self.parameters = GPChainOptimiserParameters() if parameters is None else parameters
        self.max_depth = self.requirements.start_depth \
            if self.parameters.with_auto_depth_configuration and self.requirements.start_depth \
            else self.requirements.max_depth

        self.generation_num = 0
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        generation_depth = self.max_depth if self.requirements.start_depth is None else self.requirements.start_depth

        self.chain_generation_function = partial(random_chain, chain_generation_params=self.chain_generation_params,
                                                 requirements=self.requirements, max_depth=generation_depth)

        necessary_attrs = ['add_node', 'root_node', 'replace_node_with_parents', 'update_node', 'node_childs']
        if not all([hasattr(self.chain_class, attr) for attr in necessary_attrs]):
            ex = f'Object chain_class has no required attributes for gp_optimizer'
            self.log.error(ex)
            raise AttributeError(ex)

        if not self.requirements.pop_size:
            self.requirements.pop_size = 10

        if initial_chain and type(initial_chain) != list:
            self.population = [deepcopy(initial_chain) for _ in range(self.requirements.pop_size)]
        else:
            self.population = initial_chain

    def optimise(self, objective_function, offspring_rate: float = 0.5):
        if self.population is None:
            self.population = self._make_population(self.requirements.pop_size)

        num_of_new_individuals = self.offspring_size(offspring_rate)

        with CompositionTimer(log=self.log) as t:

            if self.requirements.add_single_model_chains:
                self.best_single_model, self.requirements.primary = \
                    self._best_single_models(objective_function)

            evaluate_individuals(self.population, objective_function, self.parameters.multi_objective)

            self._add_to_history(self.population)

            self.log_info_about_best()

            for self.generation_num in range(self.requirements.num_of_generations - 1):
                self.log.info(f'Generation num: {self.generation_num}')

                if self.archive is not None:
                    self.archive.update(self.population)
                    self.history.archive_history.append(deepcopy(self.archive))

                self.num_of_gens_without_improvements = self.update_stagnation_counter()
                self.log.info(
                    f'max_depth: {self.max_depth}, no improvements: {self.num_of_gens_without_improvements}')

                if self.parameters.with_auto_depth_configuration and self.generation_num != 0:
                    self.max_depth_recount()

                individuals_to_select = regularized_population(reg_type=self.parameters.regularization_type,
                                                               population=self.population,
                                                               objective_function=objective_function,
                                                               chain_class=self.chain_class)

                num_of_parents = num_of_parents_in_crossover(num_of_new_individuals)

                selected_individuals = selection(types=self.parameters.selection_types,
                                                 population=individuals_to_select,
                                                 pop_size=num_of_parents)

                new_population = []

                for parent_num in range(0, len(selected_individuals), 2):
                    new_population += self.reproduce(selected_individuals[parent_num],
                                                     selected_individuals[parent_num + 1])

                evaluate_individuals(new_population, objective_function, self.parameters.multi_objective)

                self.prev_best = deepcopy(self.best_individual)

                self.population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                              self.population,
                                              new_population, self.num_of_inds_in_next_pop)

                if not self.parameters.multi_objective and self.with_elitism:
                    self.population.append(self.prev_best)

                self._add_to_history(self.population)
                self.log.info(f'spent time: {round(t.minutes_from_start, 1)} min')
                self.log_info_about_best()

                if t.is_time_limit_reached(self.requirements.max_lead_time, self.generation_num):
                    break

            if self.archive is not None:
                self.archive.update(self.population)
                self.history.archive_history.append(deepcopy(self.archive))

            best = self.result_individual()
            self.log.info("Result:")
            self.log_info_about_best()

        return best, self.history

    @property
    def best_individual(self) -> Any:
        if self.parameters.multi_objective:
            return self.archive
        else:
            return self.get_best_individual(self.population)

    @property
    def with_elitism(self) -> bool:
        if self.parameters.multi_objective:
            return False
        else:
            return self.requirements.pop_size > 1

    @property
    def num_of_inds_in_next_pop(self):
        return self.requirements.pop_size - 1 if self.with_elitism and not self.parameters.multi_objective \
            else self.requirements.pop_size

    def update_stagnation_counter(self) -> int:
        value = 0
        if self.generation_num != 0:
            if self.parameters.multi_objective:
                equal_best = self.is_equal_archive(self.prev_best, self.archive)
            else:
                equal_best = self.is_equal_fitness(self.prev_best.fitness, self.best_individual.fitness)
            if equal_best:
                value = self.num_of_gens_without_improvements + 1

        return value

    def log_info_about_best(self):
        if self.parameters.multi_objective:
            self.log.info(f'Pareto Front: {[item.fitness.values for item in self.archive.items]}')
        else:
            self.log.info(f'Best metric is {self.best_individual.fitness}')

    def max_depth_recount(self):
        if self.num_of_gens_without_improvements == self.parameters.depth_increase_step and \
                self.max_depth + 1 <= self.requirements.max_depth:
            self.max_depth += 1

    def get_best_individual(self, individuals: List[Any], equivalents_from_current_pop=True) -> Any:
        best_ind = min(individuals, key=lambda ind: ind.fitness)
        if equivalents_from_current_pop:
            equivalents = self.simpler_equivalents_of_best_ind(best_ind)
        else:
            equivalents = self.simpler_equivalents_of_best_ind(best_ind, individuals)

        if equivalents:
            best_candidate_id = min(equivalents, key=equivalents.get)
            best_ind = individuals[best_candidate_id]
        return best_ind

    def simpler_equivalents_of_best_ind(self, best_ind: Any, inds: List[Any] = None) -> dict:
        individuals = self.population if inds is None else inds

        sort_inds = np.argsort([ind.fitness for ind in individuals])[1:]
        simpler_equivalents = {}
        for i in sort_inds:
            is_fitness_equals_to_best = self.is_equal_fitness(best_ind.fitness, individuals[i].fitness)
            has_less_num_of_models_than_best = len(individuals[i].nodes) < len(best_ind.nodes)
            if is_fitness_equals_to_best and has_less_num_of_models_than_best:
                simpler_equivalents[i] = len(individuals[i].nodes)
        return simpler_equivalents

    def reproduce(self, selected_individual_first, selected_individual_second=None) -> Tuple[Any]:
        if selected_individual_second:
            new_inds = crossover(self.parameters.crossover_types,
                                 selected_individual_first,
                                 selected_individual_second,
                                 crossover_prob=self.requirements.crossover_prob,
                                 max_depth=self.max_depth)
        else:
            new_inds = [selected_individual_first]

        new_inds = tuple([mutation(types=self.parameters.mutation_types,
                                   chain_generation_params=self.chain_generation_params,
                                   chain=new_ind, requirements=self.requirements,
                                   max_depth=self.max_depth) for new_ind in new_inds])

        return new_inds

    def _make_population(self, pop_size: int) -> List[Any]:
        model_chains = []
        while len(model_chains) < pop_size:
            chain = self.chain_generation_function()
            if constraint_function(chain):
                model_chains.append(chain)
        return model_chains

    def _add_to_history(self, individuals: List[Any]):
        self.history.individuals.append(individuals)

    def _best_single_models(self, objective_function: Callable, num_best: int = 7):
        single_models_inds = []
        for model in self.requirements.primary:
            single_models_ind = self.chain_class([self.primary_node_func(model)])
            single_models_ind.fitness = calculate_objective(single_models_ind, objective_function,
                                                            self.parameters.multi_objective)
            if single_models_ind.fitness is not None:
                single_models_inds.append(single_models_ind)

        best_inds = sorted(single_models_inds, key=lambda ind: ind.fitness)
        return best_inds[0], [i.nodes[0].model.model_type for i in best_inds][:num_best]

    def offspring_size(self, offspring_rate: float = None):
        default_offspring_rate = 0.5 if not offspring_rate else offspring_rate
        if self.parameters.genetic_scheme_type == GeneticSchemeTypesEnum.steady_state:
            num_of_new_individuals = math.ceil(self.requirements.pop_size * default_offspring_rate)
        else:
            num_of_new_individuals = self.requirements.pop_size - 1
        return num_of_new_individuals

    def is_equal_fitness(self, first_fitness, second_fitness, atol=1e-10, rtol=1e-10):
        return np.isclose(first_fitness, second_fitness, atol=atol, rtol=rtol)

    def is_equal_archive(self, old_archive: Any, new_archive: Any) -> bool:
        if len(old_archive.items) != len(new_archive.items):
            fronts_coincidence = False
        else:
            are_inds_found = []
            for ind in new_archive:
                eq_inds = list(filter(lambda item: all(
                    [self.is_equal_fitness(obj, ind.fitness.values[obj_num]) for obj_num, obj in
                     enumerate(item.fitness.values)]), old_archive.items))
                are_inds_found.append(len(eq_inds) > 0)
            fronts_coincidence = all(are_inds_found)

        return fronts_coincidence

    def result_individual(self) -> Any:
        if not self.parameters.multi_objective:
            best = self.best_individual

            if self.requirements.add_single_model_chains and \
                    (self.best_single_model.fitness <= best.fitness):
                best = self.best_single_model
        else:
            best = self.archive
        return best
