import itertools
from dataclasses import dataclass
from deap import tools
from functools import partial
from typing import (
    Callable,
    Optional,
    Tuple,
    Any,
    Union,
    List
)
from fedot.core.chains.chain import Chain, SharedChain
from fedot.core.chains.chain_validation import validate
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.composer import Composer, ComposerRequirements
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiser, GPChainOptimiserParameters
from fedot.core.composer.optimisers.inheritance import GeneticSchemeTypesEnum
from fedot.core.composer.optimisers.mutation import MutationStrengthEnum
from fedot.core.composer.optimisers.param_free_gp_optimiser import GPChainParameterFreeOptimiser
from fedot.core.composer.optimisers.regularization import RegularizationTypesEnum
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.composer.write_history import write_composer_history_to_csv
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository, \
    RegressionMetricsEnum, MetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

sample_split_ration_for_tasks = {
    TaskTypesEnum.classification: 0.8,
    TaskTypesEnum.regression: 0.8,
    TaskTypesEnum.ts_forecasting: 0.5
}


@dataclass
class GPComposerRequirements(ComposerRequirements):
    """
    Dataclass is for defining the requirements for composition process of genetic programming composer

    :param pop_size: population size
    :param num_of_generations: maximal number of evolutionary algorithm generations
    :param crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :param mutation_prob: mutation probability
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)
    :param model_fit_time_constraint: time constraint for model fitting (seconds)
    :param start_depth: start value of tree depth
    """
    pop_size: Optional[int] = 20
    num_of_generations: Optional[int] = 100
    crossover_prob: Optional[float] = 0.8
    mutation_prob: Optional[float] = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean
    model_fit_time_constraint: Optional[int] = None
    start_depth: int = None


@dataclass
class ChainGenerationParams:
    """
    This dataclass is for defining the parameters using in chain generation process

    :param primary_node_func: the function for primary node generation
    :param secondary_node_func: the function for secondary node generation
    :param chain_class: class for the chain object
    """
    primary_node_func: Callable = PrimaryNode
    secondary_node_func: Callable = SecondaryNode
    chain_class: Callable = Chain


class GPComposer(Composer):
    """
    Genetic programming based composer
    :param optimiser: optimiser generated in GPComposerBuilder
    :param metrics: metrics used to define the quality of found solution
    :param composer_requirements: requirements for composition process
    :param initial_chain: defines the initial state of the population. If None then initial population is random.
    """

    def __init__(self, optimiser=None,
                 composer_requirements: Optional[GPComposerRequirements] = None,
                 metrics: Optional[Callable] = None,
                 initial_chain: Optional[Chain] = None):

        super().__init__(metrics=metrics, composer_requirements=composer_requirements, initial_chain=initial_chain)
        self.shared_cache = {}
        self.optimiser = optimiser

    def compose_chain(self, data: InputData, is_visualise: bool = False, is_tune: bool = False) \
            -> Union[Chain, List[Chain]]:

        if not self.optimiser:
            raise AttributeError(f'Optimiser for chain composition is not defined')

        train_data, test_data = train_test_data_setup(data,
                                                      sample_split_ration_for_tasks[data.task.task_type],
                                                      task=data.task)
        self.shared_cache.clear()
        metric_function_for_nodes = partial(self.composer_metric, self.metrics, train_data, test_data, True)

        best_chain, self.history = self.optimiser.optimise(metric_function_for_nodes)

        self.log.info('GP composition finished')

        history_inds = self.history.individuals
        historical_chains = list(itertools.chain(*history_inds))

        if is_visualise:
            if self.optimiser.parameters.multi_objective:
                ComposerVisualiser.pareto_gif_create(self.history.archive_history, self.history.individuals)
                historical_fitness = [[chain.fitness.values[0] for chain in pop] for pop in history_inds]
                ComposerVisualiser.boxplots_gif_create(self.history.individuals)
            else:
                historical_fitness = [[chain.fitness for chain in pop] for pop in history_inds]
            all_historical_fitness = list(itertools.chain(*historical_fitness))
            ComposerVisualiser.visualise_history(historical_chains, all_historical_fitness)

        write_composer_history_to_csv(historical_chains=history_inds)

        if is_tune:
            self.tune_chain(best_chain, data, self.composer_requirements.max_lead_time)
        return best_chain

    def composer_metric(self, metrics, train_data: InputData, test_data: InputData,
                        is_chain_shared: bool, chain: Chain) -> Optional[Tuple[Any]]:
        try:
            validate(chain)
            if type(metrics) is not list:
                metrics = [metrics]

            if is_chain_shared:
                evaluating_chain = SharedChain(base_chain=chain, shared_cache=self.shared_cache)
            else:
                evaluating_chain = chain
            evaluating_chain.fit(input_data=train_data,
                                 model_fit_time_constraint=self.composer_requirements.model_fit_time_constraint)
            if is_chain_shared:
                chain.computation_time = evaluating_chain.computation_time

            evaluated_metrics = []
            for metric in metrics:
                metric_func = MetricsRepository().metric_by_id(metric)
                evaluated_metrics.append(metric_func(chain, reference_data=test_data))

            evaluated_metrics = tuple(evaluated_metrics)

        except Exception as ex:
            self.log.info(f'Chain assessment warning: {ex}. Continue.')
            evaluated_metrics = None
        return evaluated_metrics

    @staticmethod
    def tune_chain(chain: Chain, data: InputData, time_limit):
        chain.fine_tune_all_nodes(input_data=data, max_lead_time=time_limit)


class GPComposerBuilder:
    def __init__(self, task: Task):
        self._composer = GPComposer()
        self.optimiser_parameters = GPChainOptimiserParameters()
        self.task = task
        self.set_default_composer_params()

    def with_optimiser_parameters(self, optimiser_parameters: GPChainOptimiserParameters):
        self.optimiser_parameters = optimiser_parameters
        return self

    def with_requirements(self, requirements: GPComposerRequirements):
        self._composer.composer_requirements = requirements
        return self

    def with_metrics(self, metrics: Optional[MetricsEnum]):
        if type(metrics) is not list:
            metrics = [metrics]
        self._composer.metrics = metrics
        return self

    def with_initial_chain(self, initial_chain: Optional[Chain]):
        self._composer.initial_chain = initial_chain
        return self

    def set_default_composer_params(self):
        if not self._composer.composer_requirements:
            models, _ = ModelTypesRepository().suitable_model(task_type=self.task.task_type)
            self._composer.composer_requirements = GPComposerRequirements(primary=models, secondary=models)
        if not self._composer.metrics:
            metric_function = ClassificationMetricsEnum.ROCAUC_penalty
            if self.task.task_type in (TaskTypesEnum.regression, TaskTypesEnum.ts_forecasting):
                metric_function = RegressionMetricsEnum.RMSE
            self._composer.metrics = [metric_function]

    def build(self) -> Composer:
        optimiser_type = GPChainOptimiser
        if self.optimiser_parameters.genetic_scheme_type == GeneticSchemeTypesEnum.parameter_free:
            optimiser_type = GPChainParameterFreeOptimiser

        chain_generation_params = ChainGenerationParams()

        archive_type = None
        if len(self._composer.metrics) > 1:
            archive_type = tools.ParetoFront()
            self.optimiser_parameters.regularization_type = RegularizationTypesEnum.none
            self.optimiser_parameters.multi_objective = True

        optimiser = optimiser_type(initial_chain=self._composer.initial_chain,
                                   requirements=self._composer.composer_requirements,
                                   chain_generation_params=chain_generation_params,
                                   parameters=self.optimiser_parameters, log=self._composer.log,
                                   archive_type=archive_type)

        self._composer.optimiser = optimiser

        return self._composer
