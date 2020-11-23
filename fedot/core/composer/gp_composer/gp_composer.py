from dataclasses import dataclass
from functools import partial
from sys import maxsize as max_int_value
from typing import (
    Callable,
    Optional,
)

from fedot.core.chain_validation import validate
from fedot.core.composer.chain import Chain, SharedChain
from fedot.core.composer.composer import Composer, ComposerRequirements
from fedot.core.composer.node import PrimaryNode, SecondaryNode
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiser, GPChainOptimiserParameters
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.composer.write_history import write_composer_history_to_csv
from fedot.core.models.data import InputData, train_test_data_setup


@dataclass
class GPComposerRequirements(ComposerRequirements):
    pop_size: Optional[int] = 20
    num_of_generations: Optional[int] = 100
    crossover_prob: Optional[float] = 0.8
    mutation_prob: Optional[float] = 0.8


class GPComposer(Composer):
    def __init__(self):
        super().__init__()
        self.shared_cache = {}

    def compose_chain(self, data: InputData, initial_chain: Optional[Chain],
                      composer_requirements: Optional[GPComposerRequirements],
                      metrics: Optional[Callable], optimiser_parameters: GPChainOptimiserParameters = None,
                      is_visualise: bool = False, is_tune: bool = False) -> Chain:

        if data.target is not None:
            train_data, test_data = train_test_data_setup(data, 0.8)
        else:
            # if the problem is unsupervised
            train_data, test_data = data, data

        self.shared_cache.clear()

        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            metrics, train_data, test_data, True)

        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements,
                                     primary_node_func=PrimaryNode,
                                     secondary_node_func=SecondaryNode, chain_class=Chain,
                                     parameters=optimiser_parameters)

        best_chain, self.history = optimiser.optimise(metric_function_for_nodes)
        historical_fitness = [chain.fitness for chain in self.history]

        if is_visualise:
            ComposerVisualiser.visualise_history(self.history, historical_fitness)

        write_composer_history_to_csv(historical_fitness=historical_fitness, historical_chains=self.history,
                                      pop_size=composer_requirements.pop_size)

        self.log.info('GP composition finished')

        if is_tune:
            self.tune_chain(best_chain, data, composer_requirements.max_lead_time)
        return best_chain

    def metric_for_nodes(self, metric_function, train_data: InputData,
                         test_data: InputData, is_chain_shared: bool,
                         chain: Chain) -> float:
        try:
            validate(chain)
            if is_chain_shared:
                chain = SharedChain(base_chain=chain, shared_cache=self.shared_cache)
            chain.fit(input_data=train_data)
            return metric_function(chain, test_data)
        except Exception as ex:
            self.log.info(f'Error in chain assessment during composition: {ex}. Continue.')
            return max_int_value

    @staticmethod
    def tune_chain(chain: Chain, data: InputData, time_limit):
        chain.fine_tune_all_nodes(input_data=data, max_lead_time=time_limit)
