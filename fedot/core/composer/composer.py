import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Any, Callable, List, Optional)

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log


@dataclass
class ComposerRequirements:
    """
    This dataclass is for defining the requirements of composition process

    :param primary: List of model types (str) for Primary Nodes
    :param secondary: List of model types (str) for Secondary Nodes
    :param max_lead_time: max time in minutes available for composition process
    :param max_depth: max depth of the result chain
    :param max_arity: maximal number of parent for node
    :param min_arity: minimal number of parent for node
    :param add_single_model_chains: allow to have chain with only one node
    """
    primary: List[str]
    secondary: List[str]
    max_lead_time: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    max_depth: int = 3
    max_arity: int = 2
    min_arity: int = 2
    add_single_model_chains: bool = True

    def __post_init__(self):
        if self.max_depth < 0:
            raise ValueError(f'invalid max_depth value')
        if self.max_arity < 0:
            raise ValueError(f'invalid max_arity value')
        if self.min_arity < 0:
            raise ValueError(f'invalid min_arity value')


class Composer(ABC):
    """
    Base class used for receiving composite models via optimization
    :param metrics: metrics used to define the quality of found solution
    :param composer_requirements: requirements for composition process
    :param optimiser_parameters: parameters used by optimization process (i.e. GPComposerRequirements)
    :param initial_chain: defines the initial state of the population. If None then initial population is random.
    :param log: optional parameter for log oject
    """

    def __init__(self, metrics: Optional[Callable], composer_requirements: ComposerRequirements,
                 optimiser_parameters: Any = None, initial_chain: Optional[Chain] = None,
                 log: Log = None):
        self.history = None
        self.metrics = metrics
        self.composer_requirements = composer_requirements
        self.optimiser_parameters = optimiser_parameters
        self.initial_chain = initial_chain

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @abstractmethod
    def compose_chain(self, data: InputData,
                      is_visualise: bool = False) -> Chain:
        """
        Base method to run the composition process

        :param data: data used for problem solving
        :param is_visualise: flag to enable visualization. Default False.
        :return: Chain object
        """
        raise NotImplementedError()
