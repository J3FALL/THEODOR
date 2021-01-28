import random
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from typing import Optional, List, Union, Type

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_tune import Tune
from fedot.core.chains.node import Node, PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.utilities.define_metric_by_task import MetricByTask


class NodeAnalysis:
    def __init__(self, approaches: Optional[List[Type['NodeAnalyzeApproach']]] = None):
        if not approaches:
            self.approaches = [NodeDeletionAnalyze, NodeTuneAnalyze, NodeReplaceModelAnalyze]
        else:
            self.approaches = approaches

    def analyze(self, chain: Chain, node_id: int,
                train_data: InputData, test_data: InputData) -> dict:

        results = dict()
        for approach in self.approaches:
            results[f'{approach.__name__}'] = approach(chain=chain,
                                                       train_data=train_data,
                                                       test_data=test_data).analyze(node_id=node_id)

        return results


class NodeAnalyzeApproach(ABC):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        self._chain = chain
        self._train_data = train_data
        self._test_data = test_data
        self.metric = MetricByTask(self._train_data.task.task_type)

    @abstractmethod
    def analyze(self, node_id) -> Union[List[dict], float]:
        """Create the difference metric(scorer, index, etc) of the changed
        chain in relation to the original one"""
        pass

    @abstractmethod
    def sample(self, *args) -> Union[Union[List[Chain], Chain], List[dict]]:
        """Changes the chain according to the approach"""
        pass

    def _compare_with_origin_by_metric(self, changed_chain: Chain) -> float:
        original_metric = self._get_metric_value(chain=self._chain)

        changed_chain_metric = self._get_metric_value(chain=changed_chain)

        return changed_chain_metric - original_metric

    def _get_metric_value(self, chain: Chain) -> float:
        chain.fit(self._train_data, use_cache=False)
        predicted = chain.predict(self._test_data)
        metric_value = self.metric.get_value(true=self._test_data,
                                             predicted=predicted)

        return metric_value


class NodeDeletionAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeDeletionAnalyze, self).__init__(chain, train_data, test_data)

    def analyze(self, node_id: int) -> Union[List[dict], float]:
        if node_id == 0:
            # TODO or warning?
            return 0.0
        else:
            shortend_chain = self.sample(node_id)
            loss = self._compare_with_origin_by_metric(shortend_chain)

            return loss

    def sample(self, node_id: int):
        chain_sample = deepcopy(self._chain)
        node_to_delete = chain_sample.nodes[node_id]
        chain_sample.delete_node_with_redirection(node_to_delete)

        return chain_sample

    def __str__(self):
        return 'NodeDeletionAnalyze'


class NodeTuneAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeTuneAnalyze, self).__init__(chain, train_data, test_data)

    def analyze(self, node_id: int) -> Union[List[dict], float]:
        tuned_chain = Tune(self._chain).fine_tune_certain_node(model_id=node_id,
                                                               input_data=self._train_data,
                                                               max_lead_time=timedelta(minutes=1),
                                                               iterations=30)
        loss = self._compare_with_origin_by_metric(tuned_chain)

        return loss

    def sample(self, *args) -> Union[List[Chain], Chain]:
        raise NotImplemented

    def __str__(self):
        return 'NodeTuneAnalyze'


class NodeReplaceModelAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeReplaceModelAnalyze, self).__init__(chain, train_data, test_data)

    def analyze(self, node_id: int,
                nodes_to_replace_to: Optional[List[Node]] = None,
                number_of_random_models: Optional[int] = 3) -> Union[List[dict], float]:
        # metric_by_task = MetricByTask(self._train_data.task.task_type)

        samples = self.sample(node_id=node_id,
                              nodes_to_replace_to=nodes_to_replace_to,
                              number_of_random_models=number_of_random_models)

        loss = []
        for sample_chain in samples:
            loss_per_sample = self._compare_with_origin_by_metric(sample_chain)

            new_node = sample_chain.nodes[node_id]
            loss.append({f'with {new_node.model.model_type} instead of {self._chain.nodes[node_id]}': loss_per_sample})

        return loss

    def sample(self, node_id: int,
               nodes_to_replace_to: Optional[List[Node]],
               number_of_random_models: Optional[int]) -> Union[List[Chain], Chain]:

        # TODO Refactor according to future different types of Nodes
        if not nodes_to_replace_to:
            if isinstance(self._chain.nodes[node_id], PrimaryNode):
                node_type = PrimaryNode
            elif isinstance(self._chain.nodes[node_id], SecondaryNode):
                node_type = SecondaryNode
            else:
                raise ValueError('Unsupported type of Node. Expected Primary or Secondary')

            nodes_to_replace_to = self._node_generation(node_type=node_type,
                                                        number_of_models=number_of_random_models)

        samples = list()
        for replacing_node in nodes_to_replace_to:
            sample_chain = deepcopy(self._chain)
            replaced_node = sample_chain.nodes[node_id]
            sample_chain.update_node(old_node=replaced_node,
                                     new_node=replacing_node)
            samples.append(sample_chain)

        return samples

    def _node_generation(self, node_type: Union[Type[PrimaryNode], Type[SecondaryNode]],
                         number_of_models: int = 3) -> List[Node]:
        available_models, _ = ModelTypesRepository().suitable_model(task_type=self._train_data.task.task_type)
        random_models = random.sample(available_models, number_of_models)

        nodes = []
        for model in random_models:
            nodes.append(node_type(model_type=model))

        return nodes

    def __str__(self):
        return 'NodeReplaceModelAnalyze'
