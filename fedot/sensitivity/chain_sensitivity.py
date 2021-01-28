from typing import List, Optional, Type

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.sensitivity.node_sensitivity import NodeAnalyzeApproach, \
    NodeAnalysis


class ChainStructureAnalyze:
    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData,
                 approaches: Optional[List[Type[NodeAnalyzeApproach]]] = None,
                 metric: str = None, nodes_ids_to_analyze: List[int] = None,
                 all_nodes: bool = False):

        if all_nodes and nodes_ids_to_analyze:
            raise ValueError("Choose only one parameter between all_nodes and nodes_ids_to_analyze")

        self.chain = chain
        self.train_data = train_data
        self.test_data = test_data
        self.approaches = approaches
        self.certain_nodes = nodes_ids_to_analyze
        self.all_nodes = all_nodes
        self.metric = metric

    def analyze(self) -> dict:
        if self.all_nodes:
            nodes_ids_to_analyze = [i for i in range(len(self.chain.nodes))]
        elif self.certain_nodes:
            nodes_ids_to_analyze = self.certain_nodes
        else:
            raise ValueError("Define nodes to analyze: all_nodes or nodes_ids_to_analyze")

        nodes_results = dict()
        for index in nodes_ids_to_analyze:
            node_result = NodeAnalysis(approaches=self.approaches). \
                analyze(chain=self.chain, node_id=index,
                        train_data=self.train_data,
                        test_data=self.test_data)

            nodes_results[f'id = {index}, model = {self.chain.nodes[index].model.model_type}'] = node_result

        return nodes_results
