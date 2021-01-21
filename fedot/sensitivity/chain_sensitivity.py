from typing import List, Optional, Type

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.sensitivity.sensitivity_facade import NodeAnalyzeApproach, \
    NodeDeletionAnalyze, NodeAnalysis


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
            nodes_to_analyze = self.chain.nodes
        elif self.certain_nodes:
            nodes_to_analyze = self.certain_nodes
        else:
            raise ValueError("Define nodes to analyze: all_nodes or nodes_ids_to_analyze")

        nodes_results = dict()
        for index in range(0, len(nodes_to_analyze)):
            node_result = NodeAnalysis(approaches=self.approaches). \
                analyze(chain=self.chain, node_id=index,
                        train_data=self.train_data,
                        test_data=self.test_data)

            nodes_results[index] = node_result

        return nodes_results

    # TODO replace metrics
    # def variance_indices(self, model_responses):
    #     self.chain.fit(self.train_data)
    #     true_output = self.chain.predict(self.test_data)
    #     true_variance = np.var(true_output.predict)
    #     indices = []
    #     for response in model_responses:
    #         indices.append(np.var(np.array(response)) / true_variance)
    #
    #     return indices
    #
    # def roc_auc_indices(self, model_responses):
    #     self.chain.fit(self.train_data)
    #     predicted_originally = self.chain.predict(self.test_data)
    #     original_roc_auc = roc_auc(y_true=self.test_data.target, y_score=predicted_originally.predict)
    #     indices = []
    #     indices_loss = []
    #     for response in model_responses:
    #         current_roc_auc = roc_auc(y_true=self.test_data.target, y_score=response)
    #         indices.append(current_roc_auc)
    #         indices_loss.append(original_roc_auc - current_roc_auc)
    #
    #     return indices, indices_loss
