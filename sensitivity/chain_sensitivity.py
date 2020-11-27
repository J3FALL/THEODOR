from copy import deepcopy

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.chain import Chain
from fedot.core.models.data import InputData


class ChainStructureAnalyze:
    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData):
        self.chain = chain
        self.train_data = train_data
        self.test_data = test_data

    def analyze(self):
        response_matrix = self.get_model_response_matrix()
        # indices = self.variance_indices(response_matrix)
        indices = self.roc_auc_indices(response_matrix)

        return indices

    def get_model_response_matrix(self):
        results_predicted = []

        samples = self.sample()
        for chain_sample in samples:
            chain_sample.fit(self.train_data)
            output_data = chain_sample.predict(self.test_data)
            results_predicted.append(output_data.predict)

        return results_predicted

    def sample(self):
        chain_samples = []
        for index in range(1, len(self.chain.nodes)):
            chain_sample = deepcopy(self.chain)
            node_to_delete = chain_sample.nodes[index]
            chain_sample.delete_node_new(node_to_delete)
            chain_samples.append(chain_sample)

        return chain_samples

    def variance_indices(self, model_responses):
        self.chain.fit(self.train_data)
        true_output = self.chain.predict(self.test_data)
        true_variance = np.var(true_output.predict)
        indices = []
        for response in model_responses:
            indices.append(np.var(np.array(response)) / true_variance)

        return indices

    def roc_auc_indices(self, model_responses):
        self.chain.fit(self.train_data)
        predicted_originally = self.chain.predict(self.test_data)
        original_roc_auc = roc_auc(y_true=self.test_data.target, y_score=predicted_originally.predict)
        indices = []
        indices_loss = []
        for response in model_responses:
            current_roc_auc = roc_auc(y_true=self.test_data.target, y_score=response)
            indices.append(current_roc_auc)
            indices_loss.append(original_roc_auc - current_roc_auc)

        return indices, indices_loss
