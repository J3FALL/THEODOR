from copy import deepcopy

from fedot.core.composer.chain import Chain
from fedot.core.composer.chain_tune import Tune
from fedot.core.models.data import InputData
from sklearn.metrics import roc_auc_score as roc_auc
from datetime import timedelta
from fedot.core.composer.metrics import RocAucMetric


class NodeAnalysis:
    def __init__(self):
        self._node_delition_analyze = NodeDeletionAnalyze
        self._model_hparams_analyze = NodeTuneAnalyze

    def analyze(self, chain: Chain, node_id: int,
                train_data: InputData, test_data: InputData):
        deletion_result = self._node_delition_analyze(chain=chain,
                                                      train_data=train_data,
                                                      test_data=test_data).analyze(node_id=node_id)

        hparams_result = self._model_hparams_analyze(chain=chain,
                                                     train_data=train_data,
                                                     test_data=test_data).analyze(node_id=node_id)
        result = self._create_analysis_result(deletion_result, hparams_result)

        return result

    def _create_analysis_result(self, *args):
        del_result, hp_result = args
        result = {
            'deleted': del_result,
            'tuned': hp_result,
        }
        return result


class NodeAnalyzeApproach:
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        self._chain = chain
        self._train_data = train_data
        self._test_data = test_data

    def analyze(self, *args) -> float:
        """Create the difference metric(scorer, index, etc) of the changed
        chain in relation to original one"""
        pass

    def sample(self, *args) -> Chain:
        """Changes the chain in the relative to the approach way"""
        pass

    def compare_with_origin(self, changed_chain: Chain):
        changed_chain.fit(input_data=self._train_data, use_cache=False)
        predicted = changed_chain.predict(input_data=self._test_data)
        changed_chain_roc_auc = roc_auc(y_true=self._test_data.target,
                                       y_score=predicted.predict)

        self._chain.fit(self._train_data)
        predicted_originally = self._chain.predict(self._test_data)
        original_roc_auc = roc_auc(y_true=self._test_data.target,
                                   y_score=predicted_originally.predict)

        return changed_chain_roc_auc - original_roc_auc


class NodeDeletionAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeDeletionAnalyze, self).__init__(chain, train_data, test_data)

    def analyze(self, node_id: int):
        shortend_chain = self.sample(node_id)
        loss = self.compare_with_origin(shortend_chain)

        return loss

    def sample(self, node_id: int):
        chain_sample = deepcopy(self._chain)
        node_to_delete = chain_sample.nodes[node_id]
        chain_sample.delete_node_new(node_to_delete)

        return chain_sample


class NodeTuneAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeTuneAnalyze, self).__init__(chain, train_data, test_data)

    def analyze(self, node_id: int):
        tuned_chain = Tune(self._chain).fine_tune_certain_node(model_id=node_id,
                                                               input_data=self._train_data,
                                                               max_lead_time=timedelta(minutes=1),
                                                               iterations=30)
        loss = self.compare_with_origin(tuned_chain)

        return loss


class NodeReplaceModelAnalayze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeReplaceModelAnalayze, self).__init__(chain, train_data, test_data)
