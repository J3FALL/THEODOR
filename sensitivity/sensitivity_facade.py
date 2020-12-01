from copy import deepcopy
from datetime import timedelta
from typing import Optional, List, Union

from fedot.core.composer.chain import Chain
from fedot.core.composer.chain_tune import Tune
from fedot.core.composer.node import Node
from fedot.core.models.data import InputData
from fedot.utilities.define_metric_by_task import MetricByTask


class NodeAnalysis:
    def __init__(self, approaches: Optional[List['NodeAnalyzeApproach']] = None):
        if approaches:
            self._node_deletion_analyze = [approach for approach in approaches
                                           if isinstance(approach, NodeDeletionAnalyze)][0]
            self._model_hparams_analyze = [approach for approach in approaches
                                           if isinstance(approach, NodeTuneAnalyze)][0]
        else:
            self._node_deletion_analyze = NodeDeletionAnalyze
            self._model_hparams_analyze = NodeTuneAnalyze

    def analyze(self, chain: Chain, node_id: int,
                train_data: InputData, test_data: InputData):
        deletion_result = self._node_deletion_analyze(chain=chain,
                                                      train_data=train_data,
                                                      test_data=test_data).analyze(node_id=node_id)

        hparams_result = self._model_hparams_analyze(chain=chain,
                                                     train_data=train_data,
                                                     test_data=test_data).analyze(node_id=node_id)
        result = self._analysis_result(deletion_result, hparams_result)

        return result

    def _analysis_result(self, *args):
        del_result, hp_result = args
        result = {
            f'{str(self._node_deletion_analyze.__name__)}': del_result,
            f'{str(self._model_hparams_analyze.__name__)}': hp_result,
        }
        return result


class NodeAnalyzeApproach:
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        self._chain = chain
        self._train_data = train_data
        self._test_data = test_data

    def analyze(self, *args) -> Union[List[float], float]:
        """Create the difference metric(scorer, index, etc) of the changed
        chain in relation to the original one"""
        pass

    def sample(self, *args) -> Union[List[Chain], Chain]:
        """Changes the chain according to the approach"""
        pass

    def compare_with_origin_by_metric(self, changed_chain: Chain,
                                      original_metric: Optional[float] = None,
                                      metric_by_task: Optional[MetricByTask] = None):
        if not metric_by_task:
            metric_by_task = MetricByTask(self._train_data.task.task_type)

        if not original_metric:
            original_metric = self._get_metric_value(chain=self._chain, metric=metric_by_task)

        changed_chain_metric = self._get_metric_value(chain=changed_chain, metric=metric_by_task)

        return changed_chain_metric - original_metric

    def _get_metric_value(self, chain: Chain, metric: MetricByTask):
        chain.fit(self._train_data, use_cache=False)
        predicted_originally = self._chain.predict(self._test_data)
        metric_value = metric.get_value(self._test_data,
                                        predicted_originally)

        return metric_value


class NodeDeletionAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeDeletionAnalyze, self).__init__(chain, train_data, test_data)

    def analyze(self, node_id: int) -> Union[List[float], float]:
        shortend_chain = self.sample(node_id)
        loss = self.compare_with_origin_by_metric(shortend_chain)

        return loss

    def sample(self, node_id: int):
        chain_sample = deepcopy(self._chain)
        node_to_delete = chain_sample.nodes[node_id]
        chain_sample.delete_node_new(node_to_delete)

        return chain_sample


class NodeTuneAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeTuneAnalyze, self).__init__(chain, train_data, test_data)

    def analyze(self, node_id: int) -> Union[List[float], float]:
        tuned_chain = Tune(self._chain).fine_tune_certain_node(model_id=node_id,
                                                               input_data=self._train_data,
                                                               max_lead_time=timedelta(minutes=1),
                                                               iterations=30)
        loss = self.compare_with_origin_by_metric(tuned_chain)

        return loss


class NodeReplaceModelAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(NodeReplaceModelAnalyze, self).__init__(chain, train_data, test_data)

    def analyze(self, node_id: int,
                nodes_to_replace_to: Optional[List[Node]]) -> Union[List[float], float]:
        metric_by_task = MetricByTask(self._train_data.task.task_type)

        samples = self.sample(node_id=node_id,
                              nodes_to_replace_to=nodes_to_replace_to)

        original_metric = self._get_metric_value(chain=self._chain, metric=metric_by_task)

        loss = []
        for sample in samples:
            loss_per_sample = self.compare_with_origin_by_metric(sample,
                                                                 metric_by_task=metric_by_task,
                                                                 original_metric=original_metric)
            loss.append(loss_per_sample)

        return loss

    def sample(self, node_id: int, nodes_to_replace_to: Optional[List[Node]]) -> Union[List[Chain], Chain]:
        if nodes_to_replace_to:
            samples = list()
            for replacing_node in nodes_to_replace_to:
                sample_chain = deepcopy(self._chain)
                replaced_node = sample_chain.nodes[node_id]
                sample_chain.update_node(old_node=replaced_node,
                                         new_node=replacing_node)
                samples.append(sample_chain)

            return samples
        # TODO Random node generation/choosing
        else:
            pass
