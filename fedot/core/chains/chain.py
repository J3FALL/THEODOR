from copy import copy, deepcopy
from datetime import timedelta
from typing import List, Optional, Union
from uuid import uuid4

import networkx as nx

from fedot.core.chains.node import (FittedModelCache, Node, PrimaryNode, SecondaryNode, SharedCache)
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.utilities.synthetic.chain_template_new import ChainTemplate

ERROR_PREFIX = 'Invalid chain configuration:'


class Chain:
    """
    Base class used for composite model structure definition

    :param nodes: Node object(s)
    :param log: Log object to record messages

    .. note::
        fitted_on_data stores the data which were used in last chain fitting (equals None if chain hasn't been
        fitted yet)
    """

    def __init__(self, nodes: Optional[Union[Node, List[Node]]] = None,
                 log: Log = None):
        self.nodes = []
        self.log = log
        self.template = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if nodes:
            if isinstance(nodes, list):
                for node in nodes:
                    self.add_node(node)
            else:
                self.add_node(nodes)
        self.fitted_on_data = None

    def fit_from_scratch(self, input_data: InputData, verbose=False):
        """
        Method used for training the chain without using cached information

        :param input_data: data used for model training
        :param verbose: flag used for status printing to console, default False
        """
        # Clean all cache and fit all models
        self.log.info('Fit chain from scratch')
        self.fit(input_data, use_cache=False, verbose=verbose)

    def cache_status_if_new_data(self, new_input_data: InputData, cache_status: bool):
        if self.fitted_on_data is not None and self.fitted_on_data is not new_input_data:
            if cache_status:
                self.log.warn('Trained model cache is not actual because you are using new dataset for training. '
                              'Parameter use_cache value changed to False')
                cache_status = False
        return cache_status

    def fit(self, input_data: InputData, use_cache=True, verbose=False):
        """
        Run training process in all nodes in chain starting with root.

        :param input_data: data used for model training
        :param use_cache: flag defining whether use cache information about previous executions or not, default True
        :param verbose: flag used for status printing to console, default False
        """
        use_cache = self.cache_status_if_new_data(new_input_data=input_data, cache_status=use_cache)

        if not use_cache:
            self._clean_model_cache()

        if input_data.task.task_type == TaskTypesEnum.ts_forecasting:
            # the make_future_prediction is useless for the fit stage
            input_data.task.task_params.make_future_prediction = False
        else:
            if not use_cache or self.fitted_on_data is None:
                self.fitted_on_data = input_data
        train_predicted = self.root_node.fit(input_data=input_data, verbose=verbose)
        return train_predicted

    def predict(self, input_data: InputData, output_mode: str = 'default'):
        """
        Run the predict process in all nodes in chain starting with root.

        :param input_data: data for prediction
        :param output_mode: desired form of output for models. Available options are:
                'default' (as is),
                'labels' (numbers of classes - for classification) ,
                'probs' (probabilities - for classification =='default'),
                'full_probs' (return all probabilities - for binary classification).
        :return: array of predicted target values
        """

        if not self.is_all_cache_actual():
            ex = 'Trained model cache is not actual or empty'
            self.log.error(ex)
            raise ValueError(ex)

        result = self.root_node.predict(input_data=input_data, output_mode=output_mode)
        return result

    def fine_tune_primary_nodes(self, input_data: InputData, iterations: int = 30,
                                max_lead_time: timedelta = timedelta(minutes=5),
                                verbose=False):
        """
        Optimize hyperparameters in primary nodes models

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        """
        # Select all primary nodes
        # Perform fine-tuning for each model in node
        if verbose:
            self.log.info('Start tuning of primary nodes')

        all_primary_nodes = [node for node in self.nodes if isinstance(node, PrimaryNode)]
        for node in all_primary_nodes:
            node.fine_tune(input_data, max_lead_time=max_lead_time, iterations=iterations)

        if verbose:
            self.log.info('End tuning')

    def fine_tune_all_nodes(self, input_data: InputData, iterations: int = 30,
                            max_lead_time: timedelta = timedelta(minutes=5),
                            verbose=False):
        """
        Optimize hyperparameters in all nodes models

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info('Start tuning of chain')

        node = self.root_node
        node.fine_tune(input_data, max_lead_time=max_lead_time, iterations=iterations)

        if verbose:
            self.log.info('End tuning')

    def add_node(self, new_node: Node):
        """
        Add new node to the Chain

        :param new_node: new Node object
        """
        if new_node not in self.nodes:
            self.nodes.append(new_node)
            if new_node.nodes_from:
                for new_parent_node in new_node.nodes_from:
                    if new_parent_node not in self.nodes:
                        self.add_node(new_parent_node)

    def _actualise_old_node_childs(self, old_node: Node, new_node: Node):
        old_node_offspring = self.node_childs(old_node)
        for old_node_child in old_node_offspring:
            old_node_child.nodes_from[old_node_child.nodes_from.index(old_node)] = new_node

    def replace_node_with_parents(self, old_node: Node, new_node: Node):
        new_node = deepcopy(new_node)
        self._actualise_old_node_childs(old_node, new_node)
        new_nodes = [parent for parent in new_node.ordered_subnodes_hierarchy if not parent in self.nodes]
        old_nodes = [node for node in self.nodes if not node in old_node.ordered_subnodes_hierarchy]
        self.nodes = new_nodes + old_nodes
        self.sort_nodes()

    def update_node(self, old_node: Node, new_node: Node):
        self._actualise_old_node_childs(old_node, new_node)
        new_node.nodes_from = old_node.nodes_from
        self.nodes.remove(old_node)
        self.nodes.append(new_node)
        self.sort_nodes()

    def delete_node(self, node: Node):
        for node_child in self.node_childs(node):
            node_child.nodes_from.remove(node)
        for subtree_node in node.ordered_subnodes_hierarchy:
            self.nodes.remove(subtree_node)

    def _clean_model_cache(self):
        for node in self.nodes:
            node.cache = FittedModelCache(node)

    def is_all_cache_actual(self):
        cache_status = [node.cache.actual_cached_state is not None for node in self.nodes]
        return all(cache_status)

    def node_childs(self, node) -> List[Optional[Node]]:
        return [other_node for other_node in self.nodes if isinstance(other_node, SecondaryNode) and
                node in other_node.nodes_from]

    def _is_node_has_child(self, node) -> bool:
        return any(self.node_childs(node))

    def import_cache(self, fitted_chain: 'Chain'):
        for node in self.nodes:
            if not node.cache.actual_cached_state:
                for fitted_node in fitted_chain.nodes:
                    if fitted_node.descriptive_id == node.descriptive_id:
                        node.cache.import_from_other_cache(fitted_node.cache)
                        break

    # TODO why trees visualisation is incorrect?
    def sort_nodes(self):
        """layer by layer sorting"""
        nodes = self.root_node.ordered_subnodes_hierarchy
        self.nodes = nodes

    def save_chain(self, path: str):
        if not self.template:
            self.template = ChainTemplate(self)
        json_object = self.template.export_to_json(path)
        return json_object

    def load_chain(self, path: str):
        self.nodes = []
        self.log = default_log(__name__)
        self.template = ChainTemplate(self)
        self.template.import_from_json(path)

    def __eq__(self, other) -> bool:
        return self.root_node.descriptive_id == other.root_node.descriptive_id

    @property
    def root_node(self) -> Optional[Node]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not self._is_node_has_child(node)]
        if len(root) > 1:
            raise ValueError(f'{ERROR_PREFIX} More than 1 root_nodes in chain')
        return root[0]

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        def _depth_recursive(node):
            if node is None:
                return 0
            if isinstance(node, PrimaryNode):
                return 1
            else:
                return 1 + max([_depth_recursive(next_node) for next_node in node.nodes_from])

        return _depth_recursive(self.root_node)


class SharedChain(Chain):
    def __init__(self, base_chain: Chain, shared_cache: dict, log=None):
        super().__init__(log=log)
        self.nodes = copy(base_chain.nodes)
        for node in self.nodes:
            node.cache = SharedCache(node, global_cached_models=shared_cache)

    def unshare(self) -> Chain:
        chain = Chain()
        chain.nodes = copy(self.nodes)
        for node in chain.nodes:
            node.cache = FittedModelCache(node)
        return chain


def as_nx_graph(chain: Chain):
    graph = nx.DiGraph()
    node_labels = {}
    new_node_idx = {}
    for node in chain.nodes:
        unique_id, label = uuid4(), str(node)
        node_labels[unique_id] = str(node)
        new_node_idx[node] = unique_id
        graph.add_node(unique_id)

    def add_edges(graph, chain, new_node_idx):
        for node in chain.nodes:
            if node.nodes_from is not None:
                for child in node.nodes_from:
                    graph.add_edge(new_node_idx[child], new_node_idx[node])

    add_edges(graph, chain, new_node_idx)
    return graph, node_labels
