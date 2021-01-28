from datetime import timedelta

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log, start_end_log_decorator
from fedot.core.chains.chain_template import ChainTemplate, ModelTemplate, extract_subtree_root


class Tune:
    def __init__(self, chain,
                 log: Log = default_log(__name__), verbose=False):
        self.chain = chain
        self.chain_template = ChainTemplate(self.chain)
        self.log = log
        self.verbose = verbose

    @start_end_log_decorator(start_msg='Starting tuning primary nodes',
                             end_msg='Primary nodes tuning is finished')
    def fine_tune_primary_nodes(self, input_data: InputData, iterations: int = 30,
                                max_lead_time: timedelta = timedelta(minutes=5)):
        """
        Optimize hyperparameters of models in primary nodes

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :return: updated chain object
        """

        all_primary_nodes = [node for node in self.chain.nodes if isinstance(node, PrimaryNode)]
        for node in all_primary_nodes:
            node.fine_tune(input_data, max_lead_time=max_lead_time, iterations=iterations)

        return self.chain

    @start_end_log_decorator(start_msg='Starting tuning root node',
                             end_msg='Root node tuning is finished')
    def fine_tune_root_node(self, input_data: InputData, iterations: int = 30,
                            max_lead_time: timedelta = timedelta(minutes=5)):
        """
        Optimize hyperparameters in the root node

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :return: updated chain object
        """

        node = self.chain.root_node
        if isinstance(node, PrimaryNode):
            # if mono-node chains
            node.fine_tune(input_data=input_data, max_lead_time=max_lead_time,
                           iterations=iterations)
        else:
            node.fine_tune(input_data=input_data, max_lead_time=max_lead_time,
                           iterations=iterations, recursive=False)

        return self.chain

    @start_end_log_decorator(start_msg='Starting tuning all nodes',
                             end_msg='All nodes tuning is finished')
    def fine_tune_all_nodes(self, input_data: InputData, iterations: int = 30,
                            max_lead_time: timedelta = timedelta(minutes=5)):
        """
        Optimize hyperparameters of models in all nodes

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :return: updated chain object
        """

        node = self.chain.root_node
        node.fine_tune(input_data, max_lead_time=max_lead_time, iterations=iterations, recursive=True)

        return self.chain

    @start_end_log_decorator(start_msg='Starting tuning chosen node',
                             end_msg='Chosen node tuning is finished')
    def fine_tune_certain_node(self, model_id, input_data: InputData, iterations: int = 30,
                               max_lead_time: timedelta = timedelta(minutes=5)):
        """
        Optimize hyperparameters of models in the certain node,
        defined by model id

        :param int model_id: number of the certain model in the chain.
        Look for it in exported json file of your model.
        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :return: updated chain object
        """

        subchain = Chain()
        new_root = extract_subtree_root(root_model_id=model_id,
                                        chain_template=self.chain_template)
        subchain.add_node(new_root)
        subchain.fit(input_data=input_data, use_cache=False)

        updated_subchain = Tune(subchain).fine_tune_root_node(input_data=input_data,
                                                              iterations=iterations,
                                                              max_lead_time=max_lead_time)

        self._update_template(model_id=model_id,
                              updated_node=updated_subchain.root_node)

        updated_chain = Chain()
        self.chain_template.convert_to_chain(chain=updated_chain)

        return updated_chain

    def _update_template(self, model_id, updated_node):
        model_template = [model_template for model_template in self.chain_template.model_templates
                          if model_template.model_id == model_id][0]
        update_node_template = ModelTemplate(updated_node)

        model_template.params = update_node_template.params
        model_template.fitted_model_path = update_node_template.fitted_model_path
