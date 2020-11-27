from abc import ABC
from collections import namedtuple
from copy import copy
from datetime import timedelta
from typing import Callable, List, Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.preprocessing import preprocessing_func_for_data
from fedot.core.data.transformation import transformation_function_for_data
from fedot.core.log import default_log
from fedot.core.models.model import Model

CachedState = namedtuple('CachedState', 'preprocessor model')


class Node(ABC):
    """
    Base class for Node definition in Chain structure

    :param nodes_from: parent nodes which information comes from
    :param model_type: str type of the model defined in model repository
    :param manual_preprocessing_func: optional function for data preprocessing.
    If not defined one of the available preprocessing strategies is used. \
    See the `preprocessors <https://github.com/nccr-itmo/FEDOT/blob/master/core/models/preprocessing.py>`__
    :param log: Log object to record messages
    """

    def __init__(self, nodes_from: Optional[List['Node']], model_type: [str, 'Model'],
                 manual_preprocessing_func: Optional[Callable] = None,
                 log=None):
        self.nodes_from = nodes_from
        self.cache = FittedModelCache(self)
        self.manual_preprocessing_func = manual_preprocessing_func
        self.log = log

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if not isinstance(model_type, str):
            self.model = model_type
        else:
            self.model = Model(model_type=model_type)

    @property
    def descriptive_id(self):
        return self._descriptive_id_recursive(visited_nodes=[])

    def _descriptive_id_recursive(self, visited_nodes):
        node_label = self.model.description
        if self.manual_preprocessing_func:
            node_label = f'{node_label}_custom_preprocessing={self.manual_preprocessing_func.__name__}'
        full_path = ''
        if self in visited_nodes:
            return 'ID_CYCLED'
        visited_nodes.append(self)
        if self.nodes_from:
            previous_items = []
            for parent_node in self.nodes_from:
                previous_items.append(f'{parent_node._descriptive_id_recursive(copy(visited_nodes))};')
            previous_items.sort()
            previous_items_str = ';'.join(previous_items)

            full_path += f'({previous_items_str})'
        full_path += f'/{node_label}'
        return full_path

    @property
    def model_tags(self) -> List[str]:
        return self.model.metadata.tags

    def output_from_prediction(self, input_data, prediction):
        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=prediction, task=input_data.task,
                          data_type=self.model.output_datatype(input_data.data_type))

    def _transform(self, input_data: InputData):
        transformed_data = transformation_function_for_data(
            input_data_type=input_data.data_type,
            required_data_types=self.model.metadata.input_types)(input_data)
        return transformed_data

    def _preprocess(self, data: InputData):
        preprocessing_func = preprocessing_func_for_data(data, self)

        if not self.cache.actual_cached_state:
            # if fitted preprocessor not found in cache
            preprocessing_strategy = \
                preprocessing_func().fit(data.features)
        else:
            # if fitted preprocessor already exists
            preprocessing_strategy = self.cache.actual_cached_state.preprocessor

        data.features = preprocessing_strategy.apply(data.features)

        return data, preprocessing_strategy

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        """
        Run training process in the node

        :param input_data: data used for model training
        :param verbose: flag used for status printing to console, default False
        """
        transformed = self._transform(input_data)
        preprocessed_data, preproc_strategy = self._preprocess(transformed)

        if not self.cache.actual_cached_state:
            if verbose:
                print('Cache is not actual')

            cached_model, model_predict = self.model.fit(data=preprocessed_data)
            self.cache.append(CachedState(preprocessor=copy(preproc_strategy),
                                          model=cached_model))
        else:
            if verbose:
                print('Model were obtained from cache')

            model_predict = self.model.predict(fitted_model=self.cache.actual_cached_state.model,
                                               data=preprocessed_data)

        return self.output_from_prediction(input_data, model_predict)

    def predict(self, input_data: InputData, output_mode: str = 'default', verbose=False) -> OutputData:
        """
        Run prediction process in the node

        :param input_data: data used for prediction
        :param output_mode: desired output for models (e.g. labels, probs, full_probs)
        :param verbose: flag used for status printing to console, default False
        """
        transformed = self._transform(input_data)
        preprocessed_data, _ = self._preprocess(transformed)

        if not self.cache:
            raise ValueError('Model must be fitted before predict')

        model_predict = self.model.predict(fitted_model=self.cache.actual_cached_state.model,
                                           data=preprocessed_data, output_mode=output_mode)

        return self.output_from_prediction(input_data, model_predict)

    def fine_tune(self, input_data: InputData,
                  max_lead_time: timedelta = timedelta(minutes=5), iterations: int = 30):
        """
        Run the process of hyperparameter optimization for the node

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        """

        transformed = self._transform(input_data)
        preprocessed_data, preproc_strategy = self._preprocess(transformed)

        fitted_model, _ = self.model.fine_tune(preprocessed_data,
                                               max_lead_time=max_lead_time,
                                               iterations=iterations)

        self.cache.append(CachedState(preprocessor=copy(preproc_strategy),
                                      model=fitted_model))

    def __str__(self):
        model = f'{self.model}'
        return model

    def __repr__(self):
        return self.__str__()

    @property
    def ordered_subnodes_hierarchy(self) -> List['Node']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.ordered_subnodes_hierarchy
        return nodes

    @property
    def custom_params(self) -> dict:
        return self.model.params

    @custom_params.setter
    def custom_params(self, params):
        if params:
            self.model.params = params


class FittedModelCache:
    def __init__(self, related_node: Node):
        self._local_cached_models = {}
        self._related_node_ref = related_node

    def append(self, fitted_model):
        self._local_cached_models[self._related_node_ref.descriptive_id] = fitted_model

    def import_from_other_cache(self, other_cache: 'FittedModelCache'):
        for entry_key in other_cache._local_cached_models.keys():
            self._local_cached_models[entry_key] = other_cache._local_cached_models[entry_key]

    def clear(self):
        self._local_cached_models = {}

    @property
    def actual_cached_state(self):
        found_model = self._local_cached_models.get(self._related_node_ref.descriptive_id, None)
        return found_model


class SharedCache(FittedModelCache):
    def __init__(self, related_node: Node, global_cached_models: dict):
        super().__init__(related_node)
        self._global_cached_models = global_cached_models

    def append(self, fitted_model):
        super().append(fitted_model)
        if self._global_cached_models is not None:
            self._global_cached_models[self._related_node_ref.descriptive_id] = fitted_model

    @property
    def actual_cached_state(self):
        found_model = super().actual_cached_state

        if not found_model and self._global_cached_models:
            found_model = self._global_cached_models.get(self._related_node_ref.descriptive_id, None)
        return found_model


class PrimaryNode(Node):
    """
    The class defines the interface of Primary nodes where initial task data is located

    :param model_type: str type of the model defined in model repository
    :param manual_preprocessing_func: optional function for data preprocessing.
    :param model: optional custom atomized_model
    :param kwargs: optional arguments (i.e. logger)
    """

    def __init__(self, model_type: [str, 'Model'],
                 manual_preprocessing_func: Optional[Callable] = None, **kwargs):
        super().__init__(nodes_from=None, model_type=model_type,
                         manual_preprocessing_func=manual_preprocessing_func, **kwargs)

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        """
        Fit the model located in the primary node

        :param input_data: data used for model training
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info(f'Trying to fit primary node with model: {self.model}')

        return super().fit(input_data, verbose)

    def predict(self, input_data: InputData,
                output_mode: str = 'default', verbose=False) -> OutputData:
        """
        Predict using the model located in the primary node

        :param input_data: data used for prediction
        :param output_mode: desired output for models (e.g. labels, probs, full_probs)
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info(f'Predict in primary node by model: {self.model}')

        return super().predict(input_data, output_mode, verbose)


class SecondaryNode(Node):
    """
    The class defines the interface of Secondary nodes modifying tha data flow in Chain

    :param model_type: str type of the model defined in model repository
    :param nodes_from: parent nodes where data comes from
    :param manual_preprocessing_func: optional function for data preprocessing.
    :param model: optional custom atomized_model
    :param kwargs: optional arguments (i.e. logger)
    """

    def __init__(self, model_type: [str, 'Model'], nodes_from: Optional[List['Node']] = None,
                 manual_preprocessing_func: Optional[Callable] = None, **kwargs):
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, model_type=model_type,
                         manual_preprocessing_func=manual_preprocessing_func, **kwargs)

    def fit(self, input_data: InputData, verbose=False) -> OutputData:
        """
        Fit the model located in the secondary node

        :param input_data: data used for model training
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info(f'Trying to fit secondary node with model: {self.model}')

        secondary_input = self._input_from_parents(input_data=input_data,
                                                   parent_operation='fit',
                                                   verbose=verbose)
        return super().fit(input_data=secondary_input)

    def predict(self, input_data: InputData, output_mode: str = 'default', verbose=False) -> OutputData:
        """
        Predict using the model located in the secondary node

        :param input_data: data used for prediction
        :param output_mode: desired output for models (e.g. labels, probs, full_probs)
        :param verbose: flag used for status printing to console, default False
        """
        if verbose:
            self.log.info(f'Obtain prediction in secondary node with model: {self.model}')

        secondary_input = self._input_from_parents(input_data=input_data,
                                                   parent_operation='predict',
                                                   verbose=verbose)

        return super().predict(input_data=secondary_input, output_mode=output_mode, verbose=verbose)

    def fine_tune(self, input_data: InputData, recursive: bool = True,
                  max_lead_time: timedelta = timedelta(minutes=5), iterations: int = 30,
                  verbose: bool = False):
        """
        Run the process of hyperparameter optimization for the node

        :param recursive: flag to initiate the tuning in the parent nodes or not, default: True
        :param input_data: data used for tuning
        :param max_lead_time: max time available for tuning process
        :param iterations: max number of iterations
        :param verbose: flag used for status printing to console, default True
        """
        if verbose:
            self.log.info(f'Tune all parent nodes in secondary node with model: {self.model}')

        if recursive:
            secondary_input = self._input_from_parents(input_data=input_data,
                                                       parent_operation='fine_tune',
                                                       max_tune_time=max_lead_time, verbose=verbose)
        else:
            secondary_input = self._input_from_parents(input_data=input_data,
                                                       parent_operation='fit',
                                                       max_tune_time=max_lead_time, verbose=verbose)

        return super().fine_tune(input_data=secondary_input)

    def _nodes_from_with_fixed_order(self):
        if self.nodes_from is not None:
            return sorted(self.nodes_from, key=lambda node: node.descriptive_id)

    def _input_from_parents(self, input_data: InputData,
                            parent_operation: str,
                            max_tune_time: Optional[timedelta] = None,
                            verbose=False) -> InputData:
        if len(self.nodes_from) == 0:
            raise ValueError()

        if verbose:
            self.log.info(f'Fit all parent nodes in secondary node with model: {self.model}')

        parent_nodes = self._nodes_from_with_fixed_order()

        are_prev_nodes_affect_target = \
            ['affects_target' in parent_node.model_tags for parent_node in parent_nodes]
        if any(are_prev_nodes_affect_target):
            # is the previous model is the model that changes target
            parent_results, target = _combine_parents_that_affects_target(parent_nodes, input_data,
                                                                          parent_operation)
        else:
            parent_results, target = _combine_parents_simple(parent_nodes, input_data,
                                                             parent_operation, max_tune_time)

        secondary_input = InputData.from_predictions(outputs=parent_results,
                                                     target=target)

        return secondary_input


def _combine_parents_that_affects_target(parent_nodes: List[Node],
                                         input_data: InputData,
                                         parent_operation: str):
    if len(parent_nodes) > 1:
        raise NotImplementedError()

    if parent_operation == 'predict':
        parent_result = parent_nodes[0].predict(input_data=input_data)
    elif parent_operation == 'fit' or parent_operation == 'fine_tune':
        parent_result = parent_nodes[0].fit(input_data=input_data)
    else:
        raise NotImplementedError()

    target = parent_result.predict
    return [parent_result], target


def _combine_parents_simple(parent_nodes: List[Node],
                            input_data: InputData,
                            parent_operation: str,
                            max_tune_time: Optional[timedelta]):
    target = input_data.target
    parent_results = []
    for parent in parent_nodes:
        if parent_operation == 'predict':
            prediction = parent.predict(input_data=input_data)
            parent_results.append(prediction)
        elif parent_operation == 'fit':
            prediction = parent.fit(input_data=input_data)
            parent_results.append(prediction)
        elif parent_operation == 'fine_tune':
            parent.fine_tune(input_data=input_data, max_lead_time=max_tune_time)
            prediction = parent.predict(input_data=input_data)
            parent_results.append(prediction)
        else:
            raise NotImplementedError()

    return parent_results, target
