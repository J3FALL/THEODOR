from abc import ABC, abstractmethod
import joblib
import os

from fedot.core.chains.node import Node
from fedot.core.data.preprocessing import preprocessing_strategy_class_by_label, preprocessing_strategy_label_by_class
from fedot.core.log import default_log, Log


class ModelTemplateAbstract(ABC):
    """
    Base class used for create different types of Model("atomized_model" or others like("knn", "xgboost")).
    Atomized_model is chain which can be used like general model.
    """

    def __init__(self, log: Log = None):
        self.model_id = None
        self.model_type = None
        self.nodes_from = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @abstractmethod
    def _model_to_template(self, node: Node, model_id: int, nodes_from: list):
        """
        Preprocessing for local fields
        :param node: current node
        :param model_id: model id in chain
        :param nodes_from: parents model's id
        """

    @abstractmethod
    def import_json(self, model_object: dict):
        """
        Parse JSON like object and fill local fields
        :param model_object: JSON like object to parse
        """

    @abstractmethod
    def convert_to_dict(self) -> dict:
        """
        Transform all object's parameters to dictionary.
        :params path: string path to save.
        :return dict: dictionary with object parameters.
        """

    def _validate_json_model_template(self, model_object: dict, required_fields: list):
        """
        Check whether there are fields in the dictionary.
        :params model_object: dictionary to check
        :params required_fields: list of fields name
        """

        for field in required_fields:
            if field not in model_object:
                message = f"Required field '{field}' is expected, but not found."
                self.log.error(message)
                raise RuntimeError(message)


class ModelTemplate(ModelTemplateAbstract):
    def __init__(self, node: Node = None, model_id: int = None,
                 nodes_from: list = None):
        super().__init__()
        self.model_name = None
        self.custom_params = None
        self.params = None
        self.fitted_model = None
        self.fitted_model_path = None
        self.preprocessor = None

        if node:
            self._model_to_template(node, model_id, nodes_from)

    def _model_to_template(self, node: Node, model_id: int, nodes_from: list):
        self.model_id = model_id
        self.model_type = node.model.model_type
        self.custom_params = node.model.params
        self.params = self._create_full_params(node)
        self.nodes_from = nodes_from

        if _is_node_fitted(node) and not _is_node_not_cached(node):
            self.model_name = _extract_model_name(node)
            self._extract_fields_of_fitted_model(node)

    def _create_full_params(self, node: Node) -> dict:
        params = {}
        if _is_node_fitted(node) and not _is_node_not_cached(node):
            params = extract_model_params(node)
            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    params[key] = value

        return params

    def _extract_fields_of_fitted_model(self, node: Node):
        model_name = f'model_{str(self.model_id)}.pkl'
        self.fitted_model_path = os.path.join('fitted_models', model_name)
        self.preprocessor = _extract_preprocessing_strategy(node)
        self.fitted_model = node.cache.actual_cached_state.model

    def convert_to_dict(self) -> dict:
        preprocessor_strategy = preprocessing_strategy_label_by_class(self.preprocessor)

        model_object = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "custom_params": self.custom_params,
            "params": self.params,
            "nodes_from": self.nodes_from,
            "fitted_model_path": self.fitted_model_path,
            "preprocessor": preprocessor_strategy
        }

        return model_object

    def export_model(self, path: str):
        _check_existing_path(path)

        if self.fitted_model:
            path_fitted_models = os.path.join(path, 'fitted_models')
            _check_existing_path(path_fitted_models)
            joblib.dump(self.fitted_model, os.path.join(path, self.fitted_model_path))

    def import_json(self, model_object: dict):
        required_fields = ['model_id', 'model_type', 'params', 'nodes_from', 'preprocessor']
        self._validate_json_model_template(model_object, required_fields)

        self.model_id = model_object['model_id']
        self.model_type = model_object['model_type']
        self.params = model_object['params']
        self.nodes_from = model_object['nodes_from']
        if "fitted_model_path" in model_object:
            self.fitted_model_path = model_object['fitted_model_path']
        if "custom_params" in model_object:
            self.custom_params = model_object['custom_params']
        if "model_name" in model_object:
            self.model_name = model_object['model_name']
        if "preprocessor" in model_object:
            preprocessor_strategy = preprocessing_strategy_class_by_label(model_object['preprocessor'])
            if preprocessor_strategy:
                self.preprocessor = preprocessor_strategy()


def _check_existing_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_model_params(node: Node):
    return node.cache.actual_cached_state.model.get_params()


def _extract_model_name(node: Node):
    return node.cache.actual_cached_state.model.__class__.__name__


def _is_node_fitted(node: Node) -> bool:
    return bool(node.cache.actual_cached_state)


def _is_node_not_cached(node: Node) -> bool:
    return bool(node.model.model_type in ['direct_data_model', 'trend_data_model', 'residual_data_model'])


def _extract_preprocessing_strategy(node: Node) -> str:
    return node.cache.actual_cached_state.preprocessor
