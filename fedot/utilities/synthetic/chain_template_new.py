import json
import os
from typing import List
from uuid import uuid4

import joblib

from fedot.core.chains.node import CachedState, Node, PrimaryNode, SecondaryNode
from fedot.core.utils import default_fedot_data_dir

DEFAULT_FITTED_MODELS_PATH = os.path.join(default_fedot_data_dir(), 'fitted_models')


class ChainTemplate:
    def __init__(self, chain):
        self.total_chain_types = {}
        self.depth = None
        self.model_templates = []
        self.unique_chain_id = str(uuid4())

        if chain.root_node:
            self._chain_to_template(chain)
        else:
            self.link_to_empty_chain = chain

    def _chain_to_template(self, chain):
        self._extract_chain_structure(chain.root_node, 0, [])
        self.depth = chain.depth

    def _extract_chain_structure(self, node: Node, model_id: int, visited_nodes: List[str]):
        """
        Recursively go through the Chain from 'root_node' to PrimaryNode's,
        creating a ModelTemplate with unique id for each Node. In addition,
        checking whether this Node has been visited yet.
        """
        if node.nodes_from:
            nodes_from = []
            for node_parent in node.nodes_from:
                if node_parent.descriptive_id in visited_nodes:
                    nodes_from.append(visited_nodes.index(node_parent.descriptive_id) + 1)
                else:
                    visited_nodes.append(node_parent.descriptive_id)
                    nodes_from.append(len(visited_nodes))
                    self._extract_chain_structure(node_parent, len(visited_nodes), visited_nodes)
        else:
            nodes_from = []

        model_template = ModelTemplate(node, model_id, nodes_from, self.unique_chain_id)

        self.model_templates.append(model_template)
        self._add_chain_type_to_state(model_template.model_type)

        return model_template

    def _add_chain_type_to_state(self, model_type: str):
        if model_type in self.total_chain_types:
            self.total_chain_types[model_type] += 1
        else:
            self.total_chain_types[model_type] = 1

    def export_to_json(self, path: str):
        path = self._create_unique_chain_id(path)
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)

        chain_template_dict = self.convert_to_dict()
        json_data = json.dumps(chain_template_dict)
        with open(os.path.join(absolute_path, f'{self.unique_chain_id}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(json.loads(json_data), indent=4))
            resulted_path = os.path.join(absolute_path, f'{self.unique_chain_id}.json')
            print(f"The chain saved in the path: {resulted_path}")

        return json_data

    def _create_unique_chain_id(self, path_to_save: str):
        name_of_files = path_to_save.split('/')
        last_file = name_of_files[-1].split('.')
        if len(last_file) >= 2:
            if last_file[-1] == 'json':
                if os.path.exists(path_to_save):
                    raise FileExistsError(f"File on path: '{os.path.abspath(path_to_save)}' exists")
                self.unique_chain_id = ''.join(last_file[0:-1])
                return '/'.join(name_of_files[0:-1])
            else:
                raise ValueError(f"Could not save chain in"
                                 f" '{last_file[-1]}' extension, use 'json' format")
        else:
            return path_to_save

    def convert_to_dict(self) -> dict:
        sorted_chain_types = self.total_chain_types
        json_nodes = list(map(lambda model_template: model_template.export_to_json(), self.model_templates))

        json_object = {
            "total_chain_types": sorted_chain_types,
            "depth": self.depth,
            "nodes": json_nodes,
        }

        return json_object

    def import_from_json(self, path: str):
        self._check_for_json_existence(path)

        with open(path) as json_file:
            json_object_chain = json.load(json_file)

        self._extract_models(json_object_chain)
        self.convert_to_chain(self.link_to_empty_chain)
        self.depth = self.link_to_empty_chain.depth
        self.link_to_empty_chain = None

    def _check_for_json_existence(self, path: str):
        absolute_path = os.path.abspath(path)
        path, name_of_file = os.path.split(path)
        name_of_file = name_of_file.split('.')
        if len(name_of_file) >= 2:
            if name_of_file[-1] == 'json':
                if not os.path.isfile(absolute_path):
                    raise FileNotFoundError(f"File on the path: {absolute_path} does not exist.")
                self.unique_chain_id = ''.join(name_of_file[0:-1])
            else:
                raise ValueError(f"Could not load chain in"
                                 f" '{name_of_file[-1]}' extension, use 'json' format")
        else:
            raise FileNotFoundError(f"Write path to 'json' format file")

    def _extract_models(self, chain_json):
        model_objects = chain_json['nodes']

        for model_object in model_objects:
            model_template = ModelTemplate()
            model_template.import_from_json(model_object)
            self._add_chain_type_to_state(model_template.model_type)
            self.model_templates.append(model_template)

    def convert_to_chain(self, chain_to_convert_to: 'Chain'):
        visited_nodes = {}
        root_template = [model_template for model_template in self.model_templates if model_template.model_id == 0][0]
        root_node = _roll_chain_structure(root_template, visited_nodes, self)
        chain_to_convert_to.nodes.clear()
        chain_to_convert_to.add_node(root_node)


def _roll_chain_structure(model_object: 'ModelTemplate', visited_nodes: dict, chain_object: ChainTemplate):
    if model_object.model_id in visited_nodes:
        return visited_nodes[model_object.model_id]
    if model_object.nodes_from:
        node = SecondaryNode(model_object.model_type)
    else:
        node = PrimaryNode(model_object.model_type)

    node.model.params = model_object.params
    nodes_from = [model_template for model_template in chain_object.model_templates
                  if model_template.model_id in model_object.nodes_from]
    node.nodes_from = [_roll_chain_structure(node_from, visited_nodes, chain_object) for node_from
                       in nodes_from]

    if model_object.fitted_model_path:
        path_to_model = os.path.abspath(model_object.fitted_model_path)
        if not os.path.isfile(path_to_model):
            raise FileNotFoundError(f"File on the path: {path_to_model} does not exist.")

        fitted_model = joblib.load(path_to_model)
        node.cache.append(CachedState(preprocessor=model_object.preprocessor,
                                      model=fitted_model))
    visited_nodes[model_object.model_id] = node
    return node


class ModelTemplate:
    def __init__(self, node: Node = None, model_id: int = None,
                 nodes_from: list = None, chain_id: str = None):
        self.model_id = None
        self.model_type = None
        self.model_name = None
        self.custom_params = None
        self.params = None
        self.nodes_from = None
        self.fitted_model_path = None
        self.preprocessor = None

        if node:
            self._model_to_template(node, model_id, nodes_from, chain_id)

    def _model_to_template(self, node: Node, model_id: int, nodes_from: list, chain_id: str):
        self.model_id = model_id
        self.model_type = node.model.model_type
        self.custom_params = node.model.params
        self.params = self._create_full_params(node)
        self.nodes_from = nodes_from

        if _is_node_fitted(node):
            self.model_name = _extract_model_name(node)
            self._extract_fitted_model(node, chain_id)
            self.preprocessor = _extract_preprocessing_strategy(node)

    def _extract_fitted_model(self, node: Node, chain_id: str):
        absolute_path = os.path.abspath(os.path.join(DEFAULT_FITTED_MODELS_PATH, chain_id))
        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path)
        model_name = f'model_{str(self.model_id)}.pkl'
        self.fitted_model_path = os.path.join(DEFAULT_FITTED_MODELS_PATH, chain_id, model_name)
        joblib.dump(node.cache.actual_cached_state.model, self.fitted_model_path)

    def _create_full_params(self, node: Node) -> dict:
        params = {}
        if _is_node_fitted(node):
            params = _extract_model_params(node)
            if isinstance(self.custom_params, dict):
                for key, value in self.custom_params.items():
                    params[key] = value

        return params

    def export_to_json(self) -> dict:

        model_object = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "custom_params": self.custom_params,
            "params": self.params,
            "nodes_from": self.nodes_from,
            "trained_model_path": self.fitted_model_path,
        }

        return model_object

    def import_from_json(self, model_object: dict):
        _validate_json_model_template(model_object)

        self.model_id = model_object['model_id']
        self.model_type = model_object['model_type']
        self.params = model_object['params']
        self.nodes_from = model_object['nodes_from']
        if "trained_model_path" in model_object:
            self.fitted_model_path = model_object['trained_model_path']
        if "custom_params" in model_object:
            self.custom_params = model_object['custom_params']
        if "model_name" in model_object:
            self.model_name = model_object['model_name']


def _validate_json_model_template(model_object: dict):
    required_fields = ['model_id', 'model_type', 'params', 'nodes_from']

    for field in required_fields:
        if field not in model_object:
            raise RuntimeError(f"Required field '{field}' is expected, but not found")


def _extract_model_params(node: Node):
    return node.cache.actual_cached_state.model.get_params()


def _extract_model_name(node: Node):
    return node.cache.actual_cached_state.model.__class__.__name__


def _is_node_fitted(node: Node) -> bool:
    return bool(node.cache.actual_cached_state)


def _extract_preprocessing_strategy(node: Node):
    return node.cache.actual_cached_state.preprocessor


def extract_subtree_root(root_model_id: int, chain_template: ChainTemplate):
    root_node = [model_template for model_template in chain_template.model_templates
                 if model_template.model_id == root_model_id][0]
    root_node = _roll_chain_structure(root_node, {}, chain_template)

    return root_node
