import itertools
import uuid
from copy import copy
from dataclasses import dataclass
from typing import List

import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import CachedState, FittedModelCache, Node, PrimaryNode, SecondaryNode
from fedot.core.data.preprocessing import Normalization
from fedot.core.models.model import InputData, Model
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.data import classification_dataset as synthetic_dataset, \
    gauss_quantiles_dataset as gauss_quantiles


@dataclass
class ModelTemplate:
    def __init__(self, model_type):
        self.id = str(uuid.uuid4())
        self.model_type = model_type
        self.input_shape = []
        self.output_shape = []
        self.parents = []
        self.model_instance = None
        self.fitted_model = None
        self.data_fit = None
        self.preprocessor = None

    def __eq__(self, other):
        return self.id == other.id


def chain_template_random(model_types, depth, models_per_level,
                          samples, features):
    models_by_level = []

    for level in range(depth - 1):
        selected_models = np.random.choice(model_types, models_per_level[level])
        templates = [ModelTemplate(model_type=model_type) for model_type in selected_models]
        models_by_level.append(templates)

    root_model = np.random.choice(model_types)
    models_by_level.append([ModelTemplate(model_type=root_model)])

    models_by_level = with_random_links(models_by_level)

    features_shape = [samples, features]
    # TODO: change target_shape later if non-classification problems will be used
    target_shape = [samples, 1]
    models_by_level = with_calculated_shapes(models_by_level,
                                             features_shape=features_shape,
                                             target_shape=target_shape)
    return models_by_level


def chain_template_balanced_tree(model_types, depth, models_per_level,
                                 samples, features):
    models_by_level = []

    for level in range(depth - 1):
        selected_models = np.random.choice(model_types, models_per_level[level])
        templates = [ModelTemplate(model_type=model_type) for model_type in selected_models]
        models_by_level.append(templates)

    assert models_per_level[-1] == 1
    root_model = np.random.choice(model_types)
    models_by_level.append([ModelTemplate(model_type=root_model)])

    models_by_level = with_balanced_tree_links(models_by_level)

    features_shape = [samples, features]
    # TODO: change target_shape later if non-classification problems will be used
    target_shape = [samples, 1]
    models_by_level = with_calculated_shapes(models_by_level,
                                             features_shape=features_shape,
                                             target_shape=target_shape)
    return models_by_level


# TODO: fix internal primary nodes appearance
def with_random_links(models_by_level):
    for current_lvl in range(len(models_by_level) - 1):
        next_lvl = current_lvl + 1
        models_on_current_lvl = models_by_level[current_lvl]

        for model in models_on_current_lvl:
            models_on_next_lvl = models_by_level[next_lvl]
            links_amount = np.random.randint(1, len(models_on_next_lvl) + 1)
            models_to_link = np.random.choice(models_by_level[next_lvl], links_amount, replace=False)
            for model_ in models_to_link:
                model_.parents.append(model)
    return models_by_level


def with_balanced_tree_links(models_by_level):
    for current_lvl in range(len(models_by_level) - 1):
        next_lvl = current_lvl + 1
        models_on_lvl = models_by_level[current_lvl]

        current_level_amount = len(models_by_level[current_lvl])
        next_level_amount = len(models_by_level[next_lvl])
        assert current_level_amount >= next_level_amount

        models_per_group = current_level_amount // next_level_amount

        current_group_idx = 0
        for model_idx in range(current_level_amount):
            if (model_idx % models_per_group) == 0 and model_idx != 0 and current_group_idx < next_level_amount - 1:
                current_group_idx += 1
            current_model = models_on_lvl[model_idx]
            current_group_parent = models_by_level[next_lvl][current_group_idx]

            current_group_parent.parents.append(current_model)

    return models_by_level


def with_calculated_shapes(models_by_level, features_shape, target_shape):
    samples_idx, classes_idx = 0, 1
    samples, classes = features_shape[samples_idx], target_shape[classes_idx]

    # Fill the first level
    for model in models_by_level[0]:
        model.input_shape = features_shape
        model.output_shape = target_shape

    for level in range(1, len(models_by_level)):
        for model in models_by_level[level]:
            if len(model.parents) == 0:
                input_features = features_shape[classes_idx]
            else:
                input_features = sum([parent.output_shape[classes_idx] for parent in model.parents])

            model.input_shape = [samples, input_features]

            model.output_shape = target_shape

    return models_by_level


def show_chain_template(models_by_level):
    for level in range(len(models_by_level)):
        print(f'Level = {level}')
        for model in models_by_level[level]:
            print(f'{model.model_type}, input = {model.input_shape}, output = {model.output_shape}')


# TODO: refactor skip_fit logic
def fit_template(chain_template, classes, with_gaussian=False, skip_fit=False):
    templates_by_models = []
    for model_template in itertools.chain.from_iterable(chain_template):
        model_instance = Model(model_type=model_template.model_type)
        model_template.model_instance = model_instance
        templates_by_models.append((model_template, model_instance))
    if skip_fit:
        return

    for template, instance in templates_by_models:
        samples, features_amount = template.input_shape

        if with_gaussian:
            features, target = gauss_quantiles(samples_amount=samples,
                                               features_amount=features_amount,
                                               classes_amount=classes)
        else:
            options = {
                'informative': features_amount,
                'redundant': 0,
                'repeated': 0,
                'clusters_per_class': 1
            }
            features, target = synthetic_dataset(samples_amount=samples,
                                                 features_amount=features_amount,
                                                 classes_amount=classes,
                                                 features_options=options)
        target = np.expand_dims(target, axis=1)
        data_train = InputData(idx=np.arange(0, samples),
                               features=features, target=target,
                               data_type=DataTypesEnum.table,
                               task=Task(TaskTypesEnum.classification))

        preproc_data = copy(data_train)
        preprocessor = Normalization().fit(preproc_data.features)
        preproc_data.features = preprocessor.apply(preproc_data.features)
        print(f'Fit {instance}')
        fitted_model, predictions = instance.fit(data=preproc_data)

        template.fitted_model = fitted_model
        template.data_fit = preproc_data
        template.preprocessor = preprocessor


def real_chain(chain_template, with_cache=True):
    nodes_by_templates = []
    for level in range(0, len(chain_template)):
        for template in chain_template[level]:
            if len(template.parents) == 0:
                node = PrimaryNode(model_type=template.model_type)
            else:
                node = SecondaryNode(nodes_from=real_parents(nodes_by_templates,
                                                             template),
                                     model_type=template.model_type)
            node.model = template.model_instance
            if with_cache:
                cache = FittedModelCache(related_node=node)
                cache.append(CachedState(preprocessor=template.preprocessor, model=template.fitted_model))
                node.cache = cache
            nodes_by_templates.append((node, template))

    chain = Chain()
    for node, _ in nodes_by_templates:
        chain.add_node(node)

    return chain


def real_parents(nodes_by_templates, template_child) -> List[Node]:
    parent_nodes = []
    for parent in template_child.parents:
        for node, template in nodes_by_templates:
            if template == parent:
                parent_nodes.append(node)
                break
    return parent_nodes
