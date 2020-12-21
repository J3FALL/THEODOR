import datetime
import os
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.composer import ComposerRequirements
from fedot.core.composer.gp_composer.fixed_structure_composer import FixedStructureComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.composer.random_composer import RandomSearchComposer
from fedot.core.data.data import InputData
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.chains.test_chain_tuning import get_class_chain


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = '../data/advanced_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_random_composer(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    available_model_types, _ = ModelTypesRepository().suitable_model(
        task_type=TaskTypesEnum.classification)

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    random_composer = RandomSearchComposer(iter_num=1)
    req = ComposerRequirements(primary=available_model_types, secondary=available_model_types)
    chain_random_composed = random_composer.compose_chain(data=dataset_to_compose,
                                                          initial_chain=None,
                                                          composer_requirements=req,
                                                          metrics=metric_function)
    chain_random_composed.fit_from_scratch(input_data=dataset_to_compose)

    predicted_random_composed = chain_random_composed.predict(dataset_to_validate)

    roc_on_valid_random_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_random_composed.predict)

    assert roc_on_valid_random_composed > 0.6


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_fixed_structure_composer(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data

    available_model_types = ['logit', 'lda', 'knn']

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    req = GPComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                 pop_size=2, num_of_generations=1,
                                 crossover_prob=0.4, mutation_prob=0.5,
                                 add_single_model_chains=False)

    reference_chain = get_class_chain()
    builder = FixedStructureComposerBuilder(task=Task(TaskTypesEnum.classification)).with_initial_chain(
        reference_chain).with_metrics(metric_function).with_requirements(req)
    composer = builder.build()

    chain_composed = composer.compose_chain(data=dataset_to_compose)
    chain_composed.fit_from_scratch(input_data=dataset_to_compose)

    predicted_random_composed = chain_composed.predict(dataset_to_validate)

    roc_on_valid_random_composed = roc_auc(y_true=dataset_to_validate.target,
                                           y_score=predicted_random_composed.predict)

    assert roc_on_valid_random_composed > 0.6
    assert chain_composed.depth == reference_chain.depth
    assert chain_composed.length == reference_chain.length


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_gp_composer_build_chain_correct(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    task = Task(TaskTypesEnum.classification)
    available_model_types, _ = ModelTypesRepository().suitable_model(
        task_type=task.task_type)

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    req = GPComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                 max_arity=2, max_depth=2, pop_size=2, num_of_generations=1,
                                 crossover_prob=0.4, mutation_prob=0.5)

    builder = GPComposerBuilder(task).with_requirements(req).with_metrics(metric_function)
    gp_composer = builder.build()
    chain_gp_composed = gp_composer.compose_chain(data=dataset_to_compose)

    chain_gp_composed.fit_from_scratch(input_data=dataset_to_compose)
    predicted_gp_composed = chain_gp_composed.predict(dataset_to_validate)

    roc_on_valid_gp_composed = roc_auc(y_true=dataset_to_validate.target,
                                       y_score=predicted_gp_composed.predict)

    assert roc_on_valid_gp_composed > 0.6


def baseline_chain():
    chain = Chain()
    last_node = SecondaryNode(model_type='xgboost',
                              nodes_from=[])
    for requirement_model in ['knn', 'logit']:
        new_node = PrimaryNode(requirement_model)
        chain.add_node(new_node)
        last_node.nodes_from.append(new_node)
    chain.add_node(last_node)

    return chain


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_composition_time(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    task = Task(TaskTypesEnum.classification)
    models_impl = ['mlp', 'knn']
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    req_terminated_evolution = GPComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        pop_size=2, num_of_generations=5, crossover_prob=0.9,
        mutation_prob=0.9, max_lead_time=datetime.timedelta(minutes=0.000001))

    builder = GPComposerBuilder(task).with_requirements(req_terminated_evolution).with_metrics(metric_function)

    gp_composer_terminated_evolution = builder.build()

    _ = gp_composer_terminated_evolution.compose_chain(data=data)

    req_completed_evolution = GPComposerRequirements(
        primary=models_impl,
        secondary=models_impl, max_arity=2,
        max_depth=2,
        pop_size=2, num_of_generations=2, crossover_prob=0.4,
        mutation_prob=0.5)

    builder = GPComposerBuilder(task).with_requirements(req_completed_evolution).with_metrics(metric_function)
    gp_composer_completed_evolution = builder.build()

    _ = gp_composer_completed_evolution.compose_chain(data=data)

    assert len(gp_composer_terminated_evolution.history) == len(gp_composer_completed_evolution.history)


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_parameter_free_composer_build_chain_correct(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_compose = data
    dataset_to_validate = data
    available_model_types, _ = ModelTypesRepository().suitable_model(
        task_type=TaskTypesEnum.classification)

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    req = GPComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                 max_arity=2, max_depth=2, pop_size=2, num_of_generations=1,
                                 crossover_prob=0.4, mutation_prob=0.5)
    opt_params = GPChainOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free)
    builder = GPComposerBuilder(task=Task(TaskTypesEnum.classification)).with_requirements(req).with_metrics(
        metric_function).with_optimiser_parameters(opt_params)
    gp_composer = builder.build()
    chain_gp_composed = gp_composer.compose_chain(data=dataset_to_compose)

    chain_gp_composed.fit_from_scratch(input_data=dataset_to_compose)
    predicted_gp_composed = chain_gp_composed.predict(dataset_to_validate)

    roc_on_valid_gp_composed = roc_auc(y_true=dataset_to_validate.target,
                                       y_score=predicted_gp_composed.predict)

    assert roc_on_valid_gp_composed > 0.6
