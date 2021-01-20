import datetime
import random

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc
from cases.credit_scoring_problem import get_scoring_data

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposerBuilder
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.composer.optimisers.selection import SelectionTypesEnum
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ComplexityMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(12)
np.random.seed(12)


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_credit_scoring_problem(train_file_path, test_file_path,
                               max_lead_time: datetime.timedelta = datetime.timedelta(minutes=120),
                               is_visualise=False):
    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    # the choice of the metric for the chain quality assessment during composition
    quality_metric = ClassificationMetricsEnum.ROCAUC
    complexity_metric = ComplexityMetricsEnum.node_num
    metrics = [quality_metric, complexity_metric]
    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=20, num_of_generations=8,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time, model_fit_time_constraint=900,
        start_depth=2)

    # GP optimiser parameters choice
    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type,
                                                      selection_types=[SelectionTypesEnum.nsga2])

    # Create builder for composer and set composer params
    builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
        metrics).with_optimiser_parameters(optimiser_parameters)

    # Create GP-based composer
    composer = builder.build()

    # the optimal chain generation by composition - the most time-consuming task
    chains_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                 is_visualise=True)

    chains_roc_auc = []
    for chain_evo_composed in chains_evo_composed:

        if is_visualise:
            ComposerVisualiser.visualise(chain_evo_composed)

        chain_evo_composed.fine_tune_primary_nodes(input_data=dataset_to_compose,
                                                   iterations=50)

        chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

        # the quality assessment for the obtained composite models
        roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed,
                                                                dataset_to_validate)

        chains_roc_auc.append(roc_on_valid_evo_composed)
        print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')

    return max(chains_roc_auc)


if __name__ == '__main__':
    full_path_train, full_path_test = get_scoring_data()
    run_credit_scoring_problem(full_path_train, full_path_test, is_visualise=True)
