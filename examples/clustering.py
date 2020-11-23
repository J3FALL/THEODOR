import datetime
import random
from copy import deepcopy
from datetime import timedelta

import numpy as np
from sklearn.metrics import precision_score as precision

from examples.utils import create_clustering_examples_from_iris
from fedot.core.composer.chain import Chain
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposer, GPComposerRequirements
from fedot.core.composer.node import PrimaryNode
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.models.data import InputData
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClusteringMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def get_composite_clustering_model(train_data: InputData,
                                   cur_lead_time: datetime.timedelta = timedelta(seconds=5)):
    task = Task(task_type=TaskTypesEnum.clustering)
    dataset_to_compose = train_data

    # the search of the models provided by the framework
    # that can be used as nodes in a chain for the selected task
    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.suitable_model(task_type=task.task_type,
                                                          forbidden_tags=['ensembler'])
    secondary_model_types, _ = models_repo.models_with_tag(['ensembler'])

    # TODO change to clustering metric
    metric_function = MetricsRepository(). \
        metric_by_id(ClusteringMetricsEnum.silhouette)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types, secondary=secondary_model_types,
        max_lead_time=cur_lead_time, max_arity=len(available_model_types), max_depth=1)

    # run the search of best suitable model
    chain_evo_composed = GPComposer().compose_chain(data=dataset_to_compose,
                                                    initial_chain=None,
                                                    composer_requirements=composer_requirements,
                                                    metrics=metric_function, is_visualise=False)
    chain_evo_composed.fit(input_data=dataset_to_compose)

    return chain_evo_composed


def get_atomic_clustering_model(train_data: InputData):
    chain = Chain(PrimaryNode('kmeans'))
    chain.fit(input_data=train_data)

    return chain


def validate_model_quality(model: Chain, dataset_to_validate: InputData):
    predicted_labels = model.predict(dataset_to_validate).predict

    prec_valid = round(precision(y_true=dataset_to_validate.target,
                                 y_pred=predicted_labels, average='macro'), 6)
    return prec_valid


if __name__ == '__main__':
    data = create_clustering_examples_from_iris()
    data_train = deepcopy(data)
    data_train.target = None
    fitted_model = get_composite_clustering_model(data_train)
    fitted_model_atomic = get_atomic_clustering_model(data_train)

    ComposerVisualiser.visualise(fitted_model)

    prec_composite = validate_model_quality(fitted_model, data)
    prec_atomic = validate_model_quality(fitted_model_atomic, data)

    print(f'Precision metric for composite is {prec_composite}')
    print(f'Precision metric for atomic {prec_atomic}')
