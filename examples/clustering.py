import random
from copy import deepcopy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def create_clustering_example():
    num_features = 2
    num_clusters = 3

    predictors, response = make_blobs(1000, num_features, centers=num_clusters, random_state=3)

    plt.scatter(predictors[:, 0], predictors[:, 1],
                c=[list(mcolors.TABLEAU_COLORS)[_] for _ in response])
    plt.show()

    return InputData(features=predictors, target=response, idx=np.arange(0, len(predictors)),
                     task=Task(TaskTypesEnum.clustering),
                     data_type=DataTypesEnum.table)


def get_atomic_clustering_model(train_data: InputData):
    chain = Chain(PrimaryNode('kmeans'))
    chain.fit(input_data=train_data)

    return chain


def validate_model_quality(model: Chain, dataset_to_validate: InputData):
    predicted_labels = model.predict(dataset_to_validate).predict

    prediction_valid = round(adjusted_rand_score(labels_true=dataset_to_validate.target,
                                                 labels_pred=predicted_labels), 6)

    return prediction_valid, predicted_labels


if __name__ == '__main__':
    data = create_clustering_example()
    data_train = deepcopy(data)
    data_train.target = None

    fitted_model = get_atomic_clustering_model(data_train)
    prediction_basic, _ = validate_model_quality(fitted_model, data)

    fitted_model.fine_tune_all_nodes(data_train, iterations=30)

    prediction_tuned, predicted_labels = validate_model_quality(fitted_model, data)

    print(f'adjusted_rand_score for basic model {prediction_basic}')
    print(f'adjusted_rand_score for tuned model {prediction_tuned}')

    print(f'Real clusters number is {len(set(data.target))}, '
          f'predicted number is {len(set(predicted_labels))}')
