from random import seed

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import train_test_data_setup
from fedot.utilities.synthetic.chain import separately_fit_chain
from fedot.utilities.synthetic.data import (
    classification_dataset, gauss_quantiles_dataset
)
from fedot.utilities.synthetic.data_benchmark import synthetic_benchmark_dataset


def data_generator_example():
    samples_total, features_amount = 100, 10
    classes = 2
    options = {
        'informative': 8,
        'redundant': 1,
        'repeated': 1,
        'clusters_per_class': 1
    }
    features, target = classification_dataset(samples_total, features_amount, classes,
                                              features_options=options,
                                              noise_fraction=0.1, full_shuffle=False)

    plt.subplot(121)
    plt.title('The first two informative features, one cluster per class')
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')

    features, target = gauss_quantiles_dataset(samples_total, features_amount=2, classes_amount=classes)
    plt.subplot(122)
    plt.title(f'Gaussian divided into {classes} quantiles')
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')

    plt.show()


def synthetic_benchmark_composing_example():
    fitted_chain = separately_fit_chain(samples=5000, features_amount=10,
                                        classes=2)
    data = synthetic_benchmark_dataset(samples_amount=5000, features_amount=10,
                                       fitted_chain=fitted_chain)

    print(f'Synthetic features: {data.features[:10]}')
    print(f'Synthetic target: {data.target[:10]}')

    train, test = train_test_data_setup(data)
    simple_chain = two_level_chain()
    simple_chain.fit(input_data=train, use_cache=False)

    print(f'ROC score on train: {roc_value(simple_chain, train)}')
    print(f'ROC score on test {roc_value(simple_chain, test)}')


def two_level_chain():
    first = PrimaryNode(model_type='logit')
    second = PrimaryNode(model_type='knn')
    third = SecondaryNode(model_type='xgboost',
                          nodes_from=[first, second])

    chain = Chain()
    for node in [first, second, third]:
        chain.add_node(node)

    return chain


def roc_value(chain: Chain, dataset_to_validate) -> float:
    predicted = chain.predict(dataset_to_validate)
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


if __name__ == '__main__':
    seed(15)
    np.random.seed(15)
    data_generator_example()
    synthetic_benchmark_composing_example()
