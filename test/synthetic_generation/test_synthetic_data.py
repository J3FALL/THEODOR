import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.utilities.synthetic.data import (
    classification_dataset, gauss_quantiles_dataset)
from fedot.utilities.synthetic.data_benchmark import synthetic_benchmark_dataset


def default_dataset_params():
    dataset_params = {
        'samples': 1000,
        'features': 10,
        'classes': 2,
        'features_options': {
            'informative': 8,
            'redundant': 1,
            'repeated': 1,
            'clusters_per_class': 1
        }
    }

    return dataset_params


def test_classification_dataset_correct():
    params = default_dataset_params()
    features, target = classification_dataset(samples_amount=params['samples'],
                                              features_amount=params['features'],
                                              classes_amount=params['classes'],
                                              features_options=params['features_options'])

    assert features.shape == (params['samples'], params['features'])
    assert target.shape == (params['samples'],)

    actual_classes = np.unique(target)
    assert len(actual_classes) == params['classes']


def test_gauss_quantiles_dataset_correct():
    params = default_dataset_params()
    features, target = gauss_quantiles_dataset(params['samples'],
                                               features_amount=params['features'],
                                               classes_amount=params['classes'])

    assert features.shape == (params['samples'], params['features'])
    assert target.shape == (params['samples'],)

    actual_classes = np.unique(target)
    assert len(actual_classes) == params['classes']


@pytest.mark.skip('Fix preprocessing issues')
def test_synthetic_benchmark_dataset_correct():
    samples, features = 5000, 10
    benchmark_data = synthetic_benchmark_dataset(samples_amount=samples,
                                                 features_amount=features)

    assert isinstance(benchmark_data, InputData)
