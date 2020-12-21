from typing import Dict

import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.chain import separately_fit_chain
from fedot.utilities.synthetic.data import classification_dataset

DEFAULT_OPTIONS = {
    'informative': 10,
    'redundant': 0,
    'repeated': 0,
    'clusters_per_class': 1
}


def synthetic_benchmark_dataset(samples_amount: int, features_amount: int,
                                classes_amount: int = 2,
                                features_options: Dict = DEFAULT_OPTIONS,
                                fitted_chain: Chain = None) -> InputData:
    """
    Generates a binary classification benchmark dataset that was obtained using
    the (TODO: add. reference) proposed fitting schema.
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param features_options: features options in key-value suitable for classification_dataset.
    :param fitted_chain: Chain with separately fitted models.
    If None then 3-level balanced tree were fitted and taken as a default.
    :return: Benchmark dataset that is ready to be used by Chain.
    """
    if fitted_chain is None:
        fitted_chain = _default_chain(samples_amount=samples_amount,
                                      features_amount=features_amount,
                                      classes_amount=classes_amount)

    if classes_amount != 2:
        raise NotImplementedError('Only binary classification tasks are supported')

    features, target = classification_dataset(samples_amount=samples_amount,
                                              features_amount=features_amount,
                                              classes_amount=classes_amount,
                                              features_options=features_options)
    target = np.expand_dims(target, axis=1)

    task = Task(TaskTypesEnum.classification)
    samples_idxs = np.arange(0, samples_amount)

    train = InputData(idx=samples_idxs,
                      features=features, target=target, task=task,
                      data_type=DataTypesEnum.table)

    synth_target = fitted_chain.predict(input_data=train).predict
    synth_labels = _to_labels(synth_target)
    data_synth_train = InputData(idx=np.arange(0, samples_amount),
                                 features=features, target=synth_labels, task=task,
                                 data_type=DataTypesEnum.table)

    # TODO: fix preproc issues

    fitted_chain.fit_from_scratch(input_data=data_synth_train)

    features, target = classification_dataset(samples_amount=samples_amount,
                                              features_amount=features_amount,
                                              classes_amount=classes_amount,
                                              features_options=features_options)
    target = np.expand_dims(target, axis=1)
    test = InputData(idx=samples_idxs,
                     features=features, target=target,
                     data_type=DataTypesEnum.table, task=task)
    synth_target = fitted_chain.predict(input_data=test).predict
    synth_labels = _to_labels(synth_target)
    data_synth_final = InputData(idx=samples_idxs,
                                 features=features,
                                 data_type=DataTypesEnum.table,
                                 target=synth_labels, task=task)

    return data_synth_final


def _default_chain(samples_amount: int, features_amount: int,
                   classes_amount: int = 2):
    chain = separately_fit_chain(samples=samples_amount, features_amount=features_amount,
                                 classes=classes_amount)
    return chain


def _to_labels(predictions):
    thr = 0.5
    labels = [0 if val <= thr else 1 for val in predictions]
    labels = np.expand_dims(np.array(labels), axis=1)
    return labels
