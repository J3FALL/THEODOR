import warnings
from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.core.algorithms.time_series.lagged_features import prepare_lagged_ts_for_prediction
from fedot.core.models.preprocessing import ImputationStrategy
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@dataclass
class Data:
    """
    Base Data type class
    """
    idx: np.array
    features: np.array
    task: Task
    data_type: DataTypesEnum

    @staticmethod
    def from_csv(file_path=None,
                 delimiter=',',
                 task: Task = Task(TaskTypesEnum.classification),
                 data_type: DataTypesEnum = DataTypesEnum.table,
                 columns_to_drop: Optional[List] = None,
                 target_column: Optional[str] = ''):
        """
        :param file_path: the path to the CSV with data
        :param columns_to_drop: the names of columns that should be dropped
        :param delimiter: the delimiter to separate the columns
        :param task: the task that should be solved with data
        :param data_type: the type of data interpretation
        :param target_column: name of target column (last column if empty and no target if None)
        :return:
        """

        data_frame = pd.read_csv(file_path, sep=delimiter)
        if columns_to_drop:
            data_frame = data_frame.drop(columns_to_drop, axis=1)
        data_frame = _convert_dtypes(data_frame=data_frame)
        data_array = np.array(data_frame).T
        idx = data_array[0]

        if target_column == '':
            target_column = data_frame.columns[-1]

        if target_column:
            target = np.array(data_frame[target_column]).astype(np.float)
            pos = list(data_frame.keys()).index(target_column)
            features = np.delete(data_array.T, [0, pos], axis=1)
        else:
            # no target in data
            features = data_array[1:].T
            target = None

        features = ImputationStrategy().fit(features).apply(features)

        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)


@dataclass
class InputData(Data):
    """
    Data class for input data for the nodes
    """
    target: Optional[np.array] = None

    @property
    def num_classes(self) -> Optional[int]:
        if self.task.task_type == TaskTypesEnum.classification:
            return len(np.unique(self.target))
        else:
            return None

    @staticmethod
    def from_predictions(outputs: List['OutputData'], target: np.array):
        if len(set([output.task.task_type for output in outputs])) > 1:
            raise ValueError('Inconsistent task types')

        task = outputs[0].task
        data_type = outputs[0].data_type
        idx = outputs[0].idx

        dataset_merging_funcs = {
            DataTypesEnum.forecasted_ts: _combine_datasets_ts,
            DataTypesEnum.ts: _combine_datasets_ts,
            DataTypesEnum.table: _combine_datasets_table
        }
        dataset_merging_funcs.setdefault(data_type, _combine_datasets_common)

        features = dataset_merging_funcs[data_type](outputs)

        return InputData(idx=idx, features=features, target=target, task=task,
                         data_type=data_type)

    def subset(self, start: int, end: int):
        if not (0 <= start <= end <= len(self.idx)):
            raise ValueError('Incorrect boundaries for subset')
        new_features = None
        if self.features is not None:
            new_features = self.features[start:end + 1]
        return InputData(idx=self.idx[start:end + 1], features=new_features,
                         target=self.target[start:end + 1], task=self.task, data_type=self.data_type)

    def prepare_for_modelling(self, is_for_fit: bool = False):
        prepared_data = self
        if (self.data_type == DataTypesEnum.ts_lagged_table or
                self.data_type == DataTypesEnum.forecasted_ts):
            prepared_data = prepare_lagged_ts_for_prediction(self, is_for_fit)
        elif self.data_type in [DataTypesEnum.table, DataTypesEnum.forecasted_ts]:
            # TODO implement NaN filling here
            pass

        return prepared_data


@dataclass
class OutputData(Data):
    """
    Data type for data predicted in the node
    """
    predict: np.array = None


def split_train_test(data, split_ratio=0.8, with_shuffle=False, task: Task = None):
    assert 0. <= split_ratio <= 1.
    if task is not None and task.task_type == TaskTypesEnum.ts_forecasting:
        split_point = int(len(data) * split_ratio)
        # move pre-history of time series from train to test sample
        data_train, data_test = (data[:split_point],
                                 copy(data[split_point - task.task_params.max_window_size:]))
    else:
        if with_shuffle:
            data_train, data_test = train_test_split(data, test_size=1. - split_ratio, random_state=42)
        else:
            split_point = int(len(data) * split_ratio)
            data_train, data_test = data[:split_point], data[split_point:]
    return data_train, data_test


def _convert_dtypes(data_frame: pd.DataFrame):
    objects: pd.DataFrame = data_frame.select_dtypes('object')
    for column_name in objects:
        warnings.warn(f'Automatic factorization for the column {column_name} with type "object" is applied.')
        encoded = pd.factorize(data_frame[column_name])[0]
        data_frame[column_name] = encoded
    data_frame = data_frame.fillna(0)
    return data_frame


def train_test_data_setup(data: InputData, split_ratio=0.8,
                          shuffle_flag=False, task: Task = None) -> Tuple[InputData, InputData]:
    if data.features is not None:
        train_data_x, test_data_x = split_train_test(data.features, split_ratio, with_shuffle=shuffle_flag, task=task)
    else:
        train_data_x, test_data_x = None, None

    train_data_y, test_data_y = split_train_test(data.target, split_ratio, with_shuffle=shuffle_flag, task=task)
    train_idx, test_idx = split_train_test(data.idx, split_ratio, with_shuffle=shuffle_flag, task=task)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=train_idx, task=data.task, data_type=data.data_type)
    test_data = InputData(features=test_data_x, target=test_data_y, idx=test_idx, task=data.task,
                          data_type=data.data_type)
    return train_data, test_data


def _combine_datasets_ts(outputs: List[OutputData]):
    features_list = list()

    expected_len = max([len(output.predict) for output in outputs])

    for elem in outputs:
        predict = elem.predict
        if len(elem.predict) != expected_len:
            raise ValueError(f'Non-equal prediction length: {len(elem.predict)} and {expected_len}')
        features_list.append(predict)

    if len(features_list) > 1:
        features = np.column_stack(features_list)
    else:
        features = features_list[0]

    return features


def _combine_datasets_table(outputs: List[OutputData]):
    features = list()
    expected_len = len(outputs[0].predict)

    for elem in outputs:
        if len(elem.predict) != expected_len:
            raise ValueError(f'Non-equal prediction length: {len(elem.predict)} and {expected_len}')
        if len(elem.predict.shape) == 1:
            features.append(elem.predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                features.append(elem.predict[:, i])

    features = np.array(features).T

    return features


def _combine_datasets_common(outputs: List[OutputData]):
    features = list()

    for elem in outputs:
        if len(elem.predict) != len(outputs[0].predict):
            raise NotImplementedError(f'Non-equal prediction length: '
                                      f'{len(elem.predict)} and {len(outputs[0].predict)}')
        if len(elem.predict.shape) == 1:
            features.append(elem.predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                features.append(elem.predict[:, i])
    return features
