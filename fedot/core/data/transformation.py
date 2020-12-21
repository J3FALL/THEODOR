from copy import copy
from typing import List

import numpy as np
import pandas as pd

from fedot.core.data.data import (
    InputData,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import extract_task_param


def ts_to_lagged_table(input_data: InputData) -> InputData:
    window_len, prediction_len = extract_task_param(input_data.task)

    transformed_data = copy(input_data)

    df = pd.DataFrame()
    df_target = None
    transformed_data.features = None

    if input_data.target is not None:
        target = list(input_data.target)
        target.append(np.nan)
        df = pd.DataFrame(data={'ts_target': target})
        df_target = pd.DataFrame(data={'ts_target': target})
        for lag in range(1, window_len + 1):
            df[f'lag_{lag}'] = df.ts_target.shift(lag)
        for lag in range(0, prediction_len):
            df_target[f'new_target_{lag}'] = df.ts_target.shift(-lag)

    if input_data.features is not None and (input_data.features.shape != input_data.target.shape or
                                            not np.allclose(input_data.features,
                                                            input_data.target,
                                                            equal_nan=True)):
        if len(input_data.features.shape) == 1:
            input_data.features.shape = input_data.features.shape + (1,)

        # iterate feature array by columns (inidvidual features)
        for feature_id, feature in enumerate(input_data.features.T):
            feature = list(feature)
            feature.append(np.nan)
            df_exog = pd.DataFrame(data={f'exog_feature': feature})
            for lag in range(1, window_len + 1):
                df[f'lag_f{feature_id}_{lag}'] = df_exog.exog_feature.shift(lag)

    df = df.drop('ts_target', axis=1)
    df_target = df_target.drop('ts_target', axis=1)

    transformed_data.features = np.squeeze(df)
    transformed_data.target = df_target

    size_diff = transformed_data.features.shape[0] - len(transformed_data.idx)
    if size_diff:
        new_idx = list(transformed_data.idx)
        new_idx.extend(['new'] * size_diff)
        transformed_data.idx = np.asarray(new_idx)

    transformed_data.data_type = DataTypesEnum.ts_lagged_table
    return transformed_data


def direct(input_data: InputData) -> InputData:
    return copy(input_data)


# from datatype / to datatype : function
_transformation_functions_for_data_types = {
    (DataTypesEnum.ts, DataTypesEnum.ts_lagged_table):
        ts_to_lagged_table,
    (DataTypesEnum.forecasted_ts, DataTypesEnum.table): direct,
    (DataTypesEnum.table, DataTypesEnum.ts): direct
}


def transformation_function_for_data(input_data_type: DataTypesEnum,
                                     required_data_types: List[DataTypesEnum]):
    if input_data_type in required_data_types:
        return direct

    transformation = None
    for required_data_type in required_data_types:
        transformation = _transformation_functions_for_data_types.get(
            (input_data_type, required_data_type), None)
        if transformation:
            break

    if not transformation:
        raise ValueError(
            f'The {input_data_type} cannot be converted to {required_data_types}')
    return transformation
