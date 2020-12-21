import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from fedot.core.repository.dataset_types import DataTypesEnum


class PreprocessingStrategy:
    def fit(self, data_to_fit) -> 'PreprocessingStrategy':
        raise NotImplementedError()

    def apply(self, data):
        raise NotImplementedError()


class Scaling(PreprocessingStrategy):
    def __init__(self, with_imputation=True):
        if with_imputation:
            self.default = ImputationStrategy()
        self.with_imputation = with_imputation
        self.scaler = preprocessing.StandardScaler()

    def fit(self, data_to_fit):
        if self.with_imputation:
            self.default.fit(data_to_fit)
            data_to_fit = self.default.apply(data_to_fit)

        data_to_fit = _expand_data(data_to_fit)
        self.scaler.fit(data_to_fit)
        return self

    def apply(self, data):
        if self.with_imputation:
            data = self.default.apply(data)

        data = _expand_data(data)
        resulted = self.scaler.transform(data)
        return resulted


class Normalization(PreprocessingStrategy):
    def __init__(self):
        self.default = ImputationStrategy()

    def fit(self, data_to_fit):
        self.default.fit(data_to_fit)
        return self

    def apply(self, data):
        modified = self.default.apply(data)
        resulted = preprocessing.normalize(modified)

        return resulted


class ImputationStrategy(PreprocessingStrategy):
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def fit(self, data_to_fit):
        self.imputer.fit(data_to_fit)
        return self

    def apply(self, data):
        modified = self.imputer.transform(data)
        return modified


class EmptyStrategy(PreprocessingStrategy):
    def fit(self, data_to_fit):
        return self

    def apply(self, data):
        result = np.asarray(data)
        if len(result.shape) == 1:
            result = np.expand_dims(result, axis=1)
        return result


class TsScalingStrategy(Scaling):
    def __init__(self):
        # the NaN preservation is important for the lagged ts features and forecasted ts
        super().__init__(with_imputation=False)


_preprocessing_for_input_data = {
    DataTypesEnum.ts: EmptyStrategy,
    DataTypesEnum.table: Scaling,
    DataTypesEnum.ts_lagged_table: TsScalingStrategy,
    DataTypesEnum.forecasted_ts: TsScalingStrategy,
}


def preprocessing_func_for_data(data: 'InputData', node: 'Node'):
    preprocessing_func = EmptyStrategy
    if 'without_preprocessing' not in node.model.metadata.tags:
        if node.manual_preprocessing_func:
            preprocessing_func = node.manual_preprocessing_func
        else:
            preprocessing_func = _preprocessing_for_input_data[data.data_type]
    return preprocessing_func


def _expand_data(data):
    if len(data.shape) == 1:
        data = data[:, None]
    return data
