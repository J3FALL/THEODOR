import numpy as np
from scipy import interpolate

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


class SimpleGapFiller:
    """
    Base class used for filling in the gaps in time series with simple methods.
    Methods from the SimpleGapFiller class can be used for comparison with more
    complex models in class ModelGapFiller

    :param gap_value: value, which identify gap elements in array
    """

    def __init__(self, gap_value: float = -100.0):
        self.gap_value = gap_value

    def linear_interpolation(self, input_data):
        """
        Method allows to restore missing values in an array
        using linear interpolation

        :param input_data: array with gaps
        :return: array without gaps
        """

        output_data = np.array(input_data)

        # The indices of the known elements
        non_nan = np.ravel(np.argwhere(output_data != self.gap_value))
        # All known elements in the array
        masked_array = output_data[non_nan]
        f_interploate = interpolate.interp1d(non_nan, masked_array)
        x = np.arange(0, len(output_data))
        output_data = f_interploate(x)
        return output_data

    def local_poly_approximation(self, input_data, degree: int = 2,
                                 n_neighbors: int = 5):
        """
        Method allows to restore missing values in an array
        using Savitzky-Golay filter

        :param input_data: array with gaps
        :param degree: degree of a polynomial function
        :param n_neighbors: number of neighboring known elements of the time
        series that the approximation is based on
        :return: array without gaps
        """

        output_data = np.array(input_data)

        i_gaps = np.ravel(np.argwhere(output_data == self.gap_value))

        # Iterately fill in the gaps in the time series
        for gap_index in i_gaps:
            # Indexes of known elements (updated at each iteration)
            i_known = np.argwhere(output_data != self.gap_value)
            i_known = np.ravel(i_known)

            # Based on the indexes we calculate how far from the gap
            # the known values are located
            id_distances = np.abs(i_known - gap_index)

            # Now we know the indices of the smallest values in the array,
            # so sort indexes
            sorted_idx = np.argsort(id_distances)
            nearest_values = []
            nearest_indices = []
            for i in sorted_idx[:n_neighbors]:
                time_index = i_known[i]
                nearest_values.append(output_data[time_index])
                nearest_indices.append(time_index)
            nearest_values = np.array(nearest_values)
            nearest_indices = np.array(nearest_indices)

            local_coefs = np.polyfit(nearest_indices, nearest_values, degree)
            est_value = np.polyval(local_coefs, gap_index)
            output_data[gap_index] = est_value

        return output_data

    def batch_poly_approximation(self, input_data, degree: int = 3,
                                 n_neighbors: int = 10):
        """
        Method allows to restore missing values in an array using
        batch polynomial approximations.
        Approximation is applied not for individual omissions, but for
        intervals of omitted values

        :param input_data: array with gaps
        :param degree: degree of a polynomial function
        :param n_neighbors: the number of neighboring known elements of
        time series that the approximation is based on
        :return: array without gaps
        """

        output_data = np.array(input_data)

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for gap in new_gap_list:
            # Find the center point of the gap
            center_index = int((gap[0] + gap[-1]) / 2)

            # Indexes of known elements (updated at each iteration)
            i_known = np.argwhere(output_data != self.gap_value)
            i_known = np.ravel(i_known)

            # Based on the indexes we calculate how far from the gap
            # the known values are located
            id_distances = np.abs(i_known - center_index)

            # Now we know the indices of the smallest values in the array,
            # so sort indexes
            sorted_idx = np.argsort(id_distances)

            # Nearest known values to the gap
            nearest_values = []
            # And their indexes
            nearest_indices = []
            for i in sorted_idx[:n_neighbors]:
                # Getting the index value for the series - output_data
                time_index = i_known[i]
                # Using this index, we get the value of each of the "neighbors"
                nearest_values.append(output_data[time_index])
                nearest_indices.append(time_index)
            nearest_values = np.array(nearest_values)
            nearest_indices = np.array(nearest_indices)

            # Local approximation by an n-th degree polynomial
            local_coefs = np.polyfit(nearest_indices, nearest_values, degree)

            # Estimate our interval according to the selected coefficients
            est_value = np.polyval(local_coefs, gap)
            output_data[gap] = est_value

        return output_data

    def _parse_gap_ids(self, gap_list: list) -> list:
        """
        Method allows parsing source array with gaps indexes

        :param gap_list: array with indexes of gaps in array
        :return: a list with separated gaps in continuous intervals
        """

        new_gap_list = []
        local_gaps = []
        for index, gap in enumerate(gap_list):
            if index == 0:
                local_gaps.append(gap)
            else:
                prev_gap = gap_list[index - 1]
                if gap - prev_gap > 1:
                    # There is a "gap" between gaps
                    new_gap_list.append(local_gaps)

                    local_gaps = []
                    local_gaps.append(gap)
                else:
                    local_gaps.append(gap)
        new_gap_list.append(local_gaps)

        return new_gap_list


class ModelGapFiller(SimpleGapFiller):
    """
    Class used for filling in the gaps in time series

    :param gap_value: value, which mask gap elements in array
    :param chain: TsForecastingChain object for filling in the gaps
    :param max_window_size: window length
    """

    def __init__(self, gap_value, chain, max_window_size: int = 50):
        super().__init__(gap_value)
        self.chain = chain
        self.max_window_size = max_window_size

    def forward_inverse_filling(self, input_data):
        """
        Method fills in the gaps in the input array using forward and inverse
        directions of predictions

        :param input_data: data with gaps to filling in the gaps in it
        :return: array without gaps
        """

        output_data = np.array(input_data)

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for batch_index in range(len(new_gap_list)):

            preds = []
            weights = []
            # Two predictions are generated for each gap - forward and backward
            for direction_function in [self._forward, self._inverse]:

                weights_list, predicted_list = direction_function(output_data,
                                                                  batch_index,
                                                                  new_gap_list)
                weights.append(weights_list)
                preds.append(predicted_list)

            preds = np.array(preds)
            weights = np.array(weights)
            result = np.average(preds, axis=0, weights=weights)

            gap = new_gap_list[batch_index]
            # Replace gaps in an array with predicted values
            output_data[gap] = result

        return output_data

    def forward_filling(self, input_data):
        """
        Method fills in the gaps in the input array using chain with only
        forward direction (i.e. time series forecasting)

        :param input_data: data with gaps to filling in the gaps in it
        :return: array without gaps
        """

        output_data = np.array(input_data)

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for gap in new_gap_list:
            # The entire time series is used for training until the gap
            timeseries_train_part = output_data[:gap[0]]

            # Adaptive prediction interval length
            len_gap = len(gap)

            # Chain for the task of filling in gaps
            predicted = self.__chain_fit_predict(timeseries_train_part,
                                                 len_gap)

            # Replace gaps in an array with predicted values
            output_data[gap] = predicted
        return output_data

    def _forward(self, timeseries_data, batch_index, new_gap_list):
        """
        The time series method makes a forward forecast based on the part
        of the time series that is located to the left of the gap.

        :param timeseries_data: one-dimensional array of a time series
        :param batch_index: index of the interval (batch) with a gap
        :param new_gap_list: array with nested lists of gap indexes

        :return weights_list: numpy array with prediction weights for
        averaging
        :return predicted_values: numpy array with predicted values in the
        gap
        """

        gap = new_gap_list[batch_index]
        timeseries_train_part = timeseries_data[:gap[0]]

        # Adaptive prediction interval length
        len_gap = len(gap)
        predicted_values = self.__chain_fit_predict(timeseries_train_part,
                                                    len_gap)
        weights_list = np.arange(len_gap, 0, -1)
        return weights_list, predicted_values

    def _inverse(self, timeseries_data, batch_index, new_gap_list):
        """
        The time series method makes an inverse forecast based on the part
        of the time series that is located to the right of the gap.

        :param timeseries_data: one-dimensional array of a time series
        :param batch_index: index of the interval (batch) with a gap
        :param new_gap_list: array with nested lists of gap indexes

        :return weights_list: numpy array with prediction weights for
        averaging
        :return predicted_values: numpy array with predicted values in the
        gap
        """

        gap = new_gap_list[batch_index]

        # If the interval with a gap is the last one in the array
        if batch_index == len(new_gap_list) - 1:
            timeseries_train_part = timeseries_data[(gap[-1] + 1):]
        else:
            next_gap = new_gap_list[batch_index + 1]
            timeseries_train_part = timeseries_data[(gap[-1] + 1):next_gap[0]]
        timeseries_train_part = np.flip(timeseries_train_part)

        # Adaptive prediction interval length
        len_gap = len(gap)

        predicted_values = self.__chain_fit_predict(timeseries_train_part,
                                                    len_gap)

        predicted_values = np.flip(predicted_values)
        weights_list = np.arange(1, (len_gap + 1), 1)
        return weights_list, predicted_values

    def __chain_fit_predict(self, timeseries_train: np.array, len_gap: int):
        """
        The method makes a prediction as a sequence of elements based on a
        training sample. There are two main parts: fit model and predict.

        :param timeseries_train: part of the time series for training the model
        :param len_gap: number of elements in the gap
        :return: array without gaps
        """

        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_gap,
                                        max_window_size=self.max_window_size,
                                        return_all_steps=False,
                                        make_future_prediction=True))

        input_data = InputData(idx=np.arange(0, len(timeseries_train)),
                               features=None,
                               target=timeseries_train,
                               task=task,
                               data_type=DataTypesEnum.ts)

        # Making predictions for the missing part in the time series
        self.chain.fit_from_scratch(input_data)

        # "Test data" for making prediction for a specific length
        test_data = InputData(idx=np.arange(0, len_gap),
                              features=None,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

        predicted_values = self.chain.forecast(initial_data=input_data,
                                               supplementary_data=test_data).predict
        return predicted_values
