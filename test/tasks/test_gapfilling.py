import numpy as np
from sklearn.metrics import mean_squared_error

from examples.time_series_gapfilling_example import get_array_with_gaps
from fedot.core.chains.node import PrimaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.utilities.ts_gapfilling import ModelGapFiller


def test_get_array_with_gaps():
    # Checking whether omissions with the correct values are generated in
    # the right places
    for gap_dict, gap_value in zip([{500: 150}, {700: 10}], [-10.0, -100.0]):
        arr_with_gaps, _ = get_array_with_gaps(gap_dict, gap_value)

        start_gaps = list(gap_dict.keys())
        index_to_check = start_gaps[0]

        assert arr_with_gaps[index_to_check] == gap_value


def test_gapfilling_inverse_ridge_correct():
    arr_with_gaps, real_values = get_array_with_gaps()

    # Find all gap indices in the array
    id_gaps = np.ravel(np.argwhere(arr_with_gaps == -100.0))

    ridge_chain = TsForecastingChain(PrimaryNode('ridge'))
    gapfiller = ModelGapFiller(gap_value=-100.0, chain=ridge_chain,
                               max_window_size=150)
    without_gap = gapfiller.forward_inverse_filling(arr_with_gaps)

    # Get only values in the gaps
    predicted_values = without_gap[id_gaps]
    true_values = real_values[id_gaps]

    rmse_test = mean_squared_error(true_values, predicted_values, squared=False)

    # The RMSE must be less than the standard deviation of random noise * 1.5
    assert rmse_test < 0.15


def test_gapfilling_forward_ridge_correct():
    arr_with_gaps, real_values = get_array_with_gaps()

    # Find all gap indices in the array
    id_gaps = np.ravel(np.argwhere(arr_with_gaps == -100.0))

    ridge_chain = TsForecastingChain(PrimaryNode('ridge'))
    gapfiller = ModelGapFiller(gap_value=-100.0, chain=ridge_chain,
                               max_window_size=150)
    without_gap = gapfiller.forward_filling(arr_with_gaps)

    # Get only values in the gaps
    predicted_values = without_gap[id_gaps]
    true_values = real_values[id_gaps]

    rmse_test = mean_squared_error(true_values, predicted_values, squared=False)

    # The RMSE must be less than the standard deviation of random noise * 2.0
    assert rmse_test < 0.2
