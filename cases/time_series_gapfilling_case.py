import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.utils import project_root
from fedot.utilities.ts_gapfilling import ModelGapFiller


def print_metrics(dataframe):
    """
    The function displays 3 metrics: Mean absolute error,
    Root mean squared error and Median absolute error

    :param: dataframe with columns 'date','temperature','ridge','composite',
    'with_gap'
    """

    gap_array = np.array(dataframe['with_gap'])
    gap_ids = np.argwhere(gap_array == -100.0)

    actual = np.array(dataframe['temperature'])[gap_ids]
    ridge_predicted = np.array(dataframe['ridge'])[gap_ids]
    composite_predicted = np.array(dataframe['composite'])[gap_ids]

    model_labels = ['Inverted ridge regression', 'Composite model']
    for predicted, model_label in zip(
            [ridge_predicted, composite_predicted], model_labels):
        print(f"{model_label}")

        mae_metric = mean_absolute_error(actual, predicted)
        print(f"Mean absolute error - {mae_metric:.2f}")

        rmse_metric = (mean_squared_error(actual, predicted)) ** 0.5
        print(f"Root mean squared error - {rmse_metric:.2f}")

        median_ae_metric = median_absolute_error(actual, predicted)
        print(f"Median absolute error - {median_ae_metric:.2f} \n")


def plot_result(dataframe):
    """
    The function draws a graph based on the dataframe

    :param: dataframe with columns 'date','temperature','ridge','composite',
    'with_gap'
    """

    gap_array = np.array(dataframe['with_gap'])
    masked_array = np.ma.masked_where(gap_array == -100.0, gap_array)

    plt.plot(dataframe['date'], dataframe['temperature'], c='blue',
             alpha=0.5, label='Actual values', linewidth=1)
    plt.plot(dataframe['date'], dataframe['ridge'], c='orange',
             alpha=0.8, label='Inverse ridge gapfilling', linewidth=1)
    plt.plot(dataframe['date'], dataframe['composite'], c='red',
             alpha=0.8, label='Composite gapfilling', linewidth=1)
    plt.plot(dataframe['date'], masked_array, c='blue')
    plt.grid()
    plt.legend()
    plt.show()


def get_composite_chain():
    """
    The method returns prepared chain of 5 models

    :return: TsForecastingChain object
    """

    node_first = PrimaryNode('trend_data_model')
    node_second = PrimaryNode('residual_data_model')
    node_trend_model = SecondaryNode('linear', nodes_from=[node_first])
    node_residual_model = SecondaryNode('linear', nodes_from=[node_second])

    node_final = SecondaryNode('linear', nodes_from=[node_trend_model,
                                                     node_residual_model])
    chain = TsForecastingChain(node_final)
    return chain


def run_gapfilling_case():
    """
    The function runs an example of filling in gaps in a time series with
    air temperature. Real data case.

    :return: pandas dataframe with columns 'date','with_gap','ridge',
    'composite','temperature'
    """

    # Load dataframe
    file_path = 'cases/data/gapfilling/TS_temperature_gapfilling.csv'
    full_path = os.path.join(str(project_root()), file_path)
    dataframe = pd.read_csv(full_path)
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Filling in gaps based on inverted ridge regression model
    ridge_chain = TsForecastingChain(PrimaryNode('ridge'))
    ridge_gapfiller = ModelGapFiller(gap_value=-100.0,
                                     chain=ridge_chain,
                                     max_window_size=250)
    with_gap_array = np.array(dataframe['with_gap'])
    without_gap_arr_ridge = ridge_gapfiller.forward_inverse_filling(with_gap_array)
    dataframe['ridge'] = without_gap_arr_ridge

    # Filling in gaps based on a chain of 5 models
    composite_chain = get_composite_chain()
    composite_gapfiller = ModelGapFiller(gap_value=-100.0,
                                         chain=composite_chain,
                                         max_window_size=1000)
    without_gap_composite = composite_gapfiller.forward_filling(with_gap_array)
    dataframe['composite'] = without_gap_composite
    return dataframe


# Example of using the algorithm to fill in gaps in a time series
# The data is daily air temperature values from the weather station
if __name__ == '__main__':
    dataframe = run_gapfilling_case()

    # Display metrics
    print_metrics(dataframe)

    # Visualise predictions
    plot_result(dataframe)
