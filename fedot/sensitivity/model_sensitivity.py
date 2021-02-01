from copy import deepcopy
# from multiprocessing import Process, Manager
from threading import Thread, Lock
from os.path import join
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample import saltelli
from SALib.sample.latin import sample as lhc_sample
from SALib.sample.morris import sample as morris_sample
from sklearn.metrics import mean_squared_error

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.node_sensitivity import NodeAnalyzeApproach
from fedot.core.models.model_template import extract_model_params


class ModelAnalyze(NodeAnalyzeApproach):
    lock = Lock()

    def __init__(self, chain: Chain, train_data, test_data: InputData):
        super(ModelAnalyze, self).__init__(chain, train_data, test_data)
        self.model_params = None
        self.model_type = None
        self.problem = None
        self.analyze_method = None
        self.sample_method = None
        self.manager_dict = {}

    def analyze(self, node_id: int,
                sa_method: str = 'sobol',
                sample_method: str = 'saltelli',
                sample_size: int = 10,
                is_oat: bool = True) -> Union[List[dict], float]:

        # check whether the chain is fitted
        if not self._chain.fitted_on_data:
            self._chain.fit(self._train_data)

        self.analyze_method = analyze_method_by_name.get(sa_method)
        self.sample_method = sample_method_by_name.get(sample_method)

        self.model_type: str = self._chain.nodes[node_id].model.model_type

        self.model_params = model_params_with_bounds_by_model_name.get(self.model_type)
        self.problem = _create_problem_for_sobol_method(self.model_params)

        samples: List[dict] = self.sample()

        response_matrix = self.get_model_response_matrix(samples, node_id)
        indices = self.analyze_method(self.problem, response_matrix)
        converted_to_json_indices = convert_results_to_json(problem=self.problem,
                                                            si=indices)

        if is_oat:
            self._one_at_a_time_analyze(node_id=node_id,
                                        sample_size=sample_size)

        return [converted_to_json_indices]

    def sample(self, *args) -> Union[Union[List[Chain], Chain], List[dict]]:
        sample_size = args[0]
        problem_samples = self.sample_method(self.problem, num_of_samples=sample_size)
        problem_samples = clean_sample_variables(problem_samples)

        return problem_samples

    def get_model_response_matrix(self, samples, node_id: int):
        model_response_matrix = []
        for sample in samples:
            chain = deepcopy(self._chain)
            chain.nodes[node_id].custom_params = sample

            chain.fit(self._train_data)
            prediction = chain.predict(self._test_data)
            model_response_matrix.append(mean_squared_error(y_true=self._test_data.target,
                                                            y_pred=prediction.predict))

        return np.array(model_response_matrix)

    def worker(self, param: dict, node_id, sample_size: int = 100):
        # sample
        problem = _create_problem_for_sobol_method(param)
        # TODO extract hardcode sample method
        samples, converted_samples = make_latin_hypercube_sample(problem, sample_size)
        cleaned_samples = clean_sample_variables(converted_samples)

        # default values of param & loss
        param_name = list(param.keys())[0]
        default_param_value = extract_model_params(self._chain.nodes[node_id]).get(param_name)
        original_predict = self._chain.predict(self._test_data)
        loss_on_default = mean_squared_error(y_true=self._test_data.target,
                                             y_pred=original_predict.predict)

        # percentage ratio
        samples = samples.reshape(1, -1)[0]
        samples = (samples - default_param_value) / default_param_value - 100
        response_matrix = self.get_model_response_matrix(cleaned_samples, node_id)
        response_matrix = (response_matrix - loss_on_default) / loss_on_default - 100

        ModelAnalyze.lock.acquire()
        self.manager_dict[f'{param_name}'] = [samples.reshape(1, -1)[0], response_matrix]
        ModelAnalyze.lock.release()

    def visualize(self, data: dict):
        x_ticks = list()
        for param in data.keys():
            x_ticks.append(param)
            x_ticks.append(f'{param}_loss')
        new_data = []
        for value in data.values():
            new_data.extend(value)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.boxplot(new_data)
        plt.xticks(range(1, len(x_ticks) + 1), x_ticks)

        plt.savefig(join(default_fedot_data_dir(), f'{self.model_type}_sa.jpg'))

    def _one_at_a_time_analyze(self, node_id, sample_size=2):
        one_at_a_time_params = [{key: value} for key, value in self.model_params.items()]
        jobs = [Thread(target=self.worker,
                       args=(param, node_id, sample_size)) for param in one_at_a_time_params]

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        for param in one_at_a_time_params:
            self.worker(param, node_id, sample_size)

        for key, value in self.manager_dict.items():
            print(f'key = {key} : {value}')

        self.visualize(data=self.manager_dict)


def sobol_method(problem, model_response) -> dict:
    sobol_indices = sobol_analyze(problem, model_response, print_to_console=False)

    return sobol_indices


def morris_method(problem, model_response) -> dict:
    pass


def make_saltelly_sample(problem, num_of_samples=100):
    params_samples = saltelli.sample(problem, num_of_samples)
    params_samples = _convert_sample_to_dict(problem, params_samples)

    return params_samples


def make_moris_sample(problem, num_of_samples=100):
    params_samples = morris_sample(problem, num_of_samples, num_levels=4)
    params_samples = _convert_sample_to_dict(problem, params_samples)

    return params_samples


def make_latin_hypercube_sample(problem, num_of_samples=100):
    samples = lhc_sample(problem, num_of_samples)
    converted_samples = _convert_sample_to_dict(problem, samples)

    return samples, converted_samples


def clean_sample_variables(samples: List[dict]):
    for sample in samples:
        for key, value in sample.items():
            if key in INTEGER_PARAMS:
                sample[key] = int(value)

    return samples


def _create_problem_for_sobol_method(params: dict):
    problem = {
        'num_vars': len(params),
        'names': list(params.keys()),
        'bounds': list()
    }

    for key, bounds in params.items():
        if bounds[0] is not str:
            bounds = list(bounds)
            problem['bounds'].append([bounds[0], bounds[-1]])
        else:
            problem['bounds'].append(bounds)
    return problem


def _convert_sample_to_dict(problem, samples) -> List[dict]:
    converted_samples = []
    names_of_params = problem['names']
    for sample in samples:
        new_params = {}
        for index, value in enumerate(sample):
            new_params[names_of_params[index]] = value
        converted_samples.append(new_params)

    return converted_samples


def convert_results_to_json(problem: dict, si: dict):
    sobol_indices = []
    for index in range(problem['num_vars']):
        var_indices = {f"{problem['names'][index]}": {
            'S1': list(si['S1'])[index],
            'S1_conf': list(si['S1_conf'])[index],
            'ST': list(si['ST'])[index],
            'ST_conf': list(si['ST_conf'])[index],
        }}
        sobol_indices.append(var_indices)

    data = {
        'problem': {
            'num_vars': problem['num_vars'],
            'names': problem['names'],
            'bounds': problem['bounds']

        },
        'sobol_indices': sobol_indices
    }

    return data


model_params_with_bounds_by_model_name = {
    'xgboost': {
        'n_estimators': [10, 100],
        'max_depth': [1, 7],
        'learning_rate': [0.1, 0.9],
        'subsample': [0.05, 1.0],
        'min_child_weight': [1, 21]},
    'logit': {
        'C': [1e-2, 10.]},
    'knn': {
        'n_neighbors': [1, 50],
        'p': [1, 2]},
    'qda': {
        'reg_param': [0.1, 0.5]},
}

analyze_method_by_name = {
    'sobol': sobol_method,
    'morris': morris_method,
}

sample_method_by_name = {
    'saltelli': make_saltelly_sample,
    'morris': make_saltelly_sample,
    'sobol_sequence': None,
    'latin_hyper_cube': make_latin_hypercube_sample,

}

INTEGER_PARAMS = ['n_estimators', 'n_neighbors', 'p', 'min_child_weight', 'max_depth']
