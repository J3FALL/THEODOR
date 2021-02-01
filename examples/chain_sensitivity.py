import json
import os

from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import project_root, default_fedot_data_dir
from fedot.sensitivity.chain_sensitivity import ChainStructureAnalyze
from fedot.sensitivity.model_sensitivity import ModelAnalyze


def get_three_depth_chain():
    logit_node_primary = PrimaryNode('logit')
    xgb_node_primary = PrimaryNode('xgboost')
    xgb_node_primary_second = PrimaryNode('xgboost')

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_primary_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_primary, xgb_node_primary])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    chain = Chain(knn_root)

    return chain


def get_scoring_data():
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)
    task = Task(TaskTypesEnum.classification)
    train = InputData.from_csv(full_path_train, task=task)
    test = InputData.from_csv(full_path_test, task=task)

    return train, test


def get_kc2_data():
    file_path = 'cases/data/kc2/kc2.csv'
    full_path = os.path.join(str(project_root()), file_path)
    task = Task(TaskTypesEnum.classification)
    data = InputData.from_csv(full_path, task=task)
    train, test = train_test_data_setup(data)

    return train, test


def run_analysis_case(train_data, test_data):
    chain = get_three_depth_chain()

    chain.fit(train_data)
    predicted_data = chain.predict(test_data)

    original_metric = - round(roc_auc(test_data.target, predicted_data.predict), 3)

    chain_analysis_result = ChainStructureAnalyze(chain=chain, train_data=train_data,
                                                  test_data=test_data, nodes_ids_to_analyze=[3, 5],
                                                  approaches=[ModelAnalyze]).analyze()

    print(f'chain analysis result {chain_analysis_result}')

    file_path_to_save = os.path.join(default_fedot_data_dir(), 'sa_result.json')

    with open(file_path_to_save, 'w', encoding='utf-8') as file:
        file.write(json.dumps(chain_analysis_result, indent=4))


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition

    # scoring case
    train_data, test_data = get_scoring_data()

    # kc2 case
    # train_data, test_data = get_kc2_data()

    run_analysis_case(train_data, test_data)
