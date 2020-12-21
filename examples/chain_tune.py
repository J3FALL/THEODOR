import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData


def get_case_train_test_data():
    train_file_path, test_file_path = get_scoring_case_data_paths()

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)
    return train_data, test_data


def get_simple_chain():
    first = PrimaryNode(model_type='xgboost')
    second = PrimaryNode(model_type='knn')
    final = SecondaryNode(model_type='logit',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain


def chain_tuning(nodes_to_tune: str, chain: Chain, train_data: InputData,
                 test_data: InputData, local_iter: int,
                 tuner_iter_num: int = 50) -> (float, list):
    several_iter_scores_test = []

    if nodes_to_tune == 'primary':
        print('primary_node_tuning')
        chain_tune_strategy = chain.fine_tune_primary_nodes
    elif nodes_to_tune == 'root':
        print('root_node_tuning')
        chain_tune_strategy = chain.fine_tune_all_nodes
    else:
        raise ValueError(f'Invalid type of nodes. Nodes must be primary or root')

    for iteration in range(local_iter):
        print(f'current local iteration {iteration}')

        # Chain tuning
        chain_tune_strategy(train_data, iterations=tuner_iter_num)

        # After tuning prediction
        chain.fit(train_data)
        after_tuning_predicted = chain.predict(test_data)

        # Metrics
        aft_tun_roc_auc = roc_auc(y_true=test_data.target,
                                  y_score=after_tuning_predicted.predict)
        several_iter_scores_test.append(aft_tun_roc_auc)

    return float(np.mean(several_iter_scores_test)), several_iter_scores_test


if __name__ == '__main__':
    train_file_path, test_file_path = get_scoring_case_data_paths()

    train_data, test_data = get_case_train_test_data()

    # Chain composition
    chain = get_simple_chain()

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)
    bfr_tun_roc_auc = roc_auc(y_true=test_data.target,
                              y_score=before_tuning_predicted.predict)

    local_iter = 5
    # Chain tuning
    after_tune_roc_auc, several_iter_scores_test = chain_tuning(nodes_to_tune='primary',
                                                                chain=chain,
                                                                train_data=train_data,
                                                                test_data=test_data,
                                                                local_iter=local_iter)

    print(f'Several test scores {several_iter_scores_test}')
    print(f'Mean test score over {local_iter} iterations: {after_tune_roc_auc}')
    print(round(bfr_tun_roc_auc, 3))
    print(round(after_tune_roc_auc, 3))
