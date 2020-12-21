from datetime import timedelta

from sklearn.metrics import roc_auc_score as roc_auc

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData


def run_chain_from_automl(train_file_path: str, test_file_path: str,
                          max_run_time: timedelta = timedelta(minutes=10)):
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    testing_target = test_data.target

    chain = Chain()
    node_tpot = PrimaryNode('tpot')

    node_tpot.model.params = {'max_run_time_sec': max_run_time.seconds}

    node_lda = PrimaryNode('lda')
    node_rf = SecondaryNode('rf')

    node_rf.nodes_from = [node_tpot, node_lda]

    chain.add_node(node_rf)

    chain.fit(train_data)
    results = chain.predict(test_data)

    roc_auc_value = roc_auc(y_true=testing_target,
                            y_score=results.predict)
    print(roc_auc_value)

    return roc_auc_value


if __name__ == '__main__':
    train_file_path, test_file_path = get_scoring_case_data_paths()
    run_chain_from_automl(train_file_path, test_file_path)
