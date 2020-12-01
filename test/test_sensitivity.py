from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.composer.node import SecondaryNode
from fedot.core.models.data import InputData
from sensitivity.chain_sensitivity import ChainStructureAnalyze
from sensitivity.sensitivity_facade import NodeDeletionAnalyze, NodeAnalysis, NodeTuneAnalyze, NodeReplaceModelAnalyze
from test.test_chain_import_export import create_four_depth_chain


def scoring_dataset():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    return train_data, test_data


def given_data():
    chain = create_four_depth_chain()
    train_data, test_data = scoring_dataset()
    node_index = 2

    return chain, train_data, test_data, node_index


def test_chain_structure_analyze():
    # given
    chain, train_data, test_data, _ = given_data()

    # when
    _, loss = ChainStructureAnalyze(chain=chain,
                                    train_data=train_data,
                                    test_data=test_data).analyze()

    assert isinstance(loss, list)


def test_node_analysis_facade():
    # given
    chain, train_data, test_data, node_index = given_data()

    # when
    node_analysis_result: dict = NodeAnalysis().analyze(chain=chain,
                                                        node_id=node_index,
                                                        train_data=train_data,
                                                        test_data=test_data)

    # then
    print(node_analysis_result)
    assert isinstance(node_analysis_result, dict)


def test_node_deletion_analyze():
    # given
    chain, train_data, test_data, node_index = given_data()

    # when
    node_analysis_result = NodeDeletionAnalyze(chain=chain,
                                               train_data=train_data,
                                               test_data=test_data).analyze(node_id=node_index)

    # then
    print(node_analysis_result)
    assert isinstance(node_analysis_result, float)


def test_node_tune_analyze():
    # given
    chain, train_data, test_data, node_index = given_data()

    # when
    node_analysis_result = NodeTuneAnalyze(chain=chain,
                                           train_data=train_data,
                                           test_data=test_data).analyze(node_id=node_index)
    # then
    assert isinstance(node_analysis_result, float)


def test_node_replacement_analyze():
    # given
    chain, train_data, test_data, node_index = given_data()

    replacing_node = SecondaryNode('lda')

    # when
    node_analysis_result = \
        NodeReplaceModelAnalyze(chain=chain,
                                train_data=train_data,
                                test_data=test_data). \
            analyze(node_id=node_index,
                    nodes_to_replace_to=[replacing_node])

    assert isinstance(node_analysis_result, list)
