from unittest.mock import patch

import pytest

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.chains.node import SecondaryNode
from fedot.core.data.data import InputData
from fedot.sensitivity.chain_sensitivity import ChainStructureAnalyze
from fedot.sensitivity.sensitivity_facade import \
    NodeDeletionAnalyze, NodeAnalysis, NodeTuneAnalyze, NodeReplaceModelAnalyze
from test.unit.utilities.test_chain_import_export import create_four_depth_chain


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


# ------------------------------------------------------------------------------
# ChainStructureAnalysis

def test_chain_structure_analyze_init():
    # given
    chain, train_data, test_data, _ = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    chain_analyzer = ChainStructureAnalyze(chain=chain,
                                           train_data=train_data,
                                           test_data=test_data,
                                           approaches=approaches)

    assert isinstance(chain_analyzer, ChainStructureAnalyze)


def test_chain_structure_analyze_init_raise_exception():
    # given
    chain, train_data, test_data, _ = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    with pytest.raises(ValueError) as exc:
        assert ChainStructureAnalyze(chain=chain,
                                     train_data=train_data,
                                     test_data=test_data,
                                     approaches=approaches,
                                     all_nodes=True,
                                     nodes_ids_to_analyze=[2])

    assert str(exc.value) == "Choose only one parameter between " \
                             "all_nodes and nodes_ids_to_analyze"


@patch('fedot.sensitivity.chain_sensitivity.ChainStructureAnalyze.analyze', return_value={'key': 'value'})
def test_chain_structure_analyze_analyze_all_nodes(mock):
    # given
    chain, train_data, test_data, _ = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    result = ChainStructureAnalyze(chain=chain,
                                   train_data=train_data,
                                   test_data=test_data,
                                   approaches=approaches,
                                   all_nodes=True).analyze()

    assert isinstance(result, dict)


@patch('fedot.sensitivity.chain_sensitivity.ChainStructureAnalyze.analyze', return_value={'key': 'value'})
def test_chain_structure_analyze_analyze_certain_nodes(mock):
    # given
    chain, train_data, test_data, _ = given_data()
    approaches = [NodeDeletionAnalyze]

    # when
    result = ChainStructureAnalyze(chain=chain,
                                   train_data=train_data,
                                   test_data=test_data,
                                   approaches=approaches,
                                   nodes_ids_to_analyze=[2]).analyze()
    assert isinstance(result, dict)


# ------------------------------------------------------------------------------
# NodeAnalysis


def test_node_analysis_facade_init_default():
    # given

    # when
    node_analyzer = NodeAnalysis()

    # then
    assert isinstance(node_analyzer, NodeAnalysis)
    assert len(node_analyzer.approaches) == 3


def test_node_analysis_facade_init_defined_approaches():
    # given
    approaches = [NodeDeletionAnalyze, NodeReplaceModelAnalyze]

    node_analyzer = NodeAnalysis(approaches=approaches)

    # then
    assert isinstance(node_analyzer, NodeAnalysis)
    assert len(node_analyzer.approaches) == 2


# @patch('fedot.sensitivity.sensitivity_facade.NodeAnalysis.analyze', return_value={'key': 'value'})
def test_node_analysis_facade_analyze():
    # given
    chain, train_data, test_data, node_index = given_data()

    # when
    node_analysis_result: dict = NodeAnalysis().analyze(chain=chain,
                                                        node_id=node_index,
                                                        train_data=train_data,
                                                        test_data=test_data)

    assert isinstance(node_analysis_result, dict)


# ------------------------------------------------------------------------------
# NodeAnalyzeApproach

# @patch('fedot.sensitivity.sensitivity_facade.NodeDeletionAnalyze.analyze', return_value=0.0)
def test_node_deletion_analyze():
    # given
    chain, train_data, test_data, node_index = given_data()

    # when
    node_analysis_result = NodeDeletionAnalyze(chain=chain,
                                               train_data=train_data,
                                               test_data=test_data).analyze(node_id=node_index)

    # then
    assert isinstance(node_analysis_result, float)


def test_node_deletion_analyze_zero_node_id():
    # given
    chain, train_data, test_data, _ = given_data()

    # when
    node_analysis_result = NodeDeletionAnalyze(chain=chain,
                                               train_data=train_data,
                                               test_data=test_data).analyze(node_id=0)

    # then
    assert isinstance(node_analysis_result, float)
    assert node_analysis_result == 0.0


# @patch('fedot.sensitivity.sensitivity_facade.NodeTuneAnalyze.analyze', return_value=0.0)
def test_node_tune_analyze():
    # given
    chain, train_data, test_data, node_index = given_data()

    # when
    node_analysis_result = NodeTuneAnalyze(chain=chain,
                                           train_data=train_data,
                                           test_data=test_data).analyze(node_id=node_index)
    # then
    assert isinstance(node_analysis_result, float)


# @patch('fedot.sensitivity.sensitivity_facade.NodeReplaceModelAnalyze.analyze', return_value=[0.0])
def test_node_replacement_analyze_defined_nodes():
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

    # then
    assert isinstance(node_analysis_result, list)


# @patch('fedot.sensitivity.sensitivity_facade.NodeReplaceModelAnalyze.analyze', return_value=[0.0, 0.0, 0.0])
def test_node_replacement_analyze_random_nodes_default_number():
    # given
    chain, train_data, test_data, node_index = given_data()

    # when
    node_analysis_result = \
        NodeReplaceModelAnalyze(chain=chain,
                                train_data=train_data,
                                test_data=test_data). \
            analyze(node_id=node_index)

    # then
    assert isinstance(node_analysis_result, list)
    assert len(node_analysis_result) == 3
