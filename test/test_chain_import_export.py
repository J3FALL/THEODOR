import json
import os
import shutil

import pytest

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.utilities.synthetic.chain_template_new import ChainTemplate, extract_subtree_root


@pytest.fixture(scope="session", autouse=True)
def creation_model_files_before_after_tests(request):
    create_json_models_files()
    request.addfinalizer(delete_json_models_files)


def create_json_models_files():
    """
    Creating JSON's files for test.
    """
    chain = create_chain()
    chain.save_chain("data/test_chain_convert_to_json.json")

    chain_fitted = create_fitted_chain()
    chain_fitted.save_chain("data/test_fitted_chain_convert_to_json.json")

    chain_empty = Chain()
    chain_empty.save_chain("data/test_empty_chain_convert_to_json.json")


def delete_json_models_files():
    """
    Delete JSON's files.
    """
    with open("data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        chain_fitted_object = json.load(json_file)

    delete_fitted_models(chain_fitted_object)

    os.remove("data/test_fitted_chain_convert_to_json.json")
    os.remove("data/test_empty_chain_convert_to_json.json")
    os.remove("data/test_chain_convert_to_json.json")


def delete_fitted_models(chain):
    """
    Delete directory and chain's local fitted models.

    :param chain: chain which model's need to delete
    """
    model_path = chain['nodes'][0]['trained_model_path']
    dir_path = os.path.dirname(os.path.abspath(model_path))
    shutil.rmtree(dir_path)


def create_chain() -> Chain:
    chain = Chain()
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = PrimaryNode('xgboost')

    node_knn = PrimaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_lda, node_knn]

    node_logit_second = SecondaryNode('logit')
    node_logit_second.nodes_from = [node_xgboost, node_lda]

    node_lda_second = SecondaryNode('lda')
    node_lda_second.custom_params = {'n_components': 1}
    node_lda_second.nodes_from = [node_logit_second, node_knn_second, node_logit]

    node_xgboost_second = SecondaryNode('xgboost')
    node_xgboost_second.nodes_from = [node_logit, node_logit_second, node_knn]

    node_knn_third = SecondaryNode('knn')
    node_knn_third.custom_params = {'n_neighbors': 8}
    node_knn_third.nodes_from = [node_lda_second, node_xgboost_second]

    chain.add_node(node_knn_third)

    return chain


def create_fitted_chain() -> Chain:
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = create_chain()
    chain.fit(train_data)

    return chain


def test_export_chain_to_json_correctly():
    chain = create_chain()
    json_actual = chain.save_chain("data/1.json")

    with open("data/test_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    os.remove("data/1.json")
    assert json_actual == json.dumps(json_expected)


def test_chain_template_to_json_correctly():
    chain = create_chain()
    chain_template = ChainTemplate(chain)
    json_actual = chain_template.convert_to_dict()

    with open("data/test_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_to_chain_correctly():
    chain = Chain()
    chain.load_chain("data/test_chain_convert_to_json.json")
    json_actual = chain.save_chain("data/1.json")

    chain_expected = create_chain()
    json_expected = chain_expected.save_chain("data/2.json")

    os.remove("data/1.json")
    os.remove("data/2.json")
    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_template_to_chain_correctly():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_from_json("data/test_chain_convert_to_json.json")
    json_actual = chain_template.convert_to_dict()

    chain_expected = create_chain()
    chain_expected_template = ChainTemplate(chain_expected)
    json_expected = chain_expected_template.convert_to_dict()

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_import_json_to_fitted_chain_correctly():
    chain = Chain()
    chain.load_chain("data/test_fitted_chain_convert_to_json.json")
    json_actual = chain.save_chain("data/1.json")

    with open("data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    os.remove("data/1.json")
    assert json_actual == json.dumps(json_expected)


def test_import_json_to_fitted_chain_template_correctly():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_from_json("data/test_fitted_chain_convert_to_json.json")
    json_actual = chain_template.convert_to_dict()

    with open("data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_empty_chain_to_json_correctly():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    json_actual = chain_template.convert_to_dict()

    with open("data/test_empty_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_export_import_for_one_chain_object_correctly():
    """
    This test checks whether it is possible to upload a new model to the same object. In other words,
     apply a sequence of commands to the chain object:
    - load_chain(path_first)
    - load_chain(path_second)
    - load_chain(path_third)
    and the last command will rewrite the chain object correctly.
    """
    chain_fitted = create_fitted_chain()
    json_first = chain_fitted.save_chain("data/2.json")

    chain_fitted_after = create_chain()
    chain_fitted_after.save_chain("data/1.json")
    chain_fitted_after.load_chain("data/2.json")

    json_second = chain_fitted_after.save_chain("data/3.json")

    for i in range(1, 4):
        os.remove(f"data/{i}.json")

    delete_fitted_models(json.loads(json_first))
    assert json_first == json_second


def test_absolute_relative_paths_correctly_no_exception():
    chain = create_chain()
    chain.save_chain("data/test/1.json")

    absolute_path = os.path.join(os.path.abspath("data/2.json"))
    chain.save_chain(absolute_path)

    chain.load_chain("data/test/1.json")
    chain.load_chain(absolute_path)

    os.remove("data/test/1.json")
    os.remove(absolute_path)
    os.rmdir("data/test")


def test_import_custom_json_object_to_chain_and_fit_correctly_no_exception():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    data_path = str(os.path.dirname(__file__))
    json_file_path = os.path.join(data_path, '..', 'test', 'data', 'test_custom_json_template.json')

    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_from_json(json_file_path)

    chain.fit(train_data)
    json_actual = chain.save_chain("data/1.json")

    delete_fitted_models(json.loads(json_actual))
    os.remove("data/1.json")


def create_four_depth_chain():
    knn_node = PrimaryNode('knn')
    lda_node = PrimaryNode('lda')
    xgb_node = PrimaryNode('xgboost')
    logit_node = PrimaryNode('logit')

    logit_node_second = SecondaryNode('logit', nodes_from=[knn_node, lda_node])
    xgb_node_second = SecondaryNode('xgboost', nodes_from=[logit_node])

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_second, xgb_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    chain = Chain()
    chain.add_node(knn_root)

    return chain


def test_extract_subtree_root():
    chain = create_four_depth_chain()
    chain_template = ChainTemplate(chain)

    expected_types = ['knn', 'logit', 'knn', 'lda', 'xgboost']
    new_root_node_id = 4

    root_node = extract_subtree_root(root_model_id=new_root_node_id,
                                     chain_template=chain_template)

    sub_chain = Chain()
    sub_chain.add_node(root_node)
    actual_types = [node.model.model_type for node in sub_chain.nodes]

    assertion_list = [True if expected_types[index] == actual_types[index] else False
                      for index in range(len(expected_types))]
    assert all(assertion_list)
