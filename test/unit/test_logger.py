import os

import pytest

from fedot.core.data.data import train_test_data_setup, InputData
from fedot.core.log import Log, LogManager, default_log
from fedot.core.models.model import Model
from test.unit.utilities.test_chain_import_export import create_four_depth_chain


@pytest.fixture()
def get_config_file():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(test_file_path, '../data', 'logging.json')
    if os.path.exists(file):
        return file


@pytest.fixture()
def get_bad_config_file():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(test_file_path, '../data', 'bad_log_config_file.yml')
    if os.path.exists(file):
        return file


def release_log(logger, log_file):
    logger.release_handlers()
    if os.path.exists(log_file):
        os.remove(log_file)


def test_default_logger_setup_correctly():
    expected_logger_info_level = 20
    test_default_log = default_log('default_test_logger')

    assert test_default_log.logger.getEffectiveLevel() == expected_logger_info_level


@pytest.mark.parametrize('data_fixture', ['get_config_file'])
def test_logger_from_config_file_setup_correctly(data_fixture, request):
    expected_logger_error_level = 40
    test_config_file = request.getfixturevalue(data_fixture)
    log = Log('test_logger', config_json_file=test_config_file)

    assert log.logger.getEffectiveLevel() == expected_logger_error_level


def test_logger_write_logs_correctly():
    test_file_path = str(os.path.dirname(__file__))
    test_log_file = os.path.join(test_file_path, 'test_log.log')
    test_log = default_log('test_log',
                           log_file=test_log_file)

    # Model data preparation
    file = os.path.join('../data', 'advanced_classification.csv')
    data = InputData.from_csv(os.path.join(test_file_path, file))
    train_data, test_data = train_test_data_setup(data=data)

    try:
        knn = Model(model_type='knnreg', log=test_log)
        model, _ = knn.fit(data=train_data)
    except Exception:
        print('Captured error')

    if os.path.exists(test_log_file):
        with open(test_log_file, 'r') as file:
            content = file.readlines()

    release_log(logger=test_log, log_file=test_log_file)
    assert 'Can not find evaluation strategy' in content[0]


def test_logger_manager_keeps_loggers_correctly():
    LogManager().clear_cache()

    chain = create_four_depth_chain()
    expected_number_of_loggers = 4

    file = os.path.join('../data', 'advanced_classification.csv')
    test_file_path = str(os.path.dirname(__file__))
    data = InputData.from_csv(os.path.join(test_file_path, file))
    train_data, _ = train_test_data_setup(data=data)

    chain.fit(train_data)

    actual_number_of_loggers = LogManager().debug['loggers_number']

    assert actual_number_of_loggers == expected_number_of_loggers


@pytest.mark.parametrize('data_fixture', ['get_bad_config_file'])
def test_logger_from_config_file_raise_exception(data_fixture, request):
    test_bad_config_file = request.getfixturevalue(data_fixture)

    with pytest.raises(Exception) as exc:
        assert Log('test_logger', config_json_file=test_bad_config_file)

    assert 'Can not open the log config file because of' in str(exc.value)


def test_logger_str():
    test_default_log = default_log('default_test_logger_for_str')

    expected_str_value = "Log object for default_test_logger_for_str module"

    assert str(test_default_log) == expected_str_value


def test_logger_repr():
    test_default_log = default_log('default_test_logger_for_repr')

    expected_repr_value = "Log object for default_test_logger_for_repr module"

    assert repr(test_default_log) == expected_repr_value
