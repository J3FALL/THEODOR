import warnings
from abc import abstractmethod
from datetime import timedelta
from typing import Optional

from sklearn.cluster import DBSCAN as SklearnDBScan, KMeans as SklearnKmeans
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import (Lasso as SklearnLassoReg,
                                  LinearRegression as SklearnLinReg,
                                  LogisticRegression as SklearnLogReg,
                                  Ridge as SklearnRidgeReg,
                                  SGDRegressor as SklearnSGD)
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB
from sklearn.neighbors import (KNeighborsClassifier as SklearnKNN,
                               KNeighborsRegressor as SklearnKNNReg)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR as SklearnSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import Log, default_log
from fedot.core.models.evaluation.custom_models.models import CustomSVC
from fedot.core.models.tuning.hyperparams import params_range_by_model
from fedot.core.models.tuning.tuners import SklearnCustomRandomTuner, SklearnTuner

warnings.filterwarnings("ignore", category=UserWarning)


class EvaluationStrategy:
    """
    Base class to define the evaluation strategy of Model object:
    the certain sklearn or any other model with fit/predict methods.

    :param model_type: str type of the model defined in model repository
    :param dict params: hyperparameters to fit the model with
    :param Log log: Log object to record messages
    """

    def __init__(self, model_type: str, params: Optional[dict] = None,
                 log=None):
        self.params_for_fit = params
        self.model_type = model_type

        self.output_mode = False

        if not log:
            self.log: Log = default_log(__name__)
        else:
            self.log: Log = log

    @abstractmethod
    def fit(self, train_data: InputData):
        """
        Main method to train the model with the data provided

        :param InputData train_data: data used for model training
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        """
        Main method to predict the target data.

        :param trained_model: trained model object
        :param InputData predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """
        raise NotImplementedError()

    @abstractmethod
    def fit_tuned(self, train_data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        """
        Main method used for hyperparameter searching

        :param train_data: data used for hyperparameter searching
        :param iterations: max number of iterations evaluable for hyperparameter optimization
        :param max_lead_time: max time(seconds) for tuning evaluation
        :return:
        """
        raise NotImplementedError()

    @property
    def implementation_info(self) -> str:
        return 'No description'


class SkLearnEvaluationStrategy(EvaluationStrategy):
    """
    This class defines the certain model implementation for the sklearn models defined in model repository

    :param str model_type: str type of the model defined in model repository
    :param dict params: hyperparameters to fit the model with
    """
    __model_by_types = {
        'xgboost': XGBClassifier,
        'xgbreg': XGBRegressor,
        'adareg': AdaBoostRegressor,
        'gbr': GradientBoostingRegressor,
        'knn': SklearnKNN,
        'knnreg': SklearnKNNReg,
        'dt': DecisionTreeClassifier,
        'dtreg': DecisionTreeRegressor,
        'treg': ExtraTreesRegressor,
        'rf': RandomForestClassifier,
        'rfr': RandomForestRegressor,
        'mlp': MLPClassifier,
        'lda': LinearDiscriminantAnalysis,
        'qda': QuadraticDiscriminantAnalysis,
        'linear': SklearnLinReg,
        'logit': SklearnLogReg,
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        'kmeans': SklearnKmeans,
        'dbscan': SklearnDBScan,
        'svc': CustomSVC,
        'svr': SklearnSVR,
        'sgdr': SklearnSGD,
        'bernb': SklearnBernoulliNB
    }

    def __init__(self, model_type: str, params: Optional[dict] = None):
        self._sklearn_model_impl = self._convert_to_sklearn(model_type)
        self._tune_strategy: SklearnTuner = Optional[SklearnTuner]
        super().__init__(model_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for model training with the data provided

        :param InputData train_data: data used for model training
        :return:
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            sklearn_model = self._sklearn_model_impl(**self.params_for_fit)
        else:
            sklearn_model = self._sklearn_model_impl()

        sklearn_model.fit(train_data.features, train_data.target)
        return sklearn_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data.

        :param trained_model: model object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """
        raise NotImplementedError()

    def fit_tuned(self, train_data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        """
        This method is used for hyperparameter searching

        :param train_data: data used for hyperparameter searching
        :param iterations: max number of iterations evaluable for hyperparameter optimization
        :param max_lead_time: max time(seconds) for tuning evaluation
        :return tuple(object, dict): model with found hyperparameters and dictionary with found hyperparameters
        """
        trained_model = self.fit(train_data=train_data)
        params_range = params_range_by_model.get(self.model_type, None)
        self._tune_strategy = SklearnCustomRandomTuner
        if not params_range:
            self.params_for_fit = None
            return trained_model, trained_model.get_params()

        tuned_params, best_model = self._tune_strategy(trained_model=trained_model,
                                                       tune_data=train_data,
                                                       params_range=params_range,
                                                       cross_val_fold_num=5,
                                                       time_limit=max_lead_time,
                                                       iterations=iterations).tune()

        if best_model or tuned_params:
            self.params_for_fit = tuned_params
            trained_model = best_model

        return trained_model, tuned_params

    def _convert_to_sklearn(self, model_type: str):
        if model_type in self.__model_by_types.keys():
            return self.__model_by_types[model_type]
        else:
            raise ValueError(f'Impossible to obtain SKlearn strategy for {model_type}')

    def _find_model_by_impl(self, impl):
        for model, model_impl in self.__model_by_types.items():
            if model_impl == impl:
                return model

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_sklearn(self.model_type))


class SkLearnClassificationStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_model, predict_data: InputData):
        """
        Predict method for classification task

        :param trained_model: model object
        :param predict_data: data used for prediction
        :return: prediction target
        """
        n_classes = len(trained_model.classes_)
        if self.output_mode == 'labels':
            prediction = trained_model.predict(predict_data.features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_model.predict_proba(predict_data.features)
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs':
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')
        return prediction


class SkLearnRegressionStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_model, predict_data: InputData):
        """
        Predict method for regression task

        :param trained_model: model object
        :param predict_data: data used for prediction
        :return:
        """
        prediction = trained_model.predict(predict_data.features)
        return prediction


class SkLearnClusteringStrategy(SkLearnEvaluationStrategy):
    def fit(self, train_data: InputData):
        """
        Fit method for clustering task

        :param train_data: data used for model training
        :return:
        """
        try:
            # TODO auto detect the number of clusters
            sklearn_model = self._sklearn_model_impl(n_clusters=3)
        except TypeError as _:
            sklearn_model = self._sklearn_model_impl()
        sklearn_model = sklearn_model.fit(train_data.features)
        return sklearn_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        """
        Predict method for clustering task
        :param trained_model: model object
        :param predict_data: data used for prediction
        :return:
        """
        try:
            prediction = trained_model.predict(predict_data.features)
        except Exception as _:
            prediction = trained_model.fit_predict(predict_data.features)
        return prediction

    def fit_tuned(self, train_data: InputData, iterations: int = 30,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        """
        This method is used for hyperparameter searching

        :param train_data: data used for hyperparameter searching
        :param iterations: max number of iterations evaluable for hyperparameter optimization
        :param max_lead_time: max time(seconds) for tuning evaluation
        :return:
        """
        raise NotImplementedError()
