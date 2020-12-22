import numpy as np

# the parameters ranges are partially derived from https://github.com/EpistasisLab/tpot
params_range_by_model = {
    'kmeans': {'n_clusters': list(range(2, 10))},
    'knn': {
        'n_neighbors': list(range(1, 50)),
        'weights': ["uniform", "distance"],
        'p': [1, 2]},
    'logit': {
        'C': [1e-2, 1e-1, 0.5, 0.9, 1., 2., 5., 10.]},
    'xgboost': {
        'n_estimators': [100],
        'max_depth': list(range(1, 7)),
        'learning_rate': list(np.arange(0.1, 0.9, 0.1)),
        'subsample': list(np.arange(0.05, 1.01, 0.05)),
        'min_child_weight': list(range(1, 21)),
        'nthread': [1]},
    'rf': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': list(range(2, 10)),
        'min_samples_leaf': list(range(1, 15)),
        'bootstrap': [True, False]},
    'xgbreg': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'objective': ['reg:squarederror']
    },
    'rfr': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
    'svr': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },
    'knnreg': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },
    'dtreg': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },
    'adareg': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },
    'gbr': {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },
    'treg': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    }
}
