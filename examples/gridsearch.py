"""Example of an hyperparameter search."""

# local
from collections import defaultdict

import numpy as np

# local
from lrmc.factor_model import CMC  # , SCMC, WCMC

# third party
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid  # KFold
from utils import train_test_data


def gridsearch(param_grid, model, X_train, O_test, X_test, score_fn):
    result = defaultdict(list)

    for param_combo in param_grid:
        model.set_params(**param_combo)
        for key, value in param_combo.items():
            result[key].append(value)

        model.fit(X_train)

        X_hat_test = model.M * O_test

        score = score_fn(X_test, X_hat_test)
        result["score"].append(score)

    return result


def kfold_gridsearch(param_grid, model, X, mask, score_fn, n_splits=5, random_state=42):
    """
    result = defaultdict(list)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for num_cv, (train_index, test_index) in enumerate(kfold.split(X)):
        result["cv_fold"].extend([num_cv] * len(param_grid))
        print(result["cv_fold"])
        # assert afs
        # X_train = X * O_train
        # X_test = X * O_test

    return
    """
    pass


def main():
    X, O_train, O_test = train_test_data()

    X_train = X * O_train
    # X_test = X * O_test

    param_grid = {"lambda1": 10 ** np.linspace(-2, 2, 5)}
    param_grid = ParameterGrid(param_grid)

    model = CMC(rank=5, n_iter=3)

    # result = gridsearch(param_grid, model, X_train, O_test,
    #                        X_test, mean_squared_error)
    # print(result)

    kfold_gridsearch(param_grid, model, X_train, O_test, mean_squared_error)


if __name__ == "__main__":
    main()
