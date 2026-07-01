"""Selecting model rank with LarsMC.

* Algorithm iterates until all variables in acive set or
  n_iter > max_iter. Thus, max_iter < rank impose sparsity.

* LassoLars is subclass of Lars which has argument `n_nonzero_coefs`.
  In LassoLars, `n_nonzero_coefs=max_iter`.

* Weights might blow-up if fitting intercept.
"""

# local
from lrmc.factor_model import LarsMC

# third party
from sklearn.metrics import mean_squared_error
from utils import train_test_data


def main():
    X, O_train, O_test = train_test_data()

    X_train = X * O_train
    X_test = X * O_test

    model = LarsMC(rank=5, alpha=1, n_iter=10)
    model.fit(X_train)

    Y_test = model.M * O_test

    score = mean_squared_error(X_test, Y_test)
    print(score)


if __name__ == "__main__":
    main()