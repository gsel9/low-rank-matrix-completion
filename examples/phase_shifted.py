"""Demonstrating the utility of SCMC on phase shifted profiles.
"""

# local
from lrmc.factor_model import SCMC

# third party
from sklearn.metrics import mean_squared_error
from utils import train_test_data


def main():
    X, O_train, O_test = train_test_data()

    X_train = X * O_train
    X_test = X * O_test

    model = SCMC(rank=5, shift_budget=range(2), n_iter=10)
    model.fit(X_train)

    Y_test = model.M * O_test

    score = mean_squared_error(X_test, Y_test)
    print(score)


if __name__ == "__main__":
    main()
