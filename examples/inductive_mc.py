"""
Simple example
"""

# third party
import numpy as np

# local
# from lmc import CMC
# from plotting import plot_profiles_and_observations
from sklearn.model_selection import train_test_split
from synthetic_data import synthetic_data_generator


def factor_model(timepoints, X, V):
    """Inductive matrix completion via coefficient regression.

    Args:
        timepoints: Times to predict
        V: Estimated basic profiles.

    Returns:
        _type_: _description_
    """
    U_star = (2 * V.T @ X) @ np.linalg.inv(V.T @ V)
    M_hat = U_star @ V.T

    return M_hat[range(timepoints.size), timepoints]


def synthetic_control(timepoints, M):
    """Synthetic control method."""
    return


def posterior_predictive(Y, M, t_pred, theta, number_of_states):
    """Posterior predictive distribution."""
    """Predict probabilities of future observations in longitudinal data
    with a latent variable model.

    Args:
        Y: A (M x T) longitudinal data matrix. Each row is a longitudinal vector with
            observed data up to times < t_pred.
        M: The data matrix computed from factor matrices derived from X (M = U @ V.T).
        t_pred: Time of predictions for each row in Y.
        theta: A confidence parameter (estimated from data in utils.py)

    Returns:
        A (M x number_of_states) matrix of probability estimates.
    """

    logl = np.ones((Y.shape[0], M.shape[0]))
    for i, y in enumerate(Y):
        omega = y != 0
        logl[i] = np.sum(
            -1.0 * theta * ((y[omega])[None, :] - M[:, omega]) ** 2, axis=1
        )

    proba_z = np.empty((Y.shape[0], number_of_states))
    domain = np.arange(1, number_of_states + 1)
    for i in range(Y.shape[0]):
        proba_z[i] = np.exp(logl[i]) @ np.exp(
            -1.0 * theta * (M[:, t_pred[i], None] - domain) ** 2
        )

    return proba_z / (np.sum(proba_z, axis=1))[:, None]


def main():
    rank = 5

    M, X = synthetic_data_generator(n_rows=100, n_timesteps=250, rank=rank)

    train_idx, test_idx = train_test_split(
        range(X.shape[0]), test_size=0.25, shuffle=False
    )

    # TODO plot predictions as cmats AND using delta scores

    # plot_profiles_and_observations(X, M, n_profiles=4,
    # # path_to_fig="./figures/ground_truth_data.pdf")

    # factorization with convolution
    # mfc_model = CMC(rank=rank, n_iter=200)
    # mfc_model.fit(X)

    # plot_profiles_and_observations(X, mfc_model.M, n_profiles=4,
    # # path_to_fig="./figures/mfc_model.pdf")


if __name__ == "__main__":
    main()
