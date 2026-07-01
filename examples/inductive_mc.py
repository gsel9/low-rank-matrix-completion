"""
Simple example
"""

# third party
import numpy as np

# local
from lrmc.factor_model import CMC
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


def posterior_predictive(Y, M, t_pred, theta, number_of_states):
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
    # Domain for probability kernel
    domain = np.arange(1, number_of_states + 1)
    for i in range(Y.shape[0]):
        proba_z[i] = np.exp(logl[i]) @ np.exp(
            -1.0 * theta * (M[:, t_pred[i], None] - domain) ** 2
        )

    return proba_z / (np.sum(proba_z, axis=1))[:, None]


def main():
    rank = 5
    n_timesteps = 250
    number_of_states = 4
    theta = 2.5

    M, X = synthetic_data_generator(
        n_rows=100,
        n_timesteps=n_timesteps,
        rank=rank,
        number_of_states=number_of_states,
        theta=theta,
    )

    train_idx, test_idx = train_test_split(
        range(X.shape[0]), test_size=0.25, shuffle=False
    )

    # Fit matrix completion only on the training entities. The test
    # entities are held out and never seen while fitting the model.
    mc_model = CMC(rank=rank, n_iter=200)
    mc_model.fit(X[train_idx])

    # Predict the state of each held-out test entity at a future time
    # point, having only observed its profile up to that point.
    t_pred_time = n_timesteps - 50
    t_pred = [t_pred_time] * len(test_idx)

    Y_test = X[test_idx].copy()
    Y_test[:, t_pred_time:] = 0

    proba_z = posterior_predictive(
        Y_test, mc_model.M, t_pred, theta=theta, number_of_states=number_of_states
    )

    predicted_states = np.argmax(proba_z, axis=1) + 1
    print(f"Predicted states for {len(test_idx)} held-out entities:")
    print(predicted_states)


if __name__ == "__main__":
    main()
