"""Base class for matrix completion algorithms.
"""

from abc import ABC

import numpy as np
from sklearn.metrics import mean_squared_error

from ..convergence import FactorConvergence


class MatrixCompletionBase(ABC):
    "Base class for matrix factorization algorithms."

    def __init__(
        self,
        rank,
        W=None,
        n_iter=100,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        random_state=42,
        missing_value=0,
        early_stopping=True,
    ):
        self.r = rank
        self.W = W
        self.n_iter = n_iter
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.random_state = random_state
        self.missing_value = missing_value
        self.early_stopping = early_stopping

        self.n_iter_ = None
        self.losses_ = None

    def _init_matrices(self, X):
        raise NotImplementedError("Should implement _init_matrices")

    def loss(self):
        raise NotImplementedError("Should implement loss")

    def run_step(self):
        raise NotImplementedError("Should implement run_step")

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # TODO: Must account for potential shift matrix
    @property
    def M(self):
        return np.array(self.U @ self.V.T, dtype=np.float32)

    def identity_weights(self):
        W = np.zeros_like(self.X)
        W[self.X != self.missing_value] = 1

        return W

    def init_basis(self):
        rnd = np.random.RandomState(self.random_state)
        return rnd.normal(size=(self.T, self.r))

    def init_coefs(self):
        rnd = np.random.RandomState(self.random_state)
        return rnd.normal(size=(self.N, self.r))

    def transform(self, X, y=None):
        # estimate least-squares coefficients from shared basic profiles
        U_star = (X @ self.V) @ np.linalg.inv(self.V.T @ self.V)
        return U_star @ self.V.T

    def score(self, X, y=None):
        """Return negative MSE on observed entries (higher is better)."""
        mask = (X != self.missing_value).astype(float)
        X_obs = X * mask
        M_obs = self.M * mask
        return -1.0 * mean_squared_error(X_obs.ravel(), M_obs.ravel())

    def fit(self, X, y=None, verbose=1):
        """Run matrix completion on input matrix X using a factorization
        model."""

        # re-initialize attributes
        self.n_iter_ = 0
        self.losses_ = []

        self._init_matrices(X)

        if self.early_stopping:
            monitor = FactorConvergence(verbose=verbose)

        for _ in range(self.n_iter):
            self.run_step()
            self.losses_.append(self.loss())

            if self.early_stopping and monitor.is_converged(self.M):
                break

            self.n_iter_ += 1
