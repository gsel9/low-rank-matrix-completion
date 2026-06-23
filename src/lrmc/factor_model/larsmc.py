# third party
import numpy as np
from sklearn.linear_model import LassoLars

# local
from .cmc import CMC


class LarsMC(CMC):
    r"""Computes Lasso path using LARS algorithm for sparsity in female-
    specific coefficients.

    .. math::
       \min F(\mathbf{U}, \mathbf{V}) + R(\mathbf{U}, \mathbf{V})

    Computes the Lasso path using the LARS algorithm.

    """

    def __init__(
        self,
        rank,
        W=None,
        n_iter=100,
        alpha=1e-20,
        n_iter_U=100,
        gamma=1.0,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        random_state=42,
        missing_value=0,
    ):
        super().__init__(
            rank=rank,
            W=W,
            n_iter=n_iter,
            gamma=gamma,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            random_state=random_state,
            missing_value=missing_value,
        )

        self.alpha = alpha
        self.n_iter_U = n_iter_U

        self.alphas = None

    def _update_U(self):
        # Expected input is y = Wx. With model S = UV.T this translates
        # to y = S.T, w = U.T, X = V.
        reg = LassoLars(
            alpha=self.alpha,
            fit_path=False,
            fit_intercept=False,
            max_iter=self.n_iter_U,
        ).fit(self.V, self.S.T)

        self.U = reg.coef_
        self.alphas = np.squeeze(reg.alphas_)

    def loss(self):
        "Evaluate the optimization objective"

        residual = self.mask * (self.X - self.U @ self.V.T)
        loss = np.square(np.linalg.norm(residual))
        loss += sum(self.alphas * np.linalg.norm(self.U, ord=1, axis=1))
        loss += self.lambda2 * np.square(np.linalg.norm(self.V - self.J))
        loss += self.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

        return loss
