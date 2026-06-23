# third party
import numpy as np

# local
from .. import utils

from ._base import MatrixCompletionBase


class TVMC(MatrixCompletionBase):
    r"""Total variation regularization.

    Based on the Chambolle Pock algorithm.

    .. math::
       \min F(\mathbf{U}, \mathbf{V}) + R(\mathbf{U}, \mathbf{V})

    Args:
        MFBase (_type_): _description_
    """

    def __init__(
        self,
        rank,
        W=None,
        n_iter=100,
        n_iter_V=100,
        zeta=0.5,
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
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            random_state=random_state,
            missing_value=missing_value,
        )
        self.n_iter_V = n_iter_V
        self.sigma = zeta
        self.tau = zeta

    def _init_matrices(self, X):
        self.X = X
        self.S = self.X.copy()
        self.mask = (self.X != 0).astype(np.float32)

        self.N, self.T = np.shape(X)
        self.nz_rows, self.nz_cols = np.nonzero(self.X)

        self.V = self.init_basis()
        self.U = self.init_coefs()

        # R is the finite-difference matrix used in the TV penalty ||RV||_1
        self.R = utils.finite_difference_matrix(self.T)

        # the minimum value for the basic profiles
        min_value = np.min(self.X[self.X != self.missing_value])
        self.J = utils.basis_baseline_value(self.V.shape, min_value)

        if self.W is None:
            self.W = self.identity_weights()

        self.I_l1 = self.lambda1 * np.identity(self.r)
        self.I_l2 = self.lambda2 * np.identity(self.r)

        self.Ir = np.eye(self.r) * (1 + self.lambda2 * self.tau)

    # TODO: derive algorithm again with norm(V - J)
    def _update_V(self):
        # NOTE: If self.n_iter_ > 0: uses solutions from previous run in
        # initialisation. Re-initialising dual variable with zeros gives
        # best performance.

        # dual variable
        self.Y = np.zeros_like(self.V)

        # define auxillary variables
        V_bar = self.V
        SU = self.tau * self.S.T @ self.U
        H = np.linalg.inv(self.Ir + self.tau * self.U.T @ self.U)

        # eval relative primal and dual residuals < tol for convergence
        for _ in range(self.n_iter_V):
            # solve for dual variable
            self.Y = self.project_inf_ball(
                self.Y + self.sigma * self.R @ V_bar
            )

            # solve for primal variable
            V_next = (SU + self.V - self.tau * self.R.T @ self.Y) @ H

            # NOTE: Using theta = 1.
            V_bar = 2 * V_next - self.V

            self.V = V_next

    def project_inf_ball(self, X):
        """Projecting X onto the infininty ball of radius lambda3
        amounts to element-wise clipping at +/- radius given by
        ```python
            np.minimum(np.abs(X), radius) * np.sign(X)
        ```
        """
        return np.clip(X, a_min=-1.0 * self.lambda3, a_max=self.lambda3)

    def _update_U(self):
        U = np.linalg.solve(
            self.V.T @ self.V + self.lambda1 * np.identity(self.r),
            self.V.T @ self.S.T,
        )
        self.U = np.transpose(U)

    def _update_S(self):
        self.S = self.U @ self.V.T
        self.S[self.nz_rows, self.nz_cols] = self.X[self.nz_rows, self.nz_cols]

    def loss(self):
        "Evaluate the optimization objective"

        # Updates to S occurs only at validation scores so must compare
        # against U, V.
        loss = np.square(
            np.linalg.norm(self.mask * (self.X - self.U @ self.V.T))
        )
        loss += self.lambda1 * np.square(np.linalg.norm(self.U))
        loss += self.lambda2 * np.square(np.linalg.norm(self.V - self.J))
        loss += self.lambda3 * np.linalg.norm(self.R @ self.V, ord=1)

        return loss

    def run_step(self):
        "Run one step of alternating minimization"

        self._update_U()
        self._update_V()
        self._update_S()
