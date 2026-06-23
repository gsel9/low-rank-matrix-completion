import numpy as np

from .. import utils

from ._base import MatrixCompletionBase


class CMC(MatrixCompletionBase):
    r"""Matrix factorization with L2 and convolutional regularization.
    Factor updates are based on analytical estimates and will therefore
    not permit and arbitrary weight matrix in the discrepancy term. Here is
    inplace math :math:`r_e`.

    .. math::
       \min F(\mathbf{U}, \mathbf{V}) + R(\mathbf{U}, \mathbf{V})

    The alternating minimization is based on the exact solution to the
    minimizers :math:`\partialF/\partial U` and :math:`\partialF/\partial V`.

    Args:
            X: Sparse data matrix used to estimate factor matrices
    """

    def __init__(
        self,
        rank,
        W=None,
        n_iter=100,
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
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            random_state=random_state,
            missing_value=missing_value,
        )
        self.gamma = gamma

    def _init_matrices(self, X):
        self.X = X
        self.S = self.X.copy()
        self.mask = (self.X != 0).astype(np.float32)

        self.N, self.T = np.shape(X)
        self.nz_rows, self.nz_cols = np.nonzero(X)

        self.V = self.init_basis()
        self.U = self.init_coefs()

        K = utils.finite_difference_matrix(self.T)
        D = utils.laplacian_kernel_matrix(self.T, self.gamma)

        # pre-compute static variables
        self.KD = K @ D
        self.DTKTKD = self.KD.T @ self.KD
        self.L2, self.Q2 = np.linalg.eigh(self.lambda3 * self.DTKTKD)

        # the minimum value for the basic profiles
        min_value = np.min(self.X[self.X != self.missing_value])
        self.J = utils.basis_baseline_value(self.V.shape, min_value)

        if self.W is None:
            self.W = self.identity_weights()

        self.I_l1 = self.lambda1 * np.identity(self.r)
        self.I_l2 = self.lambda2 * np.identity(self.r)

    def _update_V(self):
        L1, Q1 = np.linalg.eigh(self.U.T @ self.U + self.I_l2)

        hatV = (
            (self.Q2.T @ (self.S.T @ self.U + self.lambda2 * self.J))
            @ Q1
            / np.add.outer(self.L2, L1)
        )
        self.V = self.Q2 @ (hatV @ Q1.T)

    def _update_U(self):
        self.U = np.transpose(
            np.linalg.solve(self.V.T @ self.V + self.I_l1, self.V.T @ self.S.T)
        )

    def _update_S(self):
        self.S = self.U @ self.V.T
        self.S[self.nz_rows, self.nz_cols] = self.X[self.nz_rows, self.nz_cols]

    def loss(self):
        "Evaluate the optimization objective"

        loss = np.square(np.linalg.norm(self.mask * (self.X - self.U @ self.V.T)))
        loss += self.lambda1 * np.square(np.linalg.norm(self.U))
        loss += self.lambda2 * np.square(np.linalg.norm(self.V - self.J))
        loss += self.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

        return loss

    def run_step(self):
        "Run one step of alternating minimization"

        self._update_U()
        self._update_V()
        self._update_S()
