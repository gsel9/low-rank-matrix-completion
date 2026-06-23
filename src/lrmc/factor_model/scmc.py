# from typing import cast

import numpy as np
import tensorflow as tf
from .. import utils

from numpy.lib.stride_tricks import as_strided

from ._base import MatrixCompletionBase


def _custom_roll(arr, m):
    """Roll array elements with different amount per axis.

    Fast implementation of row wise shifting.

    Arguments:
    arr: two-dimensional array
    m: one dimensional list of integers, each corresponding to a shift
       of a row in arr

    NB! For very large shifts, floating point errors may cause the wrong
    results.
    """

    arr_roll = arr[:, [*range(arr.shape[1]), *range(arr.shape[1] - 1)]]
    strd_0, strd_1 = arr_roll.strides
    n = arr.shape[1]
    # config as_strided with writable=False to avoid accidentally writing to
    # the memory and corrupting the data (recommended in docs)
    result = as_strided(
        arr_roll, (*arr.shape, n), (strd_0, strd_1, strd_1), writeable=False
    )

    return result[np.arange(arr.shape[0]), (n - m) % n].astype(arr.dtype)


def _take_per_row_strided(A, start_idx, n_elem):
    """Select n_elem per row with a shift start_idx.

    Fast implementation of selection from row wise shifted sample.
    Rows are not wrapped around, i.e. if start_idx + n_elem is larger than
    the number of columns, out of range is thrown.

    That is
    >>> def simple_row_strided(A, shift_array, number_elements):
    >>>     strided_A = np.empty((A.shape[0], number_elements))
    >>>     for i in range(A.shape[0]):
    >>>         strided_A[i] = A[i, shift_array[i]:shift_array[i]+number_elements]
    """

    m, n = np.shape(A)
    A.shape = -1
    s0 = A.strides[0]
    l_indx = start_idx + n * np.arange(len(start_idx))
    out = as_strided(
        A, (len(A) - n_elem + 1, n_elem), (s0, s0), writeable=False
    )[l_indx]
    A.shape = m, n

    return out


class SCMC(MatrixCompletionBase):
    """Shifted matrix factorization with L2 and convolutional regularization
    (optional).

    Factor updates are based on gradient descent approximations, permitting an
    arbitrary weight matrix in the discrepancy term. The shift mechanism will
    maximize the correlation between vector samples in the original and
    estimated data matrices for more accurate factor estimates.

    Args:
        X: Sparse data matrix used to estimate factor matrices
        V: Initial estimate for basic vectors
        shift_budget: List of integer shift candidates to evaluate per row.

    Discussion on internal matrices:
        There are four X matrices (correspondingly for W):

        - X : The original input matrix
        - X_bc : The original input matrix with padded zeros on the time axis.
        - X_shifted : Similar to X_bc, but each row is shifted according to
            self.s.
        - X_shifts : A stack of size(s_budget) arrays similar to X_bc, but
            each shifted horizontally (time axis). Stack layer i is shifted
            s_budget[i].

        X_shifted is the only matrix altered after initialization.

    """

    def __init__(
        self,
        rank,
        shift_budget,
        W=None,
        n_iter=100,
        gamma=1.0,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        learning_rate=1e-3,
        iter_V=100,
        iter_U=100,
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
        self.s_budget = shift_budget
        self.learning_rate = learning_rate
        self.iter_V = iter_V
        self.iter_U = iter_U

    def _init_matrices(self, X):
        self.N, self.T = np.shape(X)
        # the shift amount per row
        self.s = np.zeros(self.N, dtype=int)
        # the number of possible shifts. used for padding of arrays
        self.Ns = len(self.s_budget)

        # add time points to cover extended left and right boundaries
        K = utils.finite_difference_matrix(self.T + 2 * self.Ns)
        D = utils.laplacian_kernel_matrix(self.T + 2 * self.Ns, self.gamma)
        self.KD = K @ D

        self.I1 = self.lambda1 * np.identity(self.r)
        self.I2 = self.lambda2 * np.identity(self.r)

        # Expand matrices with zeros over the extended left and right
        # boundaries.
        self.X_bc = np.hstack(
            [np.zeros((self.N, self.Ns)), X, np.zeros((self.N, self.Ns))]
        )
        V = self.init_basis()
        self.V_bc = np.vstack(
            [
                np.zeros((self.Ns, self.r)),
                V,
                np.zeros((self.Ns, self.r)),
            ]
        )
        # the minimum value for the basic profiles
        min_value = np.min(X[X != self.missing_value])
        self.J = utils.basis_baseline_value(self.V_bc.shape, min_value)

        if self.W is None:
            self.W = np.zeros_like(X)
            self.W[X != self.missing_value] = 1

        # Build W_bc before using it
        self.W_bc = np.hstack(
            [
                np.zeros((self.N, self.Ns)),
                self.W,
                np.zeros((self.N, self.Ns)),
            ]
        )

        # Implementation shifts W and X (not UV.T)
        self.X_shifted = self.X_bc.copy()
        self.W_shifted = self.W_bc.copy()
        self._fill_boundary_regions_V_bc()

        # Placeholders (s x N x T+2*Ns) for all possible candidate shifts
        self.X_shifts = np.empty((self.Ns, *self.X_bc.shape))
        self.W_shifts = np.empty((self.Ns, *self.W_bc.shape))

        # Shift X in opposite direction of V shift.
        for j, s_n in enumerate(self.s_budget):
            self.X_shifts[j] = np.roll(self.X_bc, -1 * s_n, axis=1)
            self.W_shifts[j] = np.roll(self.W_bc, -1 * s_n, axis=1)

        self.U = self._exactly_solve_U()

    @property
    def X(self):
        return _take_per_row_strided(
            self.X_shifted, self.Ns - self.s, n_elem=self.T
        )

    @property
    def V(self):
        """To be compatible with the expectation of having a V"""
        return self.V_bc

    @property
    def M(self):
        # Compute the reconstructed matrix with sample-specific shifts
        M = _take_per_row_strided(
            self.U @ self.V_bc.T, start_idx=self.Ns - self.s, n_elem=self.T
        )

        return np.array(M, dtype=np.float32)

    def _shift_X_W(self):
        self.X_shifted = _custom_roll(self.X_bc, -1 * self.s)
        self.W_shifted = _custom_roll(self.W_bc, -1 * self.s)

    def _fill_boundary_regions_V_bc(self):
        """Extrapolate the edge values in V_bc over the extended boundaries"""

        V_filled = np.zeros_like(self.V_bc)

        idx = np.arange(self.T + 2 * self.Ns)
        for i, v in enumerate(self.V_bc.T):
            v_left = v[idx <= int(self.T / 2)]
            v_right = v[idx > int(self.T / 2)]

            v_left[v_left == 0] = v_left[np.argmax(v_left != 0)]
            v_right[v_right == 0] = v_right[
                np.argmax(np.cumsum(v_right != 0))
            ]

            V_filled[:, i] = np.concatenate([v_left, v_right])

        self.V_bc = V_filled

    def _update_V(self):
        V = tf.Variable(self.V_bc, dtype=tf.float32)

        W_shifted = tf.cast(self.W_shifted, dtype=tf.float32)
        X_shifted = tf.cast(self.X_shifted, dtype=tf.float32)
        U = tf.cast(self.U, dtype=tf.float32)
        J = tf.cast(self.J, dtype=tf.float32)
        KD = tf.cast(self.KD, dtype=tf.float32)

        def _loss_V():
            frob_tensor = tf.multiply(
                W_shifted, X_shifted - (U @ tf.transpose(V))
            )
            frob_loss = tf.reduce_sum(frob_tensor**2)

            l2_loss = self.lambda2 * tf.reduce_sum((V - J) ** 2)
            conv_loss = self.lambda3 * tf.reduce_sum(
                (tf.matmul(KD, V) ** 2)
            )

            return frob_loss + l2_loss + conv_loss

        optimiser = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        for _ in tf.range(self.iter_V):
            optimiser.minimize(_loss_V, [V])

        self.V_bc = V.numpy()

    def _exactly_solve_U(self):
        """Solve for U at a fixed V.

        V is assumed to be initialized."""
        U = np.empty((self.N, self.r))

        for n in range(self.N):
            U[n] = (
                self.X_shifted[n]
                @ self.V_bc
                @ np.linalg.inv(
                    self.V_bc.T
                    @ (np.diag(self.W_shifted[n]) @ self.V_bc)
                    + self.I1
                )
            )
        return U

    def _approx_U(self):
        U = tf.Variable(self.U, dtype=tf.float32)

        W_shifted = tf.cast(self.W_shifted, dtype=tf.float32)
        X_shifted = tf.cast(self.X_shifted, dtype=tf.float32)
        V_bc = tf.cast(self.V_bc, dtype=tf.float32)

        def _loss_U():
            frob_tensor = tf.multiply(
                W_shifted,
                X_shifted - tf.matmul(U, V_bc, transpose_b=True),
            )
            frob_loss = tf.reduce_sum((frob_tensor) ** 2)

            return frob_loss + self.lambda1 * tf.reduce_sum(U**2)

        optimiser = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        for _ in tf.range(self.iter_U):
            optimiser.minimize(_loss_U, [U])

        return U.numpy()

    def _update_s(self):
        # Evaluate the discrepancy term for all possible shift candidates
        M = self.U @ self.V_bc.T
        D = (
            np.linalg.norm(
                self.W_shifts * (self.X_shifts - M[None, :, :]), axis=-1
            )
            ** 2
        )

        # Selected shifts minimize the discrepancy
        s_new = [self.s_budget[i] for i in np.argmin(D, axis=0)]

        # Update attributes only if changes to the optimal shift
        if not np.array_equal(self.s, s_new):
            self.s = np.array(s_new)
            self._shift_X_W()

    def run_step(self):
        "Perform one step of alternating minimization"

        self.U = self._approx_U()
        self._update_V()
        self._update_s()

    def loss(self):
        "Evaluate the optimization objective"

        loss = np.sum(
            np.linalg.norm(
                self.W_shifted
                * (self.X_shifted - self.U @ self.V_bc.T),
                axis=1,
            )
            ** 2
        )
        loss += self.lambda1 * np.square(np.linalg.norm(self.U))
        loss += self.lambda2 * np.square(
            np.linalg.norm(self.V_bc - self.J)
        )
        loss += self.lambda3 * np.square(
            np.linalg.norm(self.KD @ self.V_bc)
        )

        return loss
