"""Geometric matrix completion with a recurrent multi-graph CNN.

Based on the method described in Geometric Matrix Completion with Recurrent
Multi-Graph Neural Networks (in Proc. NIPS, 2017), Federico Monti, Michael M.
Bronstein, Xavier Bresson.
"""

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

try:
    import keras
    _keras = keras
except ImportError:
    _keras = tf.keras

from .. import utils

from ._base import MatrixCompletionBase


class _DiffusionNetwork(_keras.Model):
    """Multi-graph CNN coupled to an LSTM that diffuses U and V towards a
    completed matrix M = U V^T."""

    def __init__(self, n_channels, poly_order, U_dim, V_dim, cheby_row, cheby_col, n_iter=4, seed=42):
        super().__init__()

        self.n_channels = n_channels
        self.poly_order = poly_order
        self.U_dim = U_dim
        self.V_dim = V_dim
        self.cheby_row = cheby_row
        self.cheby_col = cheby_col
        self.n_iter = n_iter

        initializer = _keras.initializers.GlorotUniform(seed=seed)

        # --- Convolution weights ---
        self.W_conv_U = self.add_weight(
            shape=(poly_order * U_dim, n_channels),
            initializer=initializer,
            name="W_conv_U"
        )
        self.W_conv_V = self.add_weight(
            shape=(poly_order * V_dim, n_channels),
            initializer=initializer,
            name="W_conv_V"
        )

        self.b_conv_U = self.add_weight(shape=(n_channels,), initializer="zeros", name="b_conv_U")
        self.b_conv_V = self.add_weight(shape=(n_channels,), initializer="zeros", name="b_conv_V")

        # --- Recurrent (U, V) ---
        def rnn_weights(prefix):
            shape = (n_channels, n_channels)
            return {
                "W_f": self.add_weight(shape=shape, initializer=initializer, name=f"W_f_{prefix}"),
                "W_i": self.add_weight(shape=shape, initializer=initializer, name=f"W_i_{prefix}"),
                "W_o": self.add_weight(shape=shape, initializer=initializer, name=f"W_o_{prefix}"),
                "W_c": self.add_weight(shape=shape, initializer=initializer, name=f"W_c_{prefix}"),
                "U_f": self.add_weight(shape=shape, initializer=initializer, name=f"U_f_{prefix}"),
                "U_i": self.add_weight(shape=shape, initializer=initializer, name=f"U_i_{prefix}"),
                "U_o": self.add_weight(shape=shape, initializer=initializer, name=f"U_o_{prefix}"),
                "U_c": self.add_weight(shape=shape, initializer=initializer, name=f"U_c_{prefix}"),
                "b_f": self.add_weight(shape=(n_channels,), initializer="zeros", name=f"b_f_{prefix}"),
                "b_i": self.add_weight(shape=(n_channels,), initializer="zeros", name=f"b_i_{prefix}"),
                "b_o": self.add_weight(shape=(n_channels,), initializer="zeros", name=f"b_o_{prefix}"),
                "b_c": self.add_weight(shape=(n_channels,), initializer="zeros", name=f"b_c_{prefix}"),
            }

        self.rnn_u = rnn_weights("u")
        self.rnn_v = rnn_weights("v")

        # --- Output layers ---
        self.W_out_U = self.add_weight(
            shape=(n_channels, U_dim),
            initializer=initializer,
            name="W_out_U"
        )
        self.W_out_V = self.add_weight(
            shape=(n_channels, V_dim),
            initializer=initializer,
            name="W_out_V"
        )

        self.b_out_U = self.add_weight(shape=(1,), initializer="zeros", name="b_out_U")
        self.b_out_V = self.add_weight(shape=(1,), initializer="zeros", name="b_out_V")

    def spectral_convolution(self, cheby_terms, Z, W, b):
        outputs = [C @ Z for C in cheby_terms]
        outputs = tf.concat(outputs, axis=1)
        return tf.nn.relu(tf.matmul(outputs, W) + b)

    def rnn_step(self, X_filtered, h, c, weights):
        W_f, W_i, W_o, W_c = weights["W_f"], weights["W_i"], weights["W_o"], weights["W_c"]
        U_f, U_i, U_o, U_c = weights["U_f"], weights["U_i"], weights["U_o"], weights["U_c"]
        b_f, b_i, b_o, b_c = weights["b_f"], weights["b_i"], weights["b_o"], weights["b_c"]

        f = tf.sigmoid(X_filtered @ W_f + h @ U_f + b_f)
        i = tf.sigmoid(X_filtered @ W_i + h @ U_i + b_i)
        o = tf.sigmoid(X_filtered @ W_o + h @ U_o + b_o)
        c_tilde = tf.tanh(X_filtered @ W_c + h @ U_c + b_c)

        c = f * c + i * c_tilde
        h = o * tf.tanh(c)

        return h, c

    def call(self, inputs):
        U_tf, V_tf = inputs  # tensors

        # Initialize states
        h_u = tf.zeros((tf.shape(U_tf)[0], self.n_channels))
        c_u = tf.zeros((tf.shape(U_tf)[0], self.n_channels))
        h_v = tf.zeros((tf.shape(V_tf)[0], self.n_channels))
        c_v = tf.zeros((tf.shape(V_tf)[0], self.n_channels))

        for _ in range(self.n_iter):
            U_filt = self.spectral_convolution(self.cheby_row, U_tf, self.W_conv_U, self.b_conv_U)
            V_filt = self.spectral_convolution(self.cheby_col, V_tf, self.W_conv_V, self.b_conv_V)

            h_u, c_u = self.rnn_step(U_filt, h_u, c_u, self.rnn_u)
            h_v, c_v = self.rnn_step(V_filt, h_v, c_v, self.rnn_v)

            delta_U = tf.tanh(c_u @ self.W_out_U + self.b_out_U)
            delta_V = tf.tanh(c_v @ self.W_out_V + self.b_out_V)

            U_tf = U_tf + delta_U
            V_tf = V_tf + delta_V

        M = tf.matmul(U_tf, V_tf, transpose_b=True)
        return U_tf, V_tf, M


class GraphConvRNN(MatrixCompletionBase):
    r"""Geometric matrix completion with a recurrent multi-graph CNN.

    The factor matrices U and V are diffused through a Chebyshev spectral
    graph convolution coupled to an LSTM, conditioned on the row and column
    similarity graphs of X. The completed matrix is reconstructed as
    M = U' V'^T, where U', V' are U, V after `n_diffusion_steps` diffusion
    steps. Unlike the alternating least-squares based models in this
    package, it is the diffusion network's weights (not U and V directly)
    that are fit, by gradient descent on a masked reconstruction loss plus
    graph-trace regularization terms that penalize reconstructions which
    are non-smooth with respect to the row/column similarity graphs.

    Based on the method described in Geometric Matrix Completion with
    Recurrent Multi-Graph Neural Networks (in Proc. NIPS, 2017), Federico
    Monti, Michael M. Bronstein, Xavier Bresson.

    Args:
        rank: Number of latent factors in U and V.
        n_channels: Number of hidden channels in the spectral convolution
            and LSTM layers.
        poly_order: Order of the Chebyshev polynomial expansion of the row
            and column graph Laplacians.
        n_diffusion_steps: Number of diffusion (convolution + LSTM) steps
            applied to U and V per forward pass.
        k_neighbors: Number of neighbors used to build the row and column
            similarity graphs.
        learning_rate: Learning rate for the Adam optimizer used to fit the
            diffusion network's weights.
        lambda1: Weight of the masked reconstruction loss.
        lambda2: Weight of the row graph-trace regularization term.
        lambda3: Weight of the column graph-trace regularization term.
    """

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
        n_channels=16,
        poly_order=3,
        n_diffusion_steps=4,
        k_neighbors=5,
        learning_rate=1e-3,
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
            early_stopping=early_stopping,
        )
        self.n_channels = n_channels
        self.poly_order = poly_order
        self.n_diffusion_steps = n_diffusion_steps
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate

    def _init_matrices(self, X):
        self.X = X
        self.N, self.T = np.shape(X)

        if self.W is None:
            self.W = self.identity_weights()

        self.V = self.init_basis()
        self.U = self.init_coefs()

        S_row = utils.weighted_adjacency_matrix(X, k=self.k_neighbors)
        S_col = utils.weighted_adjacency_matrix(X.T, k=self.k_neighbors)

        # Rescale the normalized Laplacians to have eigenvalues in [-1, 1]
        self.L_row = sp.csgraph.laplacian(S_row, normed=True) - np.eye(self.N)
        self.L_col = sp.csgraph.laplacian(S_col, normed=True) - np.eye(self.T)

        cheby_row = utils.chebyshev_polynomials(self.L_row, self.poly_order)
        cheby_col = utils.chebyshev_polynomials(self.L_col, self.poly_order)

        self.network = _DiffusionNetwork(
            n_channels=self.n_channels,
            poly_order=self.poly_order,
            U_dim=self.r,
            V_dim=self.r,
            cheby_row=cheby_row,
            cheby_col=cheby_col,
            n_iter=self.n_diffusion_steps,
            seed=self.random_state,
        )
        self.optimizer = _keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _forward(self):
        U_tf = tf.constant(self.U, dtype=tf.float32)
        V_tf = tf.constant(self.V, dtype=tf.float32)
        return self.network((U_tf, V_tf))

    def _objective(self, M):
        "Masked reconstruction loss plus row/column graph-trace regularization."

        W_tf = tf.constant(self.W, dtype=tf.float32)
        X_tf = tf.constant(self.X, dtype=tf.float32)
        L_row_tf = tf.constant(self.L_row, dtype=tf.float32)
        L_col_tf = tf.constant(self.L_col, dtype=tf.float32)

        loss_acc = self.lambda1 * tf.reduce_sum((W_tf * (X_tf - M)) ** 2) / tf.reduce_sum(W_tf)
        loss_row = self.lambda2 * tf.linalg.trace(tf.transpose(M) @ L_row_tf @ M)
        loss_col = self.lambda3 * tf.linalg.trace(M @ L_col_tf @ tf.transpose(M))

        return loss_acc + loss_row + loss_col

    def run_step(self):
        "Fit the diffusion network's weights by one step of gradient descent."

        with tf.GradientTape() as tape:
            _, _, M = self._forward()
            loss = self._objective(M)

        variables = self.network.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

    def loss(self):
        "Evaluate the optimization objective"

        _, _, M = self._forward()
        return float(self._objective(M).numpy())

    @property
    def M(self):
        _, _, M = self._forward()
        return np.array(M.numpy(), dtype=np.float32)
