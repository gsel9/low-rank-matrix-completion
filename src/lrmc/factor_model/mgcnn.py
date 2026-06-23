import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from .. import utils

from ._base import MatrixCompletionBase


import tensorflow as tf

class GraphConvRNN(tf.keras.Model):
    """
    Based on the methods described in Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks (in Proc. NIPS, 2017)
Federico Monti, Michael M. Bronstein, Xavier Bresson
    """
    def __init__(self, n_channels, poly_order, U_dim, V_dim, cheby_row, cheby_col, n_iter=4, seed=42):
        super().__init__()

        self.n_channels = n_channels
        self.poly_order = poly_order
        self.U_dim = U_dim
        self.V_dim = V_dim
        self.cheby_row = cheby_row
        self.cheby_col = cheby_col
        self.n_iter = n_iter

        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

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

        # --- Recurrent (U) ---
        def rnn_weights(prefix):
            return {
                "W_f": self.add_weight((n_channels, n_channels), initializer=initializer, name=f"W_f_{prefix}"),
                "W_i": self.add_weight((n_channels, n_channels), initializer=initializer, name=f"W_i_{prefix}"),
                "W_o": self.add_weight((n_channels, n_channels), initializer=initializer, name=f"W_o_{prefix}"),
                "W_c": self.add_weight((n_channels, n_channels), initializer=initializer, name=f"W_c_{prefix}"),
                "U_f": self.add_weight((n_channels, n_channels), initializer=initializer, name=f"U_f_{prefix}"),
                "U_i": self.add_weight((n_channels, n_channels), initializer=initializer, name=f"U_i_{prefix}"),
                "U_o": self.add_weight((n_channels, n_channels), initializer=initializer, name=f"U_o_{prefix}"),
                "U_c": self.add_weight((n_channels, n_channels), initializer=initializer, name=f"U_c_{prefix}"),
                "b_f": self.add_weight((n_channels,), initializer="zeros", name=f"b_f_{prefix}"),
                "b_i": self.add_weight((n_channels,), initializer="zeros", name=f"b_i_{prefix}"),
                "b_o": self.add_weight((n_channels,), initializer="zeros", name=f"b_o_{prefix}"),
                "b_c": self.add_weight((n_channels,), initializer="zeros", name=f"b_c_{prefix}")
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
        return M
    

def train():

    def compute_loss(X_tf, M_tf, L_row_tf, L_col_tf, mask_tf):
        loss_acc = tf.norm(mask_tf * (X_tf - M_tf))**2 / tf.reduce_sum(mask_tf)
        loss_trace_row = tf.linalg.trace(tf.transpose(M_tf) @ L_row_tf @ M_tf)
        loss_trace_col = tf.linalg.trace(M_tf @ L_col_tf @ tf.transpose(M_tf))
        return loss_acc + loss_trace_row + loss_trace_col

    optimizer = tf.keras.optimizers.Adam()

    n_train_epochs = 1000
    for epoch in range(n_train_epochs):

        with tf.GradientTape() as tape:
            M_tf = model((U_tf, V_tf))
            loss = compute_loss(X_tf, M_tf, L_row_tf, L_col_tf, mask_tf)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 10 == 0:
            print(f"Step {epoch}, Loss: {loss.numpy():.4f}")
    

if __name__ == "__main__":
    import numpy as np 
    import scipy.sparse as sp

    # Data matrix
    X = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [3, 4, 3],
        [3, 2, 3],
    ])

    U = np.random.random((X.shape[0], 2))
    V = np.random.random((X.shape[1], 2))
    M = U @ V.T

    S_row = utils.weighted_adjacency_matrix(X, k=2)
    S_col = utils.weighted_adjacency_matrix(np.transpose(X), k=2)

    # Normalized Laplacians (L = I - D^{-1/2} W D^{-1/2} of similarity graphs
    L_row = sp.csgraph.laplacian(S_row, normed=True)
    L_col = sp.csgraph.laplacian(S_col, normed=True)

    # Rescaled Laplacian with all eigenvalues in [-1, 1]
    L_row_norm = L_row - np.eye(L_row.shape[0])
    L_col_norm = L_col - np.eye(L_col.shape[0])

    poly_order = 3
    cheby_row = utils.chebyshev_polynomials(L_row_norm, poly_order)
    cheby_col = utils.chebyshev_polynomials(L_col_norm, poly_order)

    model = GraphConvRNN(
        n_channels=9,
        poly_order=poly_order,
        U_dim=U.shape[1],
        V_dim=V.shape[1],
        cheby_row=cheby_row,
        cheby_col=cheby_col
    )

    U_tf = tf.constant(U, dtype=tf.float32)
    V_tf = tf.constant(V, dtype=tf.float32)

    M_tf = model((U_tf, V_tf))