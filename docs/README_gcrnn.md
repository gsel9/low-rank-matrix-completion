# Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks

`GraphConvRNN` implements the method described in *Geometric Matrix Completion
with Recurrent Multi-Graph Neural Networks* (Monti, Bronstein and Bresson,
NIPS 2017). It follows the same `MatrixCompletionBase` API as the other
factorization models in this package (`fit`, `M`, `score`, `transform`).

Rather than directly factorizing $X \approx UV^\top$ with alternating least
squares, the row and column factors $U \in \mathbb{R}^{N \times r}$ and
$V \in \mathbb{R}^{T \times r}$ are refined by a multi-graph CNN coupled to an
LSTM:

1. A Chebyshev polynomial spectral graph convolution filters $U$ (resp. $V$)
   using precomputed Chebyshev terms of the row (resp. column) similarity
   graph Laplacian.
2. The filtered signal drives one LSTM step per factor, producing an updated
   hidden state $h$ and cell state $c$.
3. The cell state is mapped back to an additive update $\Delta U$
   (resp. $\Delta V$) via a `tanh` output layer, and $U, V$ are incremented:
   $U \leftarrow U + \Delta U$, $V \leftarrow V + \Delta V$.

These three steps are repeated for `n_diffusion_steps`, after which the
completed matrix is reconstructed as $M = U'V'^\top$. Unlike the other models
in this package, it is the diffusion network's weights — not $U$ and $V$
directly — that are fit. Each call to `fit` runs one step of gradient descent
(`run_step`) on a masked reconstruction loss plus row/column graph-trace
regularization terms that penalize reconstructions which are non-smooth with
respect to the row/column similarity graphs.

## Usage

```python
import numpy as np
from sklearn.metrics import mean_squared_error

from lrmc.factor_model import GraphConvRNN

X = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [3, 4, 3],
    [3, 2, 3],
], dtype=float)

model = GraphConvRNN(rank=2, n_iter=200, k_neighbors=2, n_channels=16)
model.fit(X)

score = mean_squared_error(X, model.M)
```

## Parameters

* `rank` — number of latent factors in `U` and `V`.
* `n_iter` — number of outer gradient-descent steps used to fit the
  diffusion network's weights.
* `n_channels` — number of hidden channels used by the spectral convolution
  and LSTM layers.
* `poly_order` — order of the Chebyshev polynomial expansion of the row and
  column graph Laplacians.
* `n_diffusion_steps` — number of diffusion (convolution + LSTM) steps
  applied to `U` and `V` per forward pass.
* `k_neighbors` — number of neighbors used to build the row and column
  similarity graphs (see `utils.weighted_adjacency_matrix`).
* `learning_rate` — learning rate for the Adam optimizer used to fit the
  diffusion network's weights.
* `lambda1` — weight of the masked reconstruction loss.
* `lambda2` — weight of the row graph-trace regularization term.
* `lambda3` — weight of the column graph-trace regularization term.
* `random_state` — seed for the factor initialization and the network's
  `GlorotUniform` weight initializer.
