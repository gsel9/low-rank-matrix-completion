"""Test functionality of the GraphConvRNN factor model."""

import numpy as np

from lrmc.factor_model import GraphConvRNN


X = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [3, 4, 3],
        [3, 2, 3],
    ],
    dtype=float,
)


def _build_model(
    rank=2, n_iter=3, k_neighbors=2, n_channels=4, poly_order=3, early_stopping=False
):
    return GraphConvRNN(
        rank=rank,
        n_iter=n_iter,
        k_neighbors=k_neighbors,
        n_channels=n_channels,
        poly_order=poly_order,
        early_stopping=early_stopping,
    )


class TestGraphConvRNN:
    def test_init_matrices_hasattr(self):
        model = _build_model()
        model._init_matrices(X)

        for attr in ("X", "U", "V", "W", "L_row", "L_col", "network"):
            assert hasattr(model, attr)

    def test_init_matrices_shapes(self):
        rank = 2
        model = _build_model(rank=rank)
        model._init_matrices(X)

        assert model.U.shape == (X.shape[0], rank)
        assert model.V.shape == (X.shape[1], rank)
        assert model.W.shape == X.shape

    def test_M_shape(self):
        model = _build_model()
        model._init_matrices(X)

        assert model.M.shape == X.shape

    def test_loss_is_finite(self):
        model = _build_model()
        model._init_matrices(X)

        assert np.isfinite(model.loss())

    def test_run_step_does_not_modify_n_iter(self):
        "test run step counter not modified by subclass"
        model = _build_model()
        model._init_matrices(X)

        assert model.n_iter_ is None
        model.run_step()
        assert model.n_iter_ is None

    def test_run_step_changes_network_weights(self):
        model = _build_model()
        model._init_matrices(X)

        weights_before = [
            w.numpy().copy() for w in model.network.trainable_variables
        ]
        model.run_step()
        weights_after = model.network.trainable_variables

        assert any(
            not np.allclose(before, after.numpy())
            for before, after in zip(weights_before, weights_after)
        )

    def test_fit_tracks_losses_and_n_iter(self):
        n_iter = 3
        model = _build_model(n_iter=n_iter)
        model.fit(X, verbose=0)

        assert model.n_iter_ == n_iter
        assert len(model.losses_) == n_iter

    def test_persistent_input(self):
        """Models are expected to leave input variables like X unmodified."""

        X_initial = X.copy()

        model = _build_model()
        model.fit(X, verbose=0)

        assert np.array_equal(X, X_initial)

    def test_score_returns_finite_value(self):
        model = _build_model()
        model.fit(X, verbose=0)

        assert np.isfinite(model.score(X))
