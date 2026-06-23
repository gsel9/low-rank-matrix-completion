"""Test functionality of CMC factor model."""

import unittest

import numpy as np
from hypothesis import given, strategies
from hypothesis.extra import numpy as nps
from lrmc.factor_model import CMC


class TestCMC(unittest.TestCase):
    def _init_test_model(self, rank, data):
        array = data.draw(
            nps.arrays(
                float,
                nps.array_shapes(min_dims=2, max_dims=2),
                elements=strategies.floats(1, 4),
            )
        )

        model = CMC(rank=rank)
        model._init_matrices(array)

        return model

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_init_matrices_hasattr(self, rank, data):
        model = self._init_test_model(rank, data)

        for attr in ("X", "U", "V", "W", "S"):
            assert hasattr(model, attr)

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_init_matrices_W(self, rank, data):
        model = self._init_test_model(rank, data)

        assert np.all(model.W == 1)

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_loss(self, rank, data):
        model = self._init_test_model(rank, data)

        assert model.loss() > 0

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_update_U(self, rank, data):
        model = self._init_test_model(rank, data)
        model._update_U()

        assert model.U.shape == (model.N, rank)

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_update_V(self, rank, data):
        model = self._init_test_model(rank, data)
        model._update_V()

        assert model.V.shape == (model.T, rank)

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_update_S(self, rank, data):
        model = self._init_test_model(rank, data)
        model._update_S()

        R, C = model.nz_rows, model.nz_cols
        assert np.isclose(np.sum(model.X[R, C] - model.S[R, C]), 0)

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_run_step(self, rank, data):
        "test run step counter not modified by subclass"
        model = self._init_test_model(rank, data)
        assert model.n_iter_ is None
        model.run_step()
        assert model.n_iter_ is None

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_persistent_input(self, rank, data):
        """Test that models do not modify their input arguments.

        Models are expected to leave input variables like X, W, s_budged unmodified
        unless otherwise specified.
        >>> model = WCMF(X, V, W)
        >>> model.run_step()
        model.X should be the same as supplied during initialization.
        """

        array = data.draw(
            nps.arrays(
                float,
                nps.array_shapes(min_dims=2, max_dims=2),
                elements=strategies.floats(1, 4),
            )
        )

        array_initial = array.copy()

        model = CMC(rank=rank)
        model._init_matrices(array)

        assert np.array_equal(model.X, array_initial)
        model.run_step()
        assert np.array_equal(model.X, array_initial)
