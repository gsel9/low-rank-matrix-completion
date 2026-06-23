import numpy as np
from hypothesis import given, strategies
from lrmc.utils import (  # theta_mle,
    basis_baseline_value,
    finite_difference_matrix,
    laplacian_kernel_matrix,
)


# TODO: find a good test strategy
def _test_theta_mle():
    pass


@given(strategies.integers(min_value=10, max_value=100))
def test_finite_difference_matrix(T):
    D = finite_difference_matrix(T)
    assert np.isclose(np.sum(D, axis=1).sum(), 0)


@given(strategies.integers(min_value=10, max_value=100))
def test_laplacian_kernel_matrix(T):
    C = laplacian_kernel_matrix(T, gamma=0)
    assert np.all(np.isclose(C, 1))


@given(
    strategies.integers(min_value=10, max_value=100),
    strategies.integers(min_value=10, max_value=100),
    strategies.integers(min_value=10, max_value=100),
)
def test_basis_baseline_value(n_rows, n_cols, min_value):
    J = basis_baseline_value((n_rows, n_cols), min_value)
    assert np.all(np.isclose(J, min_value))
