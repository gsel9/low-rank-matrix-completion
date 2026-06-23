import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from lrmc.factor_model.scmc import _custom_roll, _take_per_row_strided


@given(st.data())
def test_custom_roll(data):
    "Compare the custom roll implementation to naive slow rolling"
    array = data.draw(arrays(float, array_shapes(min_dims=2, max_dims=2)))
    assume(not np.isnan(array).any())
    # Limit shifts to not be too large (1e4 arbitrarily chosen), as _custom_roll
    # is susceptible to floating point errors for large shifts.
    # This is not relevant if shifts smaller than the number
    # of time steps.
    max_abs_shift = 1e4
    shifts = data.draw(
        arrays(
            int,
            array.shape[0],
            elements=st.integers(
                min_value=-1.0 * max_abs_shift, max_value=max_abs_shift
            ),
        )
    )
    rolled = _custom_roll(array, shifts)
    for row, rolled_row, shift in zip(array, rolled, shifts):
        assert np.all(rolled_row == np.roll(row, shift))


@given(st.data())
def test_take_per_row_strided(data):
    A = data.draw(
        arrays(
            float,
            array_shapes(min_dims=2, max_dims=2, min_side=2),
            elements=st.floats(allow_nan=False),
        )
    )
    n_elem = data.draw(st.integers(min_value=0, max_value=A.shape[1] - 1))
    start_idx = data.draw(
        arrays(
            int,
            A.shape[0],
            elements=st.integers(min_value=0, max_value=A.shape[1] - n_elem - 1),
        )
    )
    strided_A = _take_per_row_strided(A, start_idx, n_elem)
    for i, row in enumerate(strided_A):
        start = start_idx[i]
        stop = start + n_elem
        assert np.array_equal(row, A[i, start:stop])
