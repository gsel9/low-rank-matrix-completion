[Installation](#installation) | [Usage](#usage) | [About](#about) | [Key features](#key-features) | [Examples](#examples) | [License](#license)

# LRMC

![GitHub CI](https://github.com/gsel9/low-rank-matrix-factorization/actions/workflows/run_tests.yml/badge.svg)
![GitHub CI](https://img.shields.io/badge/code%20style-black-000000.svg)

## About

This library offers a suite of techniques for low-rank matrix completion of longitudinal data. These techniques introduces various regularization and descrepancy terms to impose structural information on the completed matrix. 

### Low-rank matrix completion for longitudinal data

Low-rank matrix completion (LRMC) is a technique used to recover a partially observed matrix by exploiting the assumption that the matrix has a low-rank structure. In LRMC for longitudinal data, the data was collected over time, potentially at irregular intervals. 

In this context, the data can be organized as a partially observed matrix $\mathbf{X} \in \mathbb{R}^{N \times T}$, with $N$ rows corresponding to time-varying entities, and $T$ columns corresponding to the time points. Each entry $X_{n, t}$ would represent an observation for a particular entity $n$ at a particular time $t$. 

Low-rank matrix completion aims to recover the missing entries by exploiting the assumption that the matrix, when viewed as time-dependent for each entity, can be approximated by a low-rank matrix. The basic factorization model decomposes $\mathbf{X}$ into a set of shared time-varying basic profiles $\mathbf{v}_1, \dots, \mathbf{v}_r$, where $r \ll \min (\{N,T\})$, and profile-specific coefficients in $\mathbf{U}_n$. The linear combination $\mathbf{M}_n = \mathbf{U}_n \mathbf{V}^\top$ of coefficients and basic profiles yields the estimate for the reconstructed profile. 

## Key features

* [Longitudinal matrix completion](./docs/README_lmc.md)
* [Convolutional longitudinal matrix completion](./docs/README_clmc.md)
* [Total variation longitudinal matrix completion](./docs/README_tvlmc.md)
* [Least-angle regression matrix completion](./docs/README_lars.md)
* [Phase-shifted matrix completion](./docs/README_slmc.md)
* [Geometric matrix completion with recurrent multi-graph neural networks](./docs/README_gcrnn.md)

## Installation

Install the library with pip:
```
pip install .
```
This ensures dependencies listed in `pyproject.toml` are handled correctly.

## Usage

A basic example involves estimating the entries of a matrix $X$, given only the entries indicated by $O_{train}$ and use the entries in $O_{test}$ to evaluate the reconstruction accuracy.

```python
# lrmc lib
from lrmc.factor_model import CMC
from utils import train_test_data

# third party
from sklearn.metrics import mean_squared_error

X, O_train, O_test = train_test_data()

X_train = X * O_train
X_test = X * O_test

model = CMC(rank=5, n_iter=100)
model.fit(X_train)

Y_test = model.M * O_test

score = mean_squared_error(X_test, Y_test)
```

## Examples

The [examples](./examples) directory contains runnable scripts demonstrating
each model, e.g. hyperparameter search (`gridsearch.py`), rank selection with
`LarsMC` (`rank_selection.py`), and phase-shifted matrix completion with
`SCMC` (`phase_shifted.py`). Run them from within the `examples` directory
with `src` on the Python path:

```
PYTHONPATH=../src python phase_shifted.py
```

## License

This project is licensed under the [MIT License](./LICENSE).
