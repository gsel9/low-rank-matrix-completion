"""
Simple example
"""

import matplotlib.pyplot as plt

# third party
import numpy as np

# local
from lrmc.factor_model import CMC, LMC
from sklearn.metrics import mean_squared_error
from synthetic_data import synthetic_data_generator
from utils import format_axis, set_fig_size


def plot_rec_mse(
    x_coords, y_coords, fig=None, axis=None, axis_label=None, path_to_fig=None
):
    if fig is None and axis is None:
        fig, axis = plt.subplots(1, 1, figsize=set_fig_size(435, fraction=0.9))

    axis.errorbar(
        x_coords,
        np.mean(y_coords, axis=1),
        capsize=3,
        yerr=np.std(y_coords, axis=1),
        label=axis_label,
    )

    format_axis(
        axis,
        fig,
        arrowed_spines=True,
        xlim=(min(x_coords) - 0.05, max(x_coords) + 0.05),
        xlabel="Data density level",
        ylabel="Reconstruction error",
    )

    if axis_label is not None:
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
        )

    fig.tight_layout()

    if path_to_fig is not None:
        fig.savefig(path_to_fig, transparent=True, bbox_inches="tight")

    return fig, axis


def main():
    rank = 5
    n_iter = 1000

    # higher value to emphasize impact of regularization
    lambda3 = 1000

    sparsity_levels = [0.5, 1, 1.5, 2, 4, 8]

    rnd = np.random.RandomState(seed=42)
    seeds = rnd.choice(range(1000), size=10, replace=False)

    cmc_scores, lmc_scores = [], []
    for sparsity_level in sparsity_levels:
        _cmc_scores, _lmc_scores = [], []
        for seed in seeds:
            M, X = synthetic_data_generator(
                n_rows=5000,
                n_timesteps=300,
                rank=rank,
                sparsity_level=sparsity_level,
                seed=seed,
            )

            # factorization with convolution
            cmc_model = CMC(rank=rank, n_iter=n_iter, lambda3=lambda3)
            cmc_model.fit(X)
            _cmc_scores.append(mean_squared_error(M, cmc_model.M))

            # factorization without convolution
            lmc_model = LMC(rank=rank, n_iter=n_iter, lambda3=lambda3)
            lmc_model.fit(X)
            _lmc_scores.append(mean_squared_error(M, lmc_model.M))

        cmc_scores.append(_cmc_scores)
        lmc_scores.append(_lmc_scores)

    fig, axis = plot_rec_mse(sparsity_levels, cmc_scores, axis_label="CMC")
    plot_rec_mse(
        sparsity_levels,
        lmc_scores,
        fig=fig,
        axis=axis,
        axis_label="LMC"
    )


if __name__ == "__main__":
    main()
