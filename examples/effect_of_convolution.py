"""
Simple example
"""
import matplotlib.pyplot as plt

# third party
import numpy as np

# local
from lrmc.utils import finite_difference_matrix, laplacian_kernel_matrix
from utils import set_fig_size


def plot_profile(V, fig=None, axis=None, axis_label=None, path_to_fig=None):
    if fig is None and axis is None:
        fig, axis = plt.subplots(1, 1, figsize=set_fig_size(435, fraction=0.9))

    axis.plot(V, label=axis_label)

    axis.set_yticks([])
    axis.set_yticklabels([])

    axis.set_xticks([])
    axis.set_xticklabels([])

    axis.set_xlabel("Time")

    if axis_label is not None:
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            ncol=3,
            fancybox=True,
            shadow=True,
        )

    if path_to_fig is not None:
        fig.savefig(path_to_fig, transparent=True, bbox_inches="tight")

    return fig, axis


def generate_basic_profile(n_timepoints):
    centre_min, centre_max = 150, 150
    centers = np.linspace(centre_min, centre_max, 1)
    x = np.linspace(0, n_timepoints, n_timepoints)
    k, theta = 3.0, 5e-4
    V = 1 + k * np.exp(-theta * (x[:, None] - centers) ** 2)

    return V


def main():
    n_timepoints = 300

    V = generate_basic_profile(n_timepoints)
    D = finite_difference_matrix(n_timepoints)
    C = laplacian_kernel_matrix(n_timepoints)

    Q = (V - min(V)) / (max(V) - min(V)) / 4
    DV = D @ V
    CDV = (C @ D) @ V

    fig, axis = plot_profile(Q[50:270], axis_label="V")
    fig, axis = plot_profile(DV[50:270], fig=fig, axis=axis, axis_label="DV")
    plot_profile(CDV[50:270], fig=fig, axis=axis, axis_label="CDV")
    plt.show()


if __name__ == "__main__":
    main()
