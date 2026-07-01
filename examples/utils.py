# generic
import glob
import re

# third party
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# local
from synthetic_data import synthetic_data_generator


def train_test_data(n_rows=100, n_timesteps=200, rank=5, test_size=0.2, seed=42):
    """Generate a synthetic data matrix with disjoint train/test observation masks.

    Returns:
        X: The (sparsely observed) data matrix.
        O_train, O_test: Disjoint boolean masks over the observed entries in X.
    """
    _, X = synthetic_data_generator(
        n_rows=n_rows, n_timesteps=n_timesteps, rank=rank, seed=seed
    )

    observed_rows, observed_cols = np.nonzero(X)
    train_idx, test_idx = train_test_split(
        np.arange(observed_rows.size), test_size=test_size, random_state=seed
    )

    O_train = np.zeros_like(X, dtype=bool)
    O_test = np.zeros_like(X, dtype=bool)
    O_train[observed_rows[train_idx], observed_cols[train_idx]] = True
    O_test[observed_rows[test_idx], observed_cols[test_idx]] = True

    return X, O_train, O_test


def make_gif(filename, frame_dir):
    frames = [
        Image.open(image)
        for image in sorted(
            glob.glob(f"{frame_dir}/*.jpg"),
            key=lambda x: float(re.findall(r"(\d+)", x)[0]),
        )
    ]
    frame_one = frames[0]
    frame_one.save(
        filename,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=300,
        loop=0,
    )


def set_fig_size(width=None, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    if width is not None:
        if width == "beamer":
            width_pt = 307.28987
        else:
            width_pt = width

        # Width of figure (in pts)
        fig_width_pt = width_pt * fraction

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        golden_ratio = (5**0.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt

        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

        fig_dim = (fig_width_in, fig_height_in)

        return fig_dim


def set_arrowed_spines(fig, ax, eta=0, xshift=0, yshift=0):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ["bottom", "right", "top", "left"]:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1.0 / 40.0 * (ymax - ymin)
    hl = 1.0 / 40.0 * (xmax - xmin)
    ohg = 0.3  # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    ax.arrow(
        xmin - xshift,
        ymin + yshift,
        xmax - xmin + eta * (xmax - xmin) - yshift,
        0.0,
        fc="k",
        ec="k",
        lw=1,
        head_width=hw,
        head_length=hl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )

    ax.arrow(
        xmin - xshift,
        ymin + yshift,
        0.0,
        ymax - ymin + eta * (ymax - ymin) - yshift,
        fc="k",
        ec="k",
        lw=1,
        head_width=yhw,
        head_length=yhl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )


def _set_ylim(values):
    diff = np.max(values) - np.min(values)
    if diff < 0.2:
        delta = 0.5 * (0.2 - diff)
        return [np.min(values) - delta, min(1 + 1e-3, np.max(values) + delta)]

    return


def format_axis(
    axis,
    fig,
    xticklabels=None,
    x_values=None,
    xlim=None,
    ylim=None,
    n_xticks=6,
    xlabel=None,
    ylabel=None,
    grid=True,
    axis_title=None,
    arrowed_spines=True,
):
    if xticklabels is not None:
        axis.set_xticks(range(len(xticklabels)))
        axis.set_xticklabels(xticklabels)

    if n_xticks > 0 and x_values is not None:
        axis.set_xticks(np.linspace(0, 1, n_xticks + 1))
        axis.set_xticklabels(
            np.round(np.linspace(min(x_values), max(x_values), n_xticks + 1), 3)
        )

    if ylim is not None:
        axis.set_ylim(ylim[0], ylim[1])
    else:
        axis.set_ylim(_set_ylim(axis.get_ylim()))

    if xlim is not None:
        axis.set_xlim(xlim[0], xlim[1])

    if xlabel is not None:
        axis.set_xlabel(xlabel)

    if ylabel is not None:
        axis.set_ylabel(ylabel)

    if axis_title is not None:
        axis.set_title(axis_title)

    if arrowed_spines:
        set_arrowed_spines(fig, axis)

    if grid:
        axis.grid(linewidth=1, color="gray", alpha=0.2)
