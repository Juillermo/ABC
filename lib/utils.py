"""
Author: Guillermo Romero Moreno (Guillermo.RomeroMoreno@ed.ac.uk)
Date: 9/2/2022

This file contains useful functions and variables to be used in other scripts.
"""

import os
import os.path as osp
import time
import itertools

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

THIS_DIR = osp.dirname(osp.realpath(__file__))
PROJECT_DIR = osp.abspath(osp.join(THIS_DIR, os.pardir))


def plot_pairwise_relations(mat, names, title="", cbar_label="", sym=False, ax: plt.Axes = None, row: int = None,
                            log: bool = False, vmin=None, vmax=None, cbar: bool = True, norm_kwargs: dict = {},
                            boundaries=None, cbar_ticklabels=None, cbar_kwargs: dict = {}, bubbles: bool = False,
                            bubble_area_multiplier=30, **im_kwargs):
    """
    Plot pairwise statistics in a heatmap / bubble plot format.

    :param mat: square matrix with pairwise relations
    :param names: (iterable) names of the LTCs
    :param title: title of the plot
    :param cbar_label: label for the colorbar (if heatmap plot)
    :param sym: whether colours are symmetric around zero (if heatmap plot)
    :param ax: matplotlib axes object to plot on
    :param row: if given, relations of a single LTC is plotted
    :param log: whether a logarithmic scale is used (if heatmap plot)
    :param vmin: (float) lower value cap (if heatmap plot)
    :param vmax: (float) higher value cap (if heatmap plot)
    :param cbar: whether the colorbar is plotted (if heatmap plot)
    :param norm_kwargs: additional keyword arguments to pass to the matplotlib Norm object (if heatmap plot)
    :param boundaries: (iterable) boundaries of the discrete steps for colurs (if heatmap plot)
    :param cbar_ticklabels: labels to use for the colorbar (if heatmap plot)
    :param cbar_kwargs: additional keyword arguments to pass to the `plt.colorbar` function
    :param bubbles: whether to use a bubble plot instead of a heatmap
    :param bubble_area_multiplier: (float) multiplier of the value of a pairwise relationship to use as bubble area
    :param im_kwargs: additional keyword arguments to pass to the `plt.imshow` function (if heatmap plot)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12 if row is None else 1))
    else:
        fig = None

    if "cmap" not in im_kwargs:
        im_kwargs["cmap"] = plt.cm.Blues if not sym else plt.cm.RdBu
    if "norm" not in im_kwargs:
        if boundaries is not None:
            ncolors = 256
            im_kwargs["norm"] = mpl.colors.BoundaryNorm(boundaries=boundaries, ncolors=ncolors, **norm_kwargs)
        elif log:
            im_kwargs["norm"] = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        elif sym:
            im_kwargs["norm"] = mpl.colors.CenteredNorm(**norm_kwargs)
        elif vmin is not None or vmax is not None:
            im_kwargs["norm"] = mpl.colors.Normalize(vmin=vmin, vmax=vmax, **norm_kwargs)

    if not bubbles:
        im = ax.imshow(mat if row is None else mat[None, row], **im_kwargs)
        ax.spines[['right', 'top']].set_visible(False)

        if cbar:
            if ax is not None:
                cbar_kwargs["ax"] = ax
            if "extend" in norm_kwargs:
                cbar_kwargs["extend"] = norm_kwargs["extend"]
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04, label=cbar_label, **cbar_kwargs)
            if cbar_ticklabels is not None:
                formatter = mpl.ticker.LogFormatter(base=1.6, labelOnlyBase=False)
                cbar.ax.yaxis.set_major_formatter(formatter)
                cbar.ax.yaxis.set_ticks([0.3, 0.5, 0.7, 1, 2, 3, 4, 5])
                cbar.ax.set_yticklabels(cbar_ticklabels)
    else:
        mat = np.log(mat) if log else mat

        N = len(names)
        positions = list(itertools.product(range(N), range(N)))
        x_pos, y_pos = [pos[0] for pos in positions], [N - 1 - pos[1] for pos in positions]
        areas = np.array([np.abs(mat[N - 1 - y, x]) for (x,y) in zip(x_pos,y_pos)]) * bubble_area_multiplier
        colours = ["r" if mat[N - 1 - y, x] < 0 else "b" for (x,y) in zip(x_pos,y_pos)]
        ax.scatter(x_pos, y_pos, s=areas, c=colours, linewidths=2.5)

        ax.set(xlim=(-0.5, N - 0.5), ylim=(-0.5, N - 0.5))

    ax.set(title=title)
    ax.xaxis.set(ticks=range(len(names)), ticklabels=names)
    ax.yaxis.set(ticks=range(len(names)), ticklabels=names if not bubbles else names[::-1]) if row is None else None
    ax.tick_params(axis='x', labelrotation=90)

    if fig is not None:
        fig.show()


def compute_associations(df: pd.DataFrame, columns, assoc_type: str) -> np.ndarray:
    """
    Compute pairwise associations between variables using the Salton Cosine Index or the Pearsons's phi-correlation for
    binary variables.

    :param df: dataframe in 'long format' (i.e. rows as cases and columns as variables).
    :param columns: (iterable of str) columns for which pairwise associations are computed
    :param assoc_type: association type. It can be "cosine" (Salton Cosine Index), or "phi" (Pearson's phi-correlation).
    :return: square matrix containing pairwise associations of the chosen type
    """
    M = len(columns)
    adjacency = np.zeros((M, M))
    for i, col in enumerate(columns):
        for j, col2 in enumerate(columns):
            if j >= i:
                break
            partition_mask = df[col].astype(bool)
            partition_mask2 = df[col2].astype(bool)

            N = len(partition_mask)
            P1, P2 = partition_mask.sum() / N, partition_mask2.sum() / N
            if (P12 := (partition_mask & partition_mask2).sum() / N) > 0:
                if assoc_type == "cosine":
                    adjacency[i, j] = P12 / np.sqrt(P1) / np.sqrt(P2)
                elif assoc_type == "phi":
                    adjacency[i, j] = (P12 - P1 * P2) / np.sqrt(P1) / np.sqrt(P2) / np.sqrt(1 - P1) / np.sqrt(1 - P2)
                else:
                    raise Exception(f"Link type '{assoc_type}' not understood.")

    return adjacency + adjacency.T


def compute_RR(X, P_abs, M, conf_pval: float, signif_pval: float, conf_intervals=True, verbose=False):
    """
    Compute the RR values, its confidence intervals, and significance according to
    [Fisher's exact test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html), useful
    if pairs of conditions have very few observations. Fisher's exact test is computed using `scipy.stats.fisher_exact`
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).

    :param conf_pval: p-value for computing the 99% confidence intervals.
    :param signif_pval: p-value for computing significance with Fisher's exact test.
    :return: RRs, fishers_signif, RRs_conf
    """
    RRs = (X / P_abs[None, :] / P_abs[:, None] * M)

    if conf_intervals:
        # Compute confidence intervals (Katz method)
        sigmas = 1 / X - 1 / M + 1 / P_abs[None, :] / P_abs[:, None] - 1 / M ** 2
        zs = sp.stats.norm.interval(1 - conf_pval, loc=0, scale=1)
        RRs_conf = RRs * np.exp(zs[0] * sigmas), RRs * np.exp(zs[1] * sigmas)
        assert (RRs_conf[0] > RRs).sum() == 0 and (RRs_conf[1] < RRs).sum() == 0
    else:
        RRs_conf = None

    # Compute significance (Fisher's exact test)
    t0 = time.time()
    fishers = np.full_like(RRs, np.NaN)
    for i, j in itertools.combinations(range(len(X)), 2):
        n11 = X[i, j]
        n10, n01 = [P_abs[m] - X[i, j] for m in (i, j)]
        n00 = M - n10 - n01 - n11
        if n11 > 0 and n10 > 0 and n01 > 0 and n00 > 0:
            fishers[i, j] = sp.stats.fisher_exact([[n11, n01], [n10, n00]])[1]
            fishers[j, i] = fishers[i, j]
    fishers_sig = fishers <= signif_pval
    print(time.strftime("Fisher significance computed in %H hours, %M minutes, %S seconds.",
                        time.gmtime(time.time() - t0))) if verbose else None

    return RRs, fishers_sig, RRs_conf


def identify_LTC(LTC, morb_cols: list) -> tuple:
    """
    Return the id and name of an LTC by providing one of the two and the list of all LTCs

    :param LTC: int or string
    :param morb_cols: iterable of strings
    :return: LTC_id (int), LTC_name (str)
    """
    LTC_dict = {name: i for i, name in enumerate(morb_cols)}
    if isinstance(LTC, (int, np.integer)):
        LTC_id = LTC
        LTC_name = morb_cols[LTC]
    else:
        LTC_id = LTC_dict[LTC] if LTC in LTC_dict else None
        LTC_name = LTC
    return LTC_id, LTC_name


def MLTC_count(df: pd.DataFrame, columns: list) -> np.ndarray:
    """
    Returns the co-occurrence matrix of a dataframe for the given columns (LTCs), which must contain binary values. The
    diagonal elements of the matrix contain the total counts in each column (LTC).

    :param df: pandas.DataFrame in 'long-format', ie with rows as patients and columns as variables
    :param columns: list of columns containing LTC information (values must be binary)
    :return: X (np.ndarray), square matrix with co-occurrence counts in the off-diagonals and LTC counts in the diagonal
    """
    long_mat = df[columns].values
    assert (valid := ((long_mat == 0) | (long_mat == 1))).all(), f"{(~valid).sum()} cells don't contain binary values."
    return np.matmul(long_mat.T, long_mat)
