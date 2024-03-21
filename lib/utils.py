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
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

THIS_DIR = osp.dirname(osp.realpath(__file__))
PROJECT_DIR = osp.abspath(osp.join(THIS_DIR, os.pardir))


def plot_comorbidities(mat, names, title="", cbar_label="", sym=False, ax: plt.Axes = None, row=None, log=False,
                       vmin=None, vmax=None, cbar: bool = True, norm_kwargs={}, boundaries=None, cbar_ticklabels=None,
                       cbar_kwargs={}, bubbles=False, bubble_area_multiplier=30, **im_kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12 if row is None else 1))
    else:
        fig = None

    if "cmap" not in im_kwargs:
        im_kwargs["cmap"] = plt.cm.Blues if not sym else plt.cm.RdBu
    if "norm" not in im_kwargs:
        if boundaries is not None:  # https://stackoverflow.com/questions/67158801/how-to-get-proper-tick-labels-for-a-colarbar-with-discrete-logarithmic-steps)
            # boundaries = np.linspace(0, 2, 6) if L == 3 else np.linspace(0, L - 1, L + 1)
            ncolors = 256  # len(boundaries) - 1  # 256
            im_kwargs["norm"] = mpl.colors.BoundaryNorm(boundaries=boundaries, ncolors=ncolors, **norm_kwargs)
        elif log:
            im_kwargs["norm"] = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        elif sym:
            im_kwargs["norm"] = mpl.colors.CenteredNorm(**norm_kwargs)
        elif vmin is not None or vmax is not None:
            im_kwargs["norm"] = mpl.colors.Normalize(vmin=vmin, vmax=vmax, **norm_kwargs)

    # if conf_thresh is not None:
    #     conf_mat = self.extract_confidence_mask(conf_thresh=conf_thresh)
    #     S = np.ma.array(S, mask=~conf_mat)
    #     cmap.set_bad('white', 1.)

    if not bubbles:
        # Check [*sns.heatmap*](https://seaborn.pydata.org/generated/seaborn.heatmap.html) for useful heatmap function.
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
                # from matplotlib.ticker import LogFormatter
                # cb = plt.colorbar(ticks=[1, 5, 10, 20, 50], format=LogFormatter(10, labelOnlyBase=False)
                cbar.ax.set_yticklabels(cbar_ticklabels)
    else:
        # print((np.log(mat) < 0).sum())
        mat = np.log(mat) if log else mat

        N = len(names)
        # positions = list(itertools.combinations(range(N), 2))
        positions = list(itertools.product(range(N), range(N)))
        x_pos, y_pos = [pos[0] for pos in positions], [N - 1 - pos[1] for pos in positions]
        areas = np.array([np.abs(mat[N - 1 - y, x]) for (x,y) in zip(x_pos,y_pos)]) * bubble_area_multiplier
        colours = ["r" if mat[N - 1 - y, x] < 0 else "b" for (x,y) in zip(x_pos,y_pos)]
        ax.scatter(x_pos, y_pos, s=areas, c=colours, linewidths=2.5)#, alpha=0.7) # linewidths=2.5

        ax.set(xlim=(-0.5, N - 0.5), ylim=(-0.5, N - 0.5))

    ax.set(title=title)
    ax.xaxis.set(ticks=range(len(names)), ticklabels=names)
    ax.yaxis.set(ticks=range(len(names)), ticklabels=names if not bubbles else names[::-1]) if row is None else None
    ax.tick_params(axis='x', labelrotation=90)  # , top=True, labeltop=True, labelbottom=False, bottom=False)

    if fig is not None:
        fig.show()


def compute_associations(df, columns, link_type="count"):
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
                if link_type == "count":
                    adjacency[i, j] = P12 * N
                elif link_type == "RR":
                    adjacency[i, j] = P12 / P1 / P2
                elif link_type == "cosine":
                    adjacency[i, j] = P12 / np.sqrt(P1) / np.sqrt(P2)
                elif link_type == "phi":
                    adjacency[i, j] = (P12 - P1 * P2) / np.sqrt(P1) / np.sqrt(P2) / np.sqrt(1 - P1) / np.sqrt(1 - P2)
                else:
                    raise Exception(f"Link type '{link_type}' not understood.")

    return adjacency + adjacency.T


def compute_RR(X, P_abs, M, pval: float, corrected_pval: float, conf_intervals=True, verbose=False):
    """
    Compute the RR values, its confidence intervals, and significance according to Fisher's exact test.
    As the significance test, we should be using
    [Fisher's exact test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html), as we
    have many pairs of conditions with very few observations.
    Fisher's exact test is computed using `scipy.stats.fisher_exact`
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).
    :param pval: p-value for computing the 99% confidence intervals.
    :param corrected_pval: p-value for computing significance with Fisher's exact test.
    :return: RRs, fishers_signif, RRs_conf
    """
    # Compute RR
    RRs = (X / P_abs[None, :] / P_abs[:, None] * M)

    if conf_intervals:
        # Compute confidence intervals (Katz method)
        sigmas = 1 / X - 1 / M + 1 / P_abs[None, :] / P_abs[:, None] - 1 / M ** 2
        zs = sp.stats.norm.interval(1 - pval, loc=0, scale=1)
        RRs_conf = RRs * np.exp(zs[0] * sigmas), RRs * np.exp(zs[1] * sigmas)
        assert (RRs_conf[0] > RRs).sum() == 0 and (RRs_conf[1] < RRs).sum() == 0
        # self.RRs_signif = np.sign(self.RRs_conf[0] - 1) == np.sign(self.RRs_conf[1] - 1)
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
    fishers_sig = fishers <= corrected_pval
    print(time.strftime("Fisher significance computed in %H hours, %M minutes, %S seconds.",
                        time.gmtime(time.time() - t0))) if verbose else None

    return RRs, fishers_sig, RRs_conf

    # res = pd.DataFrame(
    #     [{"i": i, "j": j, "namei": model.morb_names[i], "namej": model.morb_names[j],
    #       "names": f"{model.morb_names[i]}-{model.morb_names[j]}",
    #       "Xi": P_abs[i], "Xj": P_abs[j], "Xij": X[i, j], "Pi": P_abs[i] / M, "Pj": P_abs[j] / M, "Cij": X[i, j] / M,
    #       "Pi|j": (Pij := X[i, j] / P_abs[j]), "Pj|i": (Pji := X[i, j] / P_abs[i]), "maxPi|j": max(Pij, Pji),
    #       "namei+": f"{model.morb_names[i]}{P_abs[i]}", "namej+": f"{model.morb_names[j]}{P_abs[j]}",
    #       "RR": RRs[i, j], "fisher_sig": fishers_sig[i, j]} for (i, j) in
    #      itertools.combinations(range(len(model.morb_names)), 2)])
    #
    # res["RR_f"] = res["RR"].apply(lambda x: f"{x:.3}")
    # res = res.set_index(res["names"])
    #
    # return RRs, res






def identify_LTC(LTC: object, morb_cols: list) -> tuple:
    """
    Return the id and name of an LTC by providing one of the two and the list of all LTCs
    :param LTC: int or string
    :param morb_cols: list of strings
    :return: LTC_id, LTC_name
    """
    LTC_dict = {name: i for i, name in enumerate(morb_cols)}
    if isinstance(LTC, (int, np.integer)):
        LTC_id = LTC
        LTC_name = morb_cols[LTC]
    else:
        LTC_id = LTC_dict[LTC] if LTC in LTC_dict else None
        LTC_name = LTC
    return LTC_id, LTC_name


def MLTC_count(df, columns):
    long_mat = df[columns].values
    assert (valid := ((long_mat == 0) | (long_mat == 1))).all(), f"{(~valid).sum()} cells don't contain binary values."
    return np.matmul(long_mat.T, long_mat)
