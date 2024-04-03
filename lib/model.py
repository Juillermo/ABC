#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python class to interact with the stan models.

NOTE THAT ALTHOUGH THE CODE ALLOWS TO CHOOSE AMONG TWO STAN API PYTHON PACKAGES, ONLY THE PACKAGE `cmdstanpy` HAS BEEN
PROPERLY TESTED.

Author:
Guillermo Romero Moreno <Guillermo.RomeroMoreno@ed.ac.uk>
"""

import os.path as osp
import io
import warnings
import itertools
import time

# External packages
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Own packages
from lib.utils import plot_pairwise_relations, identify_LTC, MLTC_count, PROJECT_DIR, compute_RR

MODELS_PATH = osp.abspath(osp.join(PROJECT_DIR, 'models'))  # directory from which to load model specifications
FIT_MODELS_PATH = "output"  # directory to which to save the parameters of fit models


def find_mode_data_groups(samples, window_width=0.01) -> tuple:
    """
    Find the mode of a unimodal distribution from a set of samples. It slides a window including
    `len(samples) * window_width` consecutive samples (sorted by their value) and returns the upper and lower limits of
    the shortest window.

    :param samples: iterable (vector) of numbers
    :param window_width: fraction of samples used for the sliding window
    :return: lower and upper limits of the window of `len(samples) * window_width` samples with the shortest span
    """
    n_samples_per_group = int(len(samples) * window_width)
    assert n_samples_per_group > 0, f"Number of samples {len(samples)} not enough (for window width = {window_width})."

    sorted_rvals = np.sort(samples)
    min_group_span = np.inf
    for i in range(len(samples) - n_samples_per_group):
        lower_val = sorted_rvals[i]
        higher_val = sorted_rvals[i + n_samples_per_group]
        group_span = higher_val - lower_val
        if group_span < min_group_span:
            min_group_span = group_span
            mode_range = (lower_val, higher_val)

    return mode_range


def probability_of_larger(sample1, sample2) -> tuple:
    """
    Calculates the probability that samples from one distribution are larger (smaller) than another.
    :param sample1: iterable (vector) of numbers
    :param sample2: iterable (vector) of numbers
    :return: size of the smaller distribution (int), probability that samples from one distribution are larger (smaller)
    than the other (float in [0 - 0.5])
    """
    size = min(len(sample1), len(sample2))
    pvalue = (sample1[:size] > sample2[:size]).mean()
    return size, ((1 - pvalue) if pvalue > 0.5 else pvalue)


def create_chain_palette(chains) -> dict:
    """
    Returns colours for plotting multiple chains of the MCMC process.
    :param chains: number of chains (int) or a specific group of chains (iterator)
    :return: dictionary with chain id as key and colours as values
    """
    chains = range(chains) if isinstance(chains, (int, np.integer)) else chains
    unique_colors = sns.color_palette("hls", len(chains))
    return {chain: unique_colors[i] for i, chain in enumerate(chains)}


class MLTCModel:
    """
    Base class for Bayesian models for Multiple Long-Term Conditions.

    NOTE THAT ALTHOUGH THE CODE ALLOWS CHOOSING AMONG TWO STAN API PYTHON PACKAGES, ONLY THE PACKAGE `cmdstanpy`
    HAS BEEN PROPERLY TESTED.
    """
    perf_keys = ("lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "energy__")
    train_keys = ("num_chains", "num_warmup")

    def __init__(self, model_name: str, stan_api: str, verbose: bool = False):
        """
        Base class for Bayesian models for Multiple Long-Term Conditions.
        :param model_name: name of the model (str), corresponding to the '.stan' file to be loaded
        :param stan_api: python package used as API for the *Stan* library
        :param verbose: extended debugging information
        """
        self.verbose = verbose

        self.model_name = model_name
        with io.open(osp.join(MODELS_PATH, model_name + '.stan'), 'rt', encoding='utf-8') as f:
            self.model_code = f.read()
            print(self.model_code) if verbose else None

        # Data-related variables
        self.df = None
        self.X = None  # Co-occurrence matrix
        self.P_abs = None  # Prevalence vector (in absolute values)
        self.morb_names = None  # Name of LTCs
        self.M = None  # Number of patients

        # fit-related variables
        assert stan_api in ("cmdstanpy", "pystan")
        self.stan_api = stan_api
        self.num_chains = None
        self.num_warmup = None
        self._fit = None

    def create_file_name(self, num_warmup: int) -> str:
        """
        Creates a name to be used in files to save and load fitted models by returning "_{model_name}_wu{num_warmup}"
        :param num_warmup:
        :return: name (str)
        """
        return f"_{self.model_name}_wu{num_warmup}"

    def load_fit(self, df: pd.DataFrame, fname: str, column_names, **sample_kwargs) -> bool:
        """
        Load data from a fitted model. If previous fitted data does not exist, fit a new model (and return False).

        :param df: pandas DataFrame with the data (in long format, i.e. patients as rows, health conditions as columns,
        and binary values)
        :param fname: name of the file to load the model from or save the model to
        :param column_names: list of names of columns related to health conditions
        :param sample_kwargs: kwargs for the sampling function in *Stan*.
        :return: (bool) whether the model was loaded from a previous file (True) or a new model was trained (False)
        """
        X, self.morb_names = MLTC_count(df, column_names), column_names

        # Remove rows that have zero co-occurrence cases with any other rows
        X_offdiag = np.copy(X)
        np.fill_diagonal(X_offdiag, 0)
        nonzero_mask = ~(X_offdiag.sum(axis=0) == 0)
        if (~nonzero_mask).sum() > 0:
            self.morb_names = [name for i, name in enumerate(self.morb_names) if nonzero_mask[i]]
            X = X[nonzero_mask][:, nonzero_mask]
            print(f"Removing {(~nonzero_mask).sum()} LTCs without co-occurrences. Using {len(X)} LTCs only.")

        self.df = df
        self.P_abs = np.diagonal(X).copy()
        X[np.diag_indices_from(X)] = 0
        self.X = X
        self.M = len(df)

        fpath = osp.abspath(osp.join(FIT_MODELS_PATH, fname))
        try:
            if self.stan_api == "pystan":
                self._fit = dict(np.load(fpath + ".npz"))
                print("Model loaded from:", fpath + ".npz") if self.verbose else None
                self.num_chains = self._fit["num_chains"][0] if "num_chains" in self._fit else None
                self.num_warmup = self._fit["num_warmup"][0] if "num_warmup" in self._fit else None
                if "Q" in self._fit:
                    self._fit["Q"] = self._fit["Q"].astype(np.float64)
                if "r" in self._fit and self._fit["r"].shape[0] > 1:
                    assert self._fit["r"].shape[:2] == X.shape, f"{X.shape} {(~nonzero_mask).sum()}"

            elif self.stan_api == "cmdstanpy":
                import cmdstanpy

                self._fit = cmdstanpy.from_csv(fpath)
                self.num_chains = self._fit.chains
                self.num_warmup = self._fit.num_draws_warmup
            print(f"Model loaded from {fpath}.")
            return True

        except (FileNotFoundError, ValueError):
            print(f"Model '{fpath}' not found, fitting new model...")
            self._fit_data(**sample_kwargs)
            self.save_fit(fname)
            self.load_fit(df, fname, column_names) if self.stan_api == "pystan" else None
            return False

    def _fit_data(self, num_chains: int, num_warmup: int, num_samples: int, random_seed: int = None,
                  stan_data_kwargs: dict = {}, **sample_kwargs):
        """
        Fit the *Stan* model.
        :param num_chains: Number of chains in the MCMC process.
        :param num_warmup: Number of samples discarded as 'warmup' of the MCMC process
        :param num_samples: Number of samples given as a result of the MCMC process
        :param random_seed:
        :param stan_data_kwargs: kwargs passed as inputs to the Stan model (additional to 'n', 'X', and 'M')
        :param sample_kwargs: kwargs passed to the *Stan* sampling function
        """
        self.X = np.array(self.X, dtype=int)
        N = self.X.shape[0]
        print(f"Shape of co-occurrence matrix: {self.X.shape}")

        data = {"n": N, "X": self.X, "M": self.M, **stan_data_kwargs}

        if self.stan_api == "pystan":
            import stan
            stan_model = stan.build(self.model_code, data=data, random_seed=random_seed)

        elif self.stan_api == "cmdstanpy":
            import cmdstanpy

            stan_model = cmdstanpy.CmdStanModel(stan_file=osp.join(MODELS_PATH, self.model_name + '.stan'))
            print(stan_model, stan_model.exe_info()) if self.verbose else None

        self.num_chains = num_chains
        self.num_warmup = num_warmup
        print(f"Sampling with params: {{num_chains: {num_chains}, num_samples: {num_samples}, num_warmup: "
              f"{num_warmup}}}" + (f" + {sample_kwargs}" if sample_kwargs else ""))
        t_ini = time.time()

        if self.stan_api == "pystan":
            sample_kwargs.update(num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup, max_depth=15)
        elif self.stan_api == "cmdstanpy":
            sample_kwargs.update(data=data, iter_warmup=num_warmup, iter_sampling=num_samples, chains=num_chains)
        self._fit = stan_model.sample(**sample_kwargs)

        print(time.strftime("Elapsed time: %H hours, %M minutes, %S seconds.", time.gmtime(time.time() - t_ini)))

        if self.stan_api == "pystan":
            print(self._fit.to_frame().describe().T.iloc[:7].to_string())
            print("   var       shape")
            for key in self._fit.keys():
                print(key, self._fit[key].shape)

        elif self.stan_api == "cmdstanpy":
            print(self._fit.diagnose())
            print(self._fit.summary())

        else:
            raise Exception(f"Stan API '{self.stan_api}' not understood.")

    def save_fit(self, fname: str = None):
        """
        Save the fitted model to a folder named `fname` within the directory `FIT_MODEL_PATH`.

        :param fname: name of the file to save the model. If not provided, use the same name as the model file.
        """
        fname = fname if fname is not None else self.model_name
        fpath = osp.abspath(osp.join(FIT_MODELS_PATH, fname))

        if self.stan_api == "pystan":
            import stan
            if type(self._fit) == stan.fit.Fit:
                data = {par: self._fit[par] for par in self._fit.param_names + self.perf_keys if
                        par not in ("Q", "sigma_raw", "treedepth__", "n_leapfrog__", "energy__")}
                if "Q" in self._fit:
                    data["Q"] = self._fit["Q"].astype(np.half)
            elif type(self._fit) == dict:
                data = self._fit
            else:
                raise Exception(f"Fit data type {type(self._fit)} not understood.")

            data["num_chains"] = np.array([self.num_chains])
            data["num_warmup"] = np.array([self.num_warmup])

            np.savez_compressed(fpath, **data)

        elif self.stan_api == "cmdstanpy":
            self._fit.save_csvfiles(fpath)

        print("Model saved in:", fpath)

    def _get_var(self, var_name: str, separate_chains: bool = False, n_chains=None, sub_mask=None, LTC1=None,
                 LTC2=None) -> np.ndarray:
        """
        Get the inferred from a variable from the model. "r" corresponds to association values.

        :param var_name: name of the variable.
        :param separate_chains: whether the MCMC chains are separated, returning a matrix instead of a vector.
        :param n_chains: (int or list, NOT TUPLE) whether only a subset of the chains is returned
        :param sub_mask: (boolean array) whether only a subset of the samples is returned
        :param LTC1: (int or str) only information related to a specific condition is returned
        :param LTC2: (int or str) only information related to a specific condition pair is returned
        :return: (array)
        """
        if self.stan_api == "pystan":
            if var_name not in self._fit:
                return False

            array = self._fit[var_name]
            array = array if len(array) > 1 else array[0]

        elif self.stan_api == "cmdstanpy":
            # variables related to the performance of the MCMC process have a different treatment
            if var_name in self.perf_keys:
                array = self._fit.method_variables()[var_name].T
                if not separate_chains and n_chains is None:
                    array = array.flatten()
            else:
                array = self._fit.stan_variable(var_name).T

                if separate_chains or n_chains is not None:
                    array = array.reshape(n_chains, -1) if array.ndim == 1 else np.moveaxis(
                        array.reshape(*array.shape[:-1], n_chains, -1), -2, 0)

        if n_chains is not None:
            assert type(n_chains) in (int, list), f"Argument '{n_chains}' of type '{type(n_chains)}' for var 'n_chains'" \
                                                  "not understood."
            array = array[:n_chains] if type(n_chains) == int else array[n_chains]

        if sub_mask is not None:
            array = array[..., sub_mask]

        if LTC1 is None and LTC2 is None:
            return array
        elif LTC1 is not None and LTC2 is not None:
            LTC1_id, LTC1_name = identify_LTC(LTC1, self.morb_names)
            LTC2_id, LTC2_name = identify_LTC(LTC2, self.morb_names)
            assert LTC2_id != LTC1_id, f"LTC1 ({LTC1_id}) must be different to LTC2 ({LTC2_id})"
            ind = (LTC1_id, LTC2_id) if (LTC1_id < LTC2_id and self.stan_api == "pystan") or (
                    LTC1_id > LTC2_id and self.stan_api == "cmdstanpy") else (LTC2_id, LTC1_id)

            if separate_chains or n_chains is not None:
                assert array.shape[1:3] == (len(self.morb_names),) * 2, "Combination of LTC1 and LTC2 only possible" \
                                                                        "with a square variable"
                return array[:, ind[0], ind[1]]
            else:
                assert array.shape[:2] == (len(self.morb_names),) * 2, str(array.shape[:2]) + str(len(self.morb_names))
                return array[ind]

        elif LTC1 is not None:
            LTC1_id, LTC1_name = identify_LTC(LTC1, self.morb_names)

            triu = np.triu_indices(n=len(self.morb_names), k=1)
            if self.stan_api == "pystan":
                array[triu[1], triu[0]] = array[triu]
            elif self.stan_api == "cmdstanpy":
                array[triu] = array[triu[1], triu[0]]
                array[np.diag_indices(n=len(self.morb_names))] = np.NaN
            else:
                raise Exception(f"Stan API '{self.stan_api}' not understood.")

            if separate_chains or n_chains is not None:
                assert array.shape[1] == len(self.morb_names)
                return array[:, LTC1_id]
            else:
                assert array.shape[0] == len(self.morb_names)
                return array[LTC1_id]
        else:
            raise Exception("Error: LTC1 is None and LTC2 is not None.")

    def _find_chains_changing_mode(self, n_cuts: int = None):
        """Detect chains whose posterior probability performs a big step when sampling in the steady state.
        This is done by splitting each chain at `n_cuts-1` points and comparing whether there are overlaps between
        the mean value + standard deviation of the rest of the chain before and after each of the cuts.

        :param n_cuts (default: None): Number of equidistant cut made on each chain. If `n_cuts=None`, the number of
        cut is inferred such that are separated by around 50 samples.
        :return changing_chains (list): a list containing the chain indices of those with a big step.
        """
        changing_chains = []
        lps = self._get_var("lp__", separate_chains=True)
        n_cuts = n_cuts if n_cuts is not None else max(int(len(lps) / 50), 1)
        for l, lp in enumerate(lps):
            sub_lps = np.array_split(lp, n_cuts)
            for i in range(1, len(sub_lps)):
                front = np.concatenate(sub_lps[:i])
                back = np.concatenate(sub_lps[i:])
                if (front.mean() + front.std() < back.mean() - back.std()) or (
                        front.mean() - front.std() > back.mean() + back.std()):
                    changing_chains.append(l)
                    break

        return changing_chains

    def plot_training_statistics(self, separate_chains: bool = False):
        """
        Plot the sampling statistics of the fitted model, such as the *log probability*, the *acceptance statistics*,
        the *step sizes*, the *tree depth*, the *number of leap frogs*, or the *energy*.

        :param separate_chains: whether to plot chains of the MCMC process with different colours.
        """
        if separate_chains:
            fit = self._fit
            n_chains = self.num_chains

            fig, axes = plt.subplots(1, 3, figsize=(25, 4))
            self.plot_logprob(axes[0], separate_chains)

            if "accept_stat__" in fit:
                ars = np.array([fit['accept_stat__'][0, i::n_chains] for i in range(n_chains)])
                axes[1].hist(ars.T, bins=100, log=True, histtype="stepfilled", alpha=0.5)  # histtype="barstacked")
            axes[1].set(xlabel="Acceptance rate")

            if "stepsize__" in fit:
                sss = np.array([fit['stepsize__'][0, i::n_chains] for i in range(n_chains)])
                axes[2].hist(sss.T, bins=60, histtype="stepfilled", alpha=0.5)  # , histtype="barstacked")
            axes[2].set(xlabel="Step size")
        else:
            def plot_hist(ax, var, bins, log=False, xlabel=None):
                ax.hist(vars, bins=bins, log=log) if np.any(vars := self._get_var(var)) else None
                ax.set(xlabel=xlabel if xlabel is not None else var)

            if np.any(self._get_var("treedepth__")):
                fig, axess = plt.subplots(2, 3, figsize=(25, 8))
                axes = axess[0]

                plot_hist(axess[1, 0], "treedepth__", bins=100)
                plot_hist(axess[1, 1], "n_leapfrog__", bins=100)
                plot_hist(axess[1, 2], "energy__", bins=60)
            else:
                fig, axes = plt.subplots(1, 3, figsize=(25, 4))

            self.plot_logprob(axes[0], separate_chains)
            plot_hist(axes[1], 'accept_stat__', bins=100, log=True, xlabel="Acceptance rate")
            plot_hist(axes[2], 'stepsize__', bins=60, xlabel="Step size")

        plt.tight_layout()
        fig.show()

        if changing_chains := self._find_chains_changing_mode():
            print(f"Chains {changing_chains} seem to be changing mode.")
            self.plot_logprob(n_chains=changing_chains, temporal=True)

    def plot_logprob(self, ax: plt.Axes = None, separate_chains: bool = False, n_chains=None,
                     temporal: bool = False, **get_var_kwargs):
        """
        Plot the log posterior probability of the samples from the MCMC process.

        :param ax: (None) if given, use it for the plot
        :param separate_chains: whether to use different colours for each chain of the MCMC process
        :param n_chains: (int or list, NOT TUPLE) whether only a subset of the chains is returned
        :param temporal:
        :param **get_var_kwargs: additional kwargs for the `_get_var` function.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        lps = self._get_var("lp__", n_chains=n_chains, separate_chains=separate_chains, **get_var_kwargs)
        if separate_chains or n_chains is not None:
            color_dict = create_chain_palette(n_chains if n_chains is not None else self.num_chains)
            if not temporal:
                ax.hist(lps.T, bins=100, histtype="stepfilled", alpha=0.5, color=color_dict.values())
                ax.set(xlabel="log-posterior")
            else:
                chains = color_dict.keys()
                for k, chain in enumerate(chains):
                    ax.scatter(range(len(lps[k])), lps[k], s=15, color=color_dict[chain])
                ax.set(xlabel='Iteration', xlim=(0, len(lps[k])))
        else:
            ax.hist(lps, bins=250) if not temporal else ax.scatter(range(len(lps)), lps, s=15)
            ax.set(xlabel='log-posterior' if not temporal else "Iteration")

        if fig is not None:
            fig.show()

    def plot_sigmas(self, separate_chains: bool = False, n_chains=None, LTC=None, ax: plt.Axes = None,
                    normalised: bool = True, **_get_var_kwargs):
        """
        Plot a histogram of samples from the inferred variable *sigma*, related to the probability of conditions to
        appear due to independent factors.

        :param separate_chains: whether chains of the MCMC process are plotted with different colours
        :param n_chains: (int or list) subset of MCMC chains to plot
        :param LTC: (int or str) only a single condition is plotted
        :param ax: a specific matplotlib Axes object to contain the plot
        :param normalised: whether to normalise sigma ranges (when multiple sigmas are plotted)
        :param **_get_var_kwargs: additional kwargs for the `_get_var` function
        """
        sigmas = self._get_var("sigma", n_chains=n_chains, separate_chains=separate_chains, **_get_var_kwargs)

        def plot_sigma(idx, axis, label=None):
            if separate_chains or n_chains is not None:
                color_dict = create_chain_palette(n_chains if n_chains is not None else self.num_chains)
                axis.hist(sigmas[:, idx].T, bins=100, histtype="stepfilled", alpha=0.5, color=color_dict.values())
            else:
                axis.hist(sigmas[idx], bins=100, **({"label": label} if label is not None else {}))

        if LTC is None:
            assert ax is None, "Plotting sigma with a pre-defined axis not implemented yet."
            assert not normalised or self.P_abs is not None

            fig, axes = plt.subplots(N := len(self._get_var("sigma")), 1, figsize=(20, 1.5 * N))
            for i in range(N):
                plot_sigma(i, axes[i])

                axes[i].text(0.01, 0.8, self.morb_names[i], verticalalignment='center', transform=axes[i].transAxes)
                axes[i].set(xlim=[0, sigmas.max() if not normalised else max(self.P_abs[i] / self.M,
                                                                             sigmas[..., i, :].max())])
                axes[i].tick_params(labelbottom=False) if not normalised and i != N - 1 else None

        else:
            LTC_id, LTC_name = identify_LTC(LTC, self.morb_names)
            if ax is None:
                fig, ax = plt.subplots()

            plot_sigma(LTC_id, ax, label="inferred $\sigma_i$")
            ax.set(title=LTC_name, ylabel="Count")

        if self.P_abs is not None:
            axes = axes if LTC is None else [ax]
            for i, ax in enumerate(axes):
                axes[i].axvline(self.P_abs[i if LTC is None else LTC_id] / self.M, color="k", label="prevalence")
            axes[0].legend()

        fig.show() if ax is None else None


class ABCModel(MLTCModel):
    """
    This model constraints the probability of a condition occurring independently to be shared among pairs.

    NOTE THAT ALTHOUGH THE CODE ALLOWS TO CHOOSE AMONG TWO STAN API PYTHON PACKAGES, ONLY THE PACKAGE `cmdstanpy` HAS
    BEEN PROPERLY TESTED.
    """

    def __init__(self, model_name="MLTC_atomic_hyp_mult", stan_api="cmdstanpy", verbose=False):
        """
        Bayesian models for Multiple Long-Term Conditions that constraints the probability of a condition occurring
        independently to be shared among associations.

        :param model_name: name of the model (str), corresponding to the '.stan' file to be loaded
        :param stan_api: python package used as API for the *Stan* library
        :param verbose: extended debugging information
        """
        super(ABCModel, self).__init__(model_name, stan_api, verbose=verbose)

        self.model_name = model_name
        with io.open(osp.join(MODELS_PATH, model_name + '.stan'), 'rt', encoding='utf-8') as f:
            self.model_code = f.read()
            print(self.model_code) if verbose else None

        self.RRs, self.RRs_conf, self.fishers_sig = None, None, None
        self.assoc_mode, self.assoc_credible_int, self.assoc_pvalues, self.assoc_signif = None, None, None, None

        self.association_df = None
        self.LTC_df = None
        self.sorted_prev_mapping = None

    def _fit_data(self, num_chains: int = 5, num_warmup: int = 200, num_samples: int = 50, **sample_kwargs):
        """
        Fit the *Stan* model.

        :param num_chains: Number of chains in the MCMC process.
        :param num_warmup: Number of samples discarded as 'warmup' of the MCMC process
        :param num_samples: Number of samples given as a result of the MCMC process
        :param sample_kwargs: kwargs passed to the *Stan* sampling function
        """
        upper_sigs = self.P_abs / self.M * 1.5
        upper_sigs[upper_sigs > 1] = 1
        N = len(upper_sigs)
        upper_r = np.array([[(1 - upper_sigs[i]) / upper_sigs[i] / upper_sigs[j] for j in range(N)] for i in range(N)])

        super()._fit_data(stan_data_kwargs={"P": self.P_abs, "upper_sigma": upper_sigs, "upper_r": upper_r},
                          num_chains=num_chains, num_warmup=num_warmup, num_samples=num_samples, **sample_kwargs)

    def _get_prev_sorted_mapping(self, beautify_name_func=None, **beautify_name_kwargs):
        """
        Returns a mapping of the conditions by prevalence along with the sorted labels including the value of the
        prevalence and posibly beautified.

        :param beautify_name_func: function to beautify label names
        :param beautify_name_kwargs: kwargs passed to `beautify_name_func`
        :return: (tuple) list with the mapping, sorted (beautified) labels with counts between parentheses
        """
        if self.LTC_df is None:
            self.get_conds_dataframe()  # this function produces self.LTC_df, with LTCs already sorted by prevalence

        sorted_names = list(self.LTC_df.index)
        mapping = [self.morb_names.index(name) for name in sorted_names if name in self.morb_names]

        names = self.LTC_df.index.to_series().apply(beautify_name_func,
                                                    **beautify_name_kwargs) if beautify_name_func is not None else self.LTC_df.index
        self.LTC_df["name+"] = names + " (" + self.LTC_df["Counts"].astype(int).astype(str) + ")"
        sorted_labels = [self.LTC_df.loc[name, "name+"] for name in sorted_names if name in self.morb_names]

        return mapping, sorted_labels

    def get_conds_dataframe(self) -> pd.DataFrame:
        """
        Produce dataframe include the basic information and statistics of each LTC.
        :return: pandas.DataFrame containing LTCs as rows and *Counts*, *Prevalence*, *index* ("i") as columns, with
        LTCs sorted by prevalence (highest to lowest)
        """
        LTC_data = np.array([self.P_abs, self.P_abs / self.M, range(len(self.morb_names))]).T
        self.LTC_df = pd.DataFrame(index=self.morb_names, columns=["Counts", "Prevalence", "i"],
                                   data=LTC_data).sort_values("Counts", ascending=False)
        return self.LTC_df

    def get_assoc_stats(self, credible_inteval_pvalue, significance_pvalue):
        """
        Get the basic association statistics for associations, such as the mean, median, mode, credible intervals, and
        significance.

        :param credible_inteval_pvalue: (float) p-value to used for computing credible intervals
        :param significance_pvalue: p-value used for discarding associations
        :return: (tuple of NxN matrices) assoc_mean, assoc_median, assoc_credible_int, assoc_sig (bool), assoc_mode
        """

        def compute_num_interval(samples, pvalue):
            return [np.percentile(samples, 100 * q, axis=-1) for q in (pvalue / 2, 1 - pvalue / 2)]

        triu = np.triu_indices(n=len(self.morb_names), k=1)
        assoc_values = self._get_var("r")
        if self.stan_api == "pystan":
            assoc_values[triu[1], triu[0]] = assoc_values[triu]
        elif self.stan_api == "cmdstanpy":
            assoc_values[triu] = assoc_values[triu[1], triu[0]]
            assoc_values[np.diag_indices(n=len(self.morb_names))] = np.NaN
        else:
            raise Exception(f"Stan API '{self.stan_api}' not understood.")

        assoc_mean, assoc_median = assoc_values.mean(axis=-1), np.median(assoc_values, axis=-1)
        self.assoc_credible_int = compute_num_interval(assoc_values, credible_inteval_pvalue)

        assoc_pvalues = (assoc_values > 0).mean(axis=-1)
        assoc_pvalues[assoc_pvalues > 0.5] = 1 - assoc_pvalues[assoc_pvalues > 0.5]
        assoc_pvalues[np.diag_indices(len(assoc_pvalues))] = 0.5
        self.assoc_pvalues = assoc_pvalues
        self.assoc_signif = assoc_pvalues <= significance_pvalue / 2

        t_ini = time.time()
        self.assoc_mode = np.zeros_like(assoc_mean)
        for i, j in itertools.combinations(range(len(assoc_values)), 2):
            mode_range = find_mode_data_groups(assoc_values[i, j])
            self.assoc_mode[i, j] = (mode_range[0] + mode_range[1]) / 2
        self.assoc_mode[triu[1], triu[0]] = self.assoc_mode[triu]
        print(time.strftime("Elapsed time (computing modes): %H hours, %M minutes, %S seconds.",
                            time.gmtime(time.time() - t_ini)))

        return assoc_mean, assoc_median, self.assoc_credible_int, self.assoc_signif, self.assoc_mode

    def get_multimorbidity_coupling(self) -> np.ndarray:
        """
        Compute the multimorbidity coupling as (P - sigma) / P, ie the contribution of factors common to other
        conditions for the appearance of each condition. A value of zero implies no coupling with other conditions and
        negative values imply a tendency not to have other conditions.

        :return: matrix of multimorbidity couplings of size NxS where N is the number of LTCs and S the number of
        samples.
        """
        return ((prevs := self.P_abs[:, None] / self.M) - self._get_var("sigma")) / prevs

    def get_results_dataframe(self, credible_inteval_pvalue=None, significance_pvalue=None, LTC=None, LTC2=None):
        """
        Returns a dataframe containing the statistics relating all possible pairs of conditions, i.e. (N-1)N/2 rows.
        Each row corresponds to a pair of condition. If LTC is not None, return only information about that LTC and its
        associations. If LTC2 is also provided, only return the row corresponding to both LTC and LTC2.

        :param credible_inteval_pvalue: (float) p-value to used for computing credible intervals
        :param significance_pvalue: p-value used for discarding associations
        :param LTC: (int or str)
        :param LTC2: (int or str)
        :return: dataframe containing the statistics relating all possible pairs of conditions or a subset of them (if
        LTCs are provided as arguments)
        """
        if self.association_df is None:
            assert credible_inteval_pvalue is not None, str(credible_inteval_pvalue)
            corrected_pvalue = credible_inteval_pvalue if significance_pvalue is None else significance_pvalue

            assoc_mean, assoc_median, assoc_cred_int, assoc_signif, assoc_mode = self.get_assoc_stats(
                credible_inteval_pvalue, corrected_pvalue)
            # Below are to shift the association values to the [0-inf) range.
            assoc_mean += 1
            assoc_median += 1
            assoc_mode += 1
            assoc_cred_int[0] += 1
            assoc_cred_int[1] += 1
            if self.RRs is None:
                self.RRs, self.fishers_sig, self.RRs_conf = compute_RR(self.X, P_abs=self.P_abs, M=self.M,
                                                                       conf_pval=credible_inteval_pvalue,
                                                                       signif_pval=corrected_pvalue)

            res = pd.DataFrame(
                [{"i": i, "j": j, "namei": self.morb_names[i], "namej": self.morb_names[j],
                  "names": f"{self.morb_names[i]}-{self.morb_names[j]}",
                  "Xi": self.P_abs[i], "Xj": self.P_abs[j], "Xij": self.X[i, j],
                  "Pi": self.P_abs[i] / self.M, "Pj": self.P_abs[j] / self.M, "Cij": self.X[i, j] / self.M,
                  "Pi|j": (Pij := self.X[i, j] / self.P_abs[j]), "Pj|i": (Pji := self.X[i, j] / self.P_abs[i]),
                  "maxPi|j": max(Pij, Pji),
                  "namei+": f"{self.morb_names[i]}{self.P_abs[i]}", "namej+": f"{self.morb_names[j]}{self.P_abs[j]}",
                  "ABC": assoc_mode[i, j], "a+1(num, mean)": assoc_mean[i, j], "a+1(num, median)": assoc_median[i, j],
                  "a_conf_down": assoc_cred_int[0][i, j], "a_conf_up": assoc_cred_int[1][i, j],
                  "a_sig": assoc_signif[i, j],
                  "RR": self.RRs[i, j], "RR_conf_down": self.RRs_conf[0][i, j], "RR_conf_up": self.RRs_conf[1][i, j],
                  "fisher_sig": self.fishers_sig[i, j]} for (i, j) in
                 itertools.combinations(range(len(assoc_mean)), 2)])

            res["RR/a"] = res["RR"] / res["ABC"]
            res["ABC_error_down"], res["ABC_error_up"] = (
                res["ABC"] - res["a_conf_down"], res["a_conf_up"] - res["ABC"])
            res["a_f"] = res["ABC"].apply(lambda x: f"{x:.3}")
            res["a CI (99%)"] = res["a_conf_down"].apply(lambda x: f"{x:.3}") + " - " + res["a_conf_up"].apply(
                lambda x: f"{x:.3}")
            res["a_f_sig"] = res["a_f"].copy()
            res.loc[~res["a_sig"], "a_f_sig"] = "NS"
            res["a_f_sig_CI"] = res["a_f"] + " (" + res["a CI (99%)"] + ")"
            res.loc[~res["a_sig"], "a_f_sig_CI"] = "NS"

            res["RR_error_down"], res["RR_error_up"] = (res["RR"] - res["RR_conf_down"], res["RR_conf_up"] - res["RR"])
            res["RR_f"] = res["RR"].apply(lambda x: f"{x:.3}")
            res["RR CI (99%)"] = res["RR_conf_down"].apply(lambda x: f"{x:.3}") + " - " + res["RR_conf_up"].apply(
                lambda x: f"{x:.3}")
            res["RR_f_sig_CI"] = res["RR_f"] + " (" + res["RR CI (99%)"] + ")"
            res.loc[~res["fisher_sig"], "RR_f_sig_CI"] = "NS"
            self.association_df = res.set_index(res["names"])

        if LTC is None:
            assert LTC2 is None
            return self.association_df
        else:
            if LTC not in self.morb_names:
                return False
            reduced_df = pd.concat((self.association_df[self.association_df["namei"] == LTC],
                                    self.association_df[self.association_df["namej"] == LTC]))
            if LTC2 is None:
                return reduced_df
            else:
                assert LTC2 != LTC, LTC2 + "=" + LTC
                if LTC2 not in self.morb_names or LTC2 == LTC:
                    return False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return pd.concat((reduced_df[self.association_df["namei"] == LTC2],
                                      reduced_df[self.association_df["namej"] == LTC2])).iloc[0]

    def plot_parameters(self, LTC1=None, LTC2=None, prior: bool = False, **var_kwargs):
        """
        Produce plot showing the parameter distributions.

        :param LTC1: if given, show only parameters related to LTC1
        :param LTC2: if given along LTC1, show only parameters related to the pair LTC1, LTC2
        :param prior: if true, include the prior distributions in the plots
        :param var_kwargs: kwargs passed to the `_get_var` method
        """
        if LTC1 is None and LTC2 is None:
            self.plot_assoc(**var_kwargs)
            self.plot_sigmas(**var_kwargs)
            self.plot_priors(**var_kwargs)
        else:
            assert LTC1 is not None and LTC2 is not None
            self.plot_distributions(LTC1, LTC2, prior=False, **var_kwargs)
            self.plot_distributions(LTC1, LTC2, prior=prior, log_scale=True, **var_kwargs)

    def plot_assoc(self, separate_chains: bool = False, n_chains=None, **var_kwargs):
        """
        Plot the distributions of association values.

        :param separate_chains: whether chains of the MCMC process are coloured differently
        :param n_chains: (int or list) if only a subset of chains is plotted
        :param **var_kwargs: additonal kwargs to pass to the `_get_var` method
        """
        N = len(self.morb_names)
        sigmas = self._get_var('sigma', n_chains=n_chains, separate_chains=separate_chains, **var_kwargs)
        assoc_vals = self._get_var("r", n_chains=n_chains, separate_chains=separate_chains, **var_kwargs)
        if not separate_chains and n_chains is None and assoc_vals.ndim == 3 and sigmas.ndim == 2:
            assoc_vals = assoc_vals[None, :, ...]
            sigmas = sigmas[None, :, ...]
        for i, j in itertools.product(range(N), range(N)):
            if i == j:
                assoc_vals[:, i, j] = 0
            elif i > j:
                assoc_vals[:, i, j] = assoc_vals[:, j, i]

        n_plots = 5
        figsize = 3 * n_plots
        fig, axes = plt.subplots(n_plots, n_plots, figsize=(figsize, figsize))

        for i, j in itertools.product(range(n_plots), range(n_plots)):
            ax = axes[i][j]
            if i == j:  # sigmas
                for k in range(self.num_chains if separate_chains or n_chains is not None else 1):
                    sns.distplot(sigmas[k, i], kde=False, bins=15, ax=ax, label=str(k), hist_kws={'alpha': 1 / 2})
            elif i > j:  # associations
                for k in range(self.num_chains if separate_chains else 1):
                    sns.distplot(assoc_vals[k, i, j], kde=False, hist_kws={'alpha': 1 / 2}, bins=15, ax=ax,
                                 label=str(k))
                # ax.axvline(self.RRs[j, i] - 1, color="grey", label="RR - 1")
                ax.axvline(0, color="green", label="no association")
                ax.text(0.8, 0.8, f"P0={self.P_abs[i]}\nP1={self.P_abs[j]}\nX={self.X[j, i]}", transform=ax.transAxes,
                        ha="center", va="center")

            ax.set(xlabel=f"{j}: {self.morb_names[j]}") if i == n_plots - 1 else None
            ax.set(ylabel=f"{i}: {self.morb_names[i]}") if j == 0 else None

        fig.show()

    def plot_heatmap(self, var, title: str = None, ax: plt.Axes = None, max_val=None, highlight_top: bool = False,
                     bubbles: bool = False, order=None, groups=None, mirror: bool = False, log: bool = True,
                     beautify_name_func=None, beautify_name_kwargs={}, **plot_comorbidities_kwargs):
        """
        Plot a heatmap of associations between health conditions. Only significant associations are plotted.

        :param var ("ABC", "RR", "RR_only"): Association type. If "RR_only", it shows associations significant to RR but
        not ABC
        :param title: title of the plot (str)
        :param ax: ax object to include the plot
        :param max_val: upper cap on association values
        :param highlight_top: whether the top 3 conditions should be highlighted
        :param bubbles: bubbleplot instead of heatmap
        :param order: order of LTCs (list of str)
        :param groups: groupings of morbidities (list of lists of str)
        :param mirror: whether both sides of the triangle are show
        :param log: if a logarithmic scale should be used for the colours
        :param beautify_name_func: beautifying function to use on the LTC labels
        :param beautify_name_kwargs: kwargs for the `beautify_name_func`
        :param plot_comorbidities_kwargs: additional kwargs passed to the `plot_comorbidities` function
        :return: ax object with the plotted heatmap
        """
        assert var in ("ABC", "RR", "RR_only", "Cij", "Pi|j"), str(var)
        assert self.association_df is not None, ("Before running `plot_heatmap()`, you need to generate an association "
                                                 "dataframe via 'get_results_dataframe()'.")
        mirror = mirror if var != "Pi|j" else True
        if ax is None:
            fig, ax = plt.subplots()

        N = len(self.morb_names)
        self.get_conds_dataframe() if self.LTC_df is None else None
        if order is None:
            mapping, sorted_labels = self._get_prev_sorted_mapping(beautify_name_func=beautify_name_func,
                                                                   **beautify_name_kwargs)
        else:
            assert type(order) == list, str(type(order)) + str(order)
            sorted_labels = [label for label in order if label in self.morb_names]
            mapping = [self.morb_names.index(label) for label in sorted_labels if label in self.morb_names]
        line_kwargs = dict(ls="-", c="r")

        if var in ("ABC", "RR"):
            sig_mask = self.association_df["a_sig" if var == "ABC" else "fisher_sig"] == True
        elif var == "RR_only":
            sig_mask = (self.association_df["a_sig"] == False) & (self.association_df["fisher_sig"] == True)
        elif var not in ("Cij", "Pi|j"):
            raise Exception(f"Plotting variable '{var}' not understood.")
        sig_dat = self.association_df[sig_mask] if var not in ("Cij", "Pi|j") else self.association_df

        vals = np.full_like(self.assoc_mode, np.NaN)
        var = "RR" if var == "RR_only" else var
        for k, (_, row) in enumerate(sig_dat.sort_values(var if var != "Pi|j" else "Cij", ascending=False).iterrows()):
            i, j = mapping.index(row["i"]), mapping.index(row["j"])
            if i > j:
                i, j = j, i
            vals[j, i] = row[var] if var != "Pi|j" else row["Xij"] / row["Xj"]
            if mirror:
                vals[i, j] = row[var] if var != "Pi|j" else row["Xij"] / row["Xi"]

            if k < 3 and highlight_top:
                ax.plot([-0.5, i], [j, j], **line_kwargs)
                ax.plot([i, i], [N - 0.5, j], **line_kwargs)

        max_val = np.nanmax(vals) if max_val is None else max_val
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = None

        # Grids on half of the triangle
        grid_kwargs = {"color": "lightgrey", "linewidth": 0.5}
        if not bubbles:
            for x in np.arange(-.5, N - 1, 1):
                ax.plot([x, x], [x if not mirror else -0.5, N - 0.5], **grid_kwargs)
                ax.plot([-0.5, (x + 1) if not mirror else (N - 0.5)], [x + 1, x + 1], **grid_kwargs)
                ax.plot([-0.5, N - 0.5], [-0.5, N - 0.5], color="k", linewidth=0.5) if mirror else None
        else:
            if not mirror:
                for x in np.arange(0, N, 1):
                    ax.plot([x, x], [-0.5, N - x - 2], **grid_kwargs)
                    ax.plot([-0.5, x - 1], [N - x - 1, N - x - 1], **grid_kwargs)
            else:
                ax.grid(**grid_kwargs)
                ax.plot([-0.5, N - 0.5], [N - 0.5, -0.5], color="k", linewidth=0.5)

            if groups is not None:
                # Additional blue lines to separate the groups
                for group in groups:
                    for name in group[::-1]:
                        if name in self.morb_names:
                            LTC_i = mapping.index(identify_LTC(name, self.morb_names)[0])
                            ax.axvline(LTC_i + 0.5), ax.axhline(N - LTC_i - 1.5)
                            break

        plot_pairwise_relations(vals, sorted_labels, ax=ax, title=f"{var}" if title is None else title, sym=True, log=log,
                                vmin=1 / max_val, vmax=max_val, bubbles=bubbles,
                                **plot_comorbidities_kwargs)

        if fig is not None:
            fig.tight_layout(), fig.show()
        return ax

    def plot_multimorbidity_coupling(self, pvalue=None, signif: bool = False, xlim=None, ax: plt.Axes = None,
                                     beautify_name_func=None, **beautify_name_kwargs):
        """
        Plot multimorbidity coupling of all LTCs.

        :param pvalue: (float) p-value for selecting credible intervals
        :param signif: whether to remove values for LTCs whose coupling is not significant (i.e. the credible interval
        crosses zero.
        :param xlim: limits for x-axis
        :param ax: axis object to plot on
        :param beautify_name_func: function to beautify the names of the LTC labels
        :param beautify_name_kwargs: kwargs for the `beautify_name_func`
        :return: matrix with values of coupling (Nxsamples, where N is the number of LTCs)
        """
        couplings = self.get_multimorbidity_coupling()
        if signif:
            couplings[np.percentile(couplings, q=pvalue * 100, axis=1) < 0] = np.NaN
        mapping, sorted_labels = self._get_prev_sorted_mapping(beautify_name_func=beautify_name_func,
                                                               **beautify_name_kwargs)

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))
        box_kwargs = {} if pvalue is None else {"whis": [100 * pvalue, 100 * (1 - pvalue)]}
        sns.boxplot(data=100 * couplings[mapping].T, orient="h", ax=ax, showfliers=False, **box_kwargs)
        ax.axvline(0, color="r")

        ax.set_yticklabels(sorted_labels)
        ax.set(xlabel="Multimorbidity dependence score (%)")
        ax.set(xlim=xlim) if xlim is not None else None
        ax.grid()

        return couplings

    def plot_node_degrees(self, ax, pvalue, title: str = None, catch_all: bool = False, log: bool = False,
                          sort_by: str = "prev", bootstrapped_RR=None, beautify_name_func=lambda x, **kwargs: x,
                          beautify_name_kwargs={}, **ax_set_kwargs):
        """
        Plot the strength (or average association) of each LTC within the multimorbidity network.

        :param ax: The matplotlib axes to plot on
        :param pvalue: (float) p-value to use for calculating the error bars and boxplot whiskers
        :param title: title of plot
        :param catch_all: whether to include non-significant LTCs in the computation of strength (or average association)
        :param log: if True, a logarithmic scale is used
        :param sort_by: it can be "prev" or "degree", indicating whether LTCs are sorted by their prevalence or in
        descending order of strength / degree / average association
        :param bootstrapped_RR: (NxB matrix, where B is the number of bootstraps) if given, average associations with RR
        are given with confidence intervals as well
        :param beautify_name_func: function to beautify LTC labels
        :param beautify_name_kwargs: kwargs to pass to `beautify_name_func`
        :param ax_set_kwargs: kwargs to pass to the `plt.Axes.set` method
        :return: ABC_deg (values of degree / strength / avg associations by ABC), RR_deg (values of degree / strength /
        avg associations by RR), mapping (mapping of the chosen order of LTCs), sorted_labels (of the chosen order of
        LTCs)
        """
        assert sort_by in ("prev", "degree")
        dat = self.association_df

        ABC_deg, RR_deg = {}, {}
        dat_sig = dat[dat["a_sig"]].loc[:, ["namei", "namej"]]
        RR_dat_sig = dat[dat["fisher_sig"]].loc[:, ["namei", "namej"]]

        for i, LTC1 in enumerate(self.morb_names):
            if catch_all:
                ABCs = 1 + self._get_var("r", LTC1=LTC1)
                ABC_deg[LTC1] = np.delete(ABCs, i, axis=0).mean(axis=0)

            elif LTC1 in dat_sig.values:
                sig_ABCs = []
                for LTC2 in self.morb_names:
                    if (((dat_sig["namei"] == LTC1) & (dat_sig["namej"] == LTC2)).sum() > 0) or (
                            ((dat_sig["namej"] == LTC1) & (dat_sig["namei"] == LTC2)).sum() > 0):
                        r_values = 1 + self._get_var("r", LTC1=LTC1, LTC2=LTC2)
                        sig_ABCs.append(r_values)
                sig_ABCs = np.array(sig_ABCs)
                ABC_deg[LTC1] = sig_ABCs.sum(axis=0) if sig_ABCs.ndim > 1 else sig_ABCs
            else:
                ABC_deg[LTC1] = np.full_like(ABC_deg["Hypertension"], np.NaN)

            if LTC1 in RR_dat_sig.values:
                LTC1_id = identify_LTC(LTC1, self.morb_names)[0]
                sig_RRs = []
                for LTC2 in self.morb_names:
                    if (((RR_dat_sig["namei"] == LTC1) & (RR_dat_sig["namej"] == LTC2)).sum() > 0) or (
                            ((RR_dat_sig["namej"] == LTC1) & (RR_dat_sig["namei"] == LTC2)).sum() > 0):
                        RR_values = self.RRs[LTC1_id, identify_LTC(LTC2, self.morb_names)[0]]
                        sig_RRs.append(RR_values)
                    else:
                        sig_RRs.append(1 if catch_all else 0)
                RR_deg[LTC1] = sum(sig_RRs) if not catch_all else np.mean(sig_RRs)
            else:
                RR_deg[LTC1] = 0 if not catch_all else 1

        degs = np.array(list(ABC_deg.values()))
        mapping = self._get_prev_sorted_mapping()[0] if sort_by == "prev" else np.median(degs, axis=1).argsort()[::-1]
        sorted_labels = [self.morb_names[el] for el in mapping]

        data = pd.melt(pd.DataFrame(ABC_deg), var_name="LTC", value_name="ABC_deg_i")
        sns.boxplot(data=data, y="LTC", x="ABC_deg_i", orient="h", order=sorted_labels, ax=ax, showfliers=False,
                    whis=[100 * pvalue / 2, 100 * (1 - pvalue / 2)], color="lightgrey", log_scale=log)

        ax.axvline(0 if not catch_all else 1, color="r")

        RR_marker = "o"
        if bootstrapped_RR is None:
            [ax.plot(RR_deg[name], i, marker=RR_marker, color="blue", markersize=8) for i, name in
             enumerate(sorted_labels)]
        else:
            dat = pd.DataFrame(bootstrapped_RR)
            data = pd.melt(dat, var_name="LTC", value_name="RR_deg_i")
            sns.pointplot(data=data, y="LTC", x="RR_deg_i", order=sorted_labels, ax=ax, orient="h", capsize=0.8,
                          err_kws=dict(linewidth=1), errorbar=("pi", 100 * (1 - pvalue / 2)), color="blue",
                          log_scale=log, linestyles=" ")

        ax.grid(lw=0.2)

        sorted_labels = self._get_prev_sorted_mapping(beautify_labels=beautify_name_func, **beautify_name_kwargs)[
            1] if sort_by == "prev" else [beautify_name_func(label, **beautify_name_kwargs) for label in sorted_labels]
        ax.set_yticklabels(sorted_labels)
        ax.set(xlabel=("Node strengths" if not catch_all else "Average association"), **ax_set_kwargs)
        ax.legend(handles=[mpl.lines.Line2D([0], [0], color='blue', marker=RR_marker, label='RR', ls=""),
                           mpl.patches.Patch(color="lightgrey", label="ABC")], loc='upper right')
        ax.set(title=title) if title is not None else None

        return ABC_deg, RR_deg, mapping, sorted_labels

    def plot_priors(self, **var_kwargs):
        """
        Plot the distributions of the prior parameters.

        :param **var_kwargs: keyword arguments for the `_get_var` method.
        """
        mus = self._get_var('mu_lognormal_prior', **var_kwargs)
        stds = self._get_var('std_lognormal_prior', **var_kwargs)
        alphas = self._get_var('alpha_beta_prior', **var_kwargs)
        betas = self._get_var('beta_beta_prior', **var_kwargs)

        fig, axes = plt.subplots(2, 2)
        axes[0, 0].hist(mus.T, histtype="stepfilled", alpha=0.5), axes[0, 0].set(ylabel="mu")
        axes[0, 1].hist(stds.T, histtype="stepfilled", alpha=0.5), axes[0, 1].set(ylabel="std")
        axes[1, 0].hist(alphas.T, histtype="stepfilled", alpha=0.5), axes[1, 0].set(ylabel="alpha")
        axes[1, 1].hist(betas.T, histtype="stepfilled", alpha=0.5), axes[1, 1].set(ylabel="beta")
        fig.tight_layout(), fig.show()

    def diagnose_condition_pair(self, LTC1, LTC2):
        """
        Show detailed model information about a pair of conditions
        :param LTC1: (int or str)
        :param LTC2: (int or str)
        """
        self.plot_parameters(LTC1=LTC1, LTC2=LTC2, prior=True)
        self.posterior_predictive_check(LTC1=LTC1, LTC2=LTC2)

    def posterior_predictive_check(self, LTC1, LTC2):
        """
        Check the mismatch between the data and samples extracted from the inferred generative model, for a pair of
        conditions. THIS FUNCTION MAY NEED UPDATING.
        :param LTC1:
        :param LTC2:
        """
        LTC1_id, LTC1_name = identify_LTC(LTC1, self.morb_names)
        LTC2_id, LTC2_name = identify_LTC(LTC2, self.morb_names)
        assert LTC2_id != LTC1_id
        ind = (LTC1_id, LTC2_id) if LTC1_id < LTC2_id else (LTC2_id, LTC1_id)

        fig, axes = plt.subplots(1, 3, figsize=(20, 4))

        ax = axes[0]
        hist_kwg = dict(kde=False, bins=25, ax=ax)
        Xs = self._get_var(f"X_post", LTC1=LTC1, LTC2=LTC2)
        sns.histplot(Xs, label=f"post pred check", discrete=False if Xs.max() > hist_kwg["bins"] else True, **hist_kwg)
        ax.axvline(self.X[ind], c="k", label="Data")
        ax.set(xlabel="Co-occurrence")
        ax.legend()

        P_preds = self._get_var(f"P_post")
        for i in range(len(self.morb_names)):
            P_preds[i, i] = 0
        for l in range(2):
            ax = axes[l + 1]
            hist_kwg["ax"] = ax
            sns.histplot(P_preds[ind[l], :].flat, label=f"post pred check",
                         discrete=False if P_preds[ind[l], :].max() > hist_kwg["bins"] else True, **hist_kwg)
            ax.axvline(self.P_abs[ind[l]], c="k", label=f"Data")
            ax.set(xlabel=f"Prevalence of {ind[l]}: {self.morb_names[ind[l]]}")
            ax.legend()

        fig.show()

    def plot_distributions(self, LTC1, LTC2, log_scale: bool = False, prior: bool = False,
                           separate_chains: bool = False, bins=15, xlim=None, **var_kwargs):
        """
        Plot the distributions of association values and independent factors for a pair of conditions.

        :param LTC1: (int or str)
        :param LTC2: (int or str)
        :param log_scale: whether to use a logarithmic scale
        :param prior: whether the priors of the distributions are included
        :param separate_chains: whether to plot each chain of MCMC process as a separate distribution
        :param bins: for the histogram
        :param xlim: limits of the x-axis
        :param **var_kwargs: additional keyword arguments to pass to the `_get_var` method
        """
        LTCs = (LTC1, LTC2)
        fig, axes = plt.subplots(1, 3, figsize=(20, 4))

        ax = axes[0]
        hist_kwg = dict(kde=True if not log_scale else False, log_scale=log_scale, bins=bins, ax=ax)
        ABCij = self._get_var("r", separate_chains=separate_chains if not prior else False, LTC1=LTC1, LTC2=LTC2,
                             **var_kwargs)
        ABC_lim = ABCij[np.logical_and(ABCij >= xlim[0], ABCij <= xlim[1])] if xlim is not None else ABCij

        log_add = 1 if log_scale else 0
        if not prior:
            [sns.histplot(ABC_lim[k] + log_add, color=f"C{k}", **hist_kwg) for k in
             range(self.num_chains)] if separate_chains else sns.histplot(ABCij + log_add, **hist_kwg)
        else:
            ABC_prior_ij = self._get_var("r_priors", separate_chains=False, LTC1=LTC1, LTC2=LTC2, **var_kwargs)
            if xlim is not None:
                ABC_prior_ij = ABC_prior_ij[(ABC_prior_ij >= xlim[0]) & (ABC_prior_ij <= xlim[1])]

            sns.histplot(ABC_lim + log_add, label="posterior", **hist_kwg)
            sns.histplot(ABC_prior_ij + log_add, label="prior", color="C1", **hist_kwg)

        ax.axvline(ABCij.mean() + log_add, c="r", label=f"ABC (mean): {ABCij.mean():.2}")
        ax.axvline(np.median(ABCij) + log_add, c="pink", label=f"ABC (median): {np.median(ABCij):.2}")

        LTC1_id, LTC1_name = identify_LTC(LTC1, self.morb_names)
        LTC2_id, LTC2_name = identify_LTC(LTC2, self.morb_names)
        assert LTC2_id != LTC1_id
        ind = (LTC1_id, LTC2_id) if LTC1_id < LTC2_id else (LTC2_id, LTC1_id)

        ABC_kde = (ABCij if xlim is None else ABCij[ABCij < xlim[1]]) if not log_scale else np.log10(ABCij + 1)
        kde = sp.stats.gaussian_kde(ABC_kde.flatten())
        x = np.linspace(ABC_kde.min(), ABC_kde.max(), 100)
        kdex = kde(x)
        kde_mode = x[np.argmax(kdex)] if not log_scale else 10 ** x[np.argmax(kdex)]
        ax.axvline(kde_mode, color="blue", label=f"ABC ({'log' if log_scale else ''}mode): {kde_mode:.2}")

        ABC_MAP = ABCij[np.argmax(self._get_var("lp__"))]
        ax.axvline(ABC_MAP, color="orange", label=f"ABC (MAP): {ABC_MAP:.2}")

        if self.RRs is not None:
            ax.axvline(self.RRs[ind], c="grey", label=f"RR: {self.RRs[ind]:.2}")
        ax.axvline(1 - log_add, c="g", label="no association")

        ax.set(xlabel="Associations, ABC" + (" + 1" if log_add else ""),
               title=f"{LTC1_name} - {LTC2_name} ({self.X[LTC1_id, LTC2_id]} cases)")
        if xlim is not None and not log_scale:
            ax.set(xlim=xlim)

        ax.legend()

        for l in range(2):
            ax = axes[l + 1]
            hist_kwg["ax"] = ax
            hist_kwg["kde"] = False
            if not prior:
                sigma_l = self._get_var('sigma', separate_chains=separate_chains, LTC1=LTCs[l], **var_kwargs)
                [sns.histplot(sigma_l[k], color=f"C{k}", **hist_kwg) for k in
                 range(self.num_chains)] if separate_chains else sns.histplot(sigma_l, **hist_kwg)
            else:
                sigma_l = self._get_var('sigma', separate_chains=False, LTC1=LTCs[l], **var_kwargs)
                sigma_prior_l = self._get_var('sigma_prior_post', separate_chains=False, LTC1=LTCs[l], **var_kwargs)
                sns.histplot(sigma_l, label="posterior", **hist_kwg)
                sns.histplot(sigma_prior_l, label="prior", color="C1", **hist_kwg)

            LTC_id, LTC_name = identify_LTC(LTCs[l], self.morb_names)
            ax.axvline(self.P_abs[LTC_id] / self.M, c="k", label=f"Prevalence: {self.P_abs[LTC_id] / self.M:.2}")
            ax.set(xlabel=f"{LTC_name}, $\\sigma_{{{LTC_id}}}$")
            ax.legend()

        fig.show()

    def compute_RR_normal_params(self, LTC1, LTC2):
        """
        Compute the mean and standard deviation of the normal distribution of RR of conditions LTC1 and LTC2.
        :param LTC1: (int or str)
        :param LTC2: (int or str)
        :return: mean (float), std (float), bool (whether the RR is significant according to Fishers' exact test)
        """
        indices = tuple(identify_LTC(cond, self.morb_names)[0] for cond in (LTC1, LTC2))

        if self.fishers_sig[indices]:
            m = np.log(self.RRs[indices])
            std = 1 / self.X[indices] - 1 / self.M + 1 / self.P_abs[indices[0]] / self.P_abs[
                indices[1]] - 1 / self.M ** 2
            return m, std, True
        else:
            return None, None, False
