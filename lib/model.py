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
from lib.utils import plot_comorbidities, identify_LTC, MLTC_count, PROJECT_DIR, compute_RR

MODELS_PATH = osp.abspath(osp.join(PROJECT_DIR, 'models'))  # directory from which to load model specifications
FIT_MODELS_PATH = "/disk/scratch/gromero/output"  # directory to which to save the parameters of fit models


def find_mode_data_groups(rvals, n_groups=100):
    # TODO: fix this function
    n_samples_per_group = int(len(rvals) / n_groups)
    assert n_samples_per_group > 0, f"Number of samples not enough (for {n_groups} groups): {n_samples_per_group}"

    sorted_rvals = np.sort(rvals)
    min_group_span = np.inf
    for i in np.arange(0, len(rvals), n_samples_per_group):
        if i + n_samples_per_group >= len(rvals):
            break

        lower_rval = sorted_rvals[i]
        higher_rval = sorted_rvals[i + n_samples_per_group]
        group_span = higher_rval - lower_rval
        if group_span < min_group_span:
            min_group_span = group_span
            mode_range = (lower_rval, higher_rval)

    return mode_range


def probability_of_larger(sample1, sample2):
    # TODO: include docstring for this function
    size = min(len(sample1), len(sample2))
    pvalue = (sample1[:size] > sample2[:size]).mean()
    return size, ((1 - pvalue) if pvalue > 0.5 else pvalue)


def create_chain_palette(chains) -> dict:
    """
    For plotting multiple chains of the MCMC process.
    :param chains: number of chains (int) or a specific group of chains (iterator)
    :return: dictionary with chain id as key and colours as values
    """
    chains = range(chains) if isinstance(chains, (int, np.integer)) else chains
    unique_colors = sns.color_palette("hls", len(chains))
    return {chain: unique_colors[i] for i, chain in enumerate(chains)}


class MLTCModel:
    """
NOTE THAT ALTHOUGH THE CODE ALLOWS TO CHOOSE AMONG TWO STAN API PYTHON PACKAGES, ONLY THE PACKAGE `cmdstanpy` HAS BEEN
PROPERLY TESTED.
    """
    perf_keys = ("lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "energy__")
    train_keys = ("num_chains", "num_warmup")

    def __init__(self, model_name: str, stan_api: str, verbose: bool = False):
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

    def create_file_name(self, num_warmup: int):
        return f"_{self.model_name}_wu{num_warmup}"

    def load_fit(self, df: pd.DataFrame, fname: str, column_names, **sample_kwargs) -> bool:
        """

        :param df: pandas DataFrame with the data (in long format, i.e. patients as rows, health conditions as columns,
        and binary values)
        :param fname: name of the file to load the model from or save the model to
        :param column_names: list of names of columns related to health conditions
        :param sample_kwargs: kwargs for the sampling function in *Stan*.
        :return: (bool) whether the model was loaded from a previous file (True) or a new model was trained (False)
        """
        X, self.morb_names = MLTC_count(df, column_names), column_names

        nonzero_mask = ~(X.sum(axis=0) == 0)
        if (~nonzero_mask).sum() > 0:
            self.morb_names = [name for i, name in enumerate(self.morb_names) if nonzero_mask[i]]
            X = X[nonzero_mask][:, nonzero_mask]
            print(f"Removing conditions without cases. Using {len(X)} only.")

        self.df = df
        self.P_abs = np.diagonal(X).copy() if column_names is not None else np.array(
            [self.df.get_prevalence(LTC, abs=True) for LTC in self.morb_names])
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
                self._fit = cmdstanpy.from_csv(fpath)
                self.num_chains = self._fit.chains
                self.num_warmup = self._fit.num_draws_warmup

            return True

        except (FileNotFoundError, ValueError):
            print(f"Model '{fpath}' not found, fitting new model...")
            self._fit_data(**sample_kwargs)
            self.save_fit(fname)
            self.load_fit(df, fname, column_names) if self.stan_api == "pystan" else None
            return False

    def _fit_data(self, num_chains: int, num_warmup: int, num_samples: int, random_seed: int = None,
                  data_kwargs: dict = {}, **sample_kwargs):
        self.X = np.array(self.X, dtype=int)
        N = self.X.shape[0]
        print(f"Shape of co-occurrence matrix: {self.X.shape}")

        # , "Tp": 1}#, # "Tm" : 1, "rho_prior": (10, 30)}  #"r_std_prior":10, "sigma_prior":(1,1)}
        data = {"n": N, "X": self.X, "M": self.M, **data_kwargs}

        if self.stan_api == "pystan":
            import stan
            stan_model = stan.build(self.model_code, data=data, random_seed=random_seed)

        elif self.stan_api == "cmdstanpy":
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

        # self.plot_training_statistics(separate_chains=False)

    def save_fit(self, fname=None):
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

    def _get_var(self, key: str, separate_chains=False, n_chains=None, sub_mask=None, LTC1=None, LTC2=None):
        if self.stan_api == "pystan":
            if key not in self._fit:
                return False

            array = self._fit[key]
            array = array if len(array) > 1 else array[0]

        elif self.stan_api == "cmdstanpy":
            if key in self.perf_keys:
                array = self._fit.method_variables()[key].T
                if not separate_chains and n_chains is None:
                    array = array.flatten()
            else:
                array = self._fit.stan_variable(key).T

                if separate_chains or n_chains is not None:
                    array = array.reshape(n_chains, -1) if array.ndim == 1 else np.moveaxis(
                        array.reshape(*array.shape[:-1], n_chains, -1), -2, 0)

        if n_chains is not None:
            if type(n_chains) == int:
                array = array[:n_chains]
            elif type(n_chains) == list:
                array = array[n_chains]  # tuples don't work with this
            else:
                raise Exception(f"Argument '{n_chains}' for var 'n_chains' not understood.")

        if sub_mask is not None:
            array = array[..., sub_mask]

        if LTC1 is None and LTC2 is None:
            return array
        elif LTC1 is not None and LTC2 is not None:
            LTC1_id, LTC1_name = identify_LTC(LTC1, self.morb_names)
            LTC2_id, LTC2_name = identify_LTC(LTC2, self.morb_names)
            assert LTC2_id != LTC1_id
            ind = (LTC1_id, LTC2_id) if (LTC1_id < LTC2_id and self.stan_api == "pystan") or (
                    LTC1_id > LTC2_id and self.stan_api == "cmdstanpy") else (LTC2_id, LTC1_id)

            if separate_chains or n_chains is not None:
                assert array.shape[1:3] == (len(self.morb_names),) * 2
                return array[:, ind[0], ind[1]]
            else:
                assert array.shape[:2] == (len(self.morb_names),) * 2, str(array.shape[:2]) + str(len(self.morb_names))
                return array[ind]

        elif LTC1 is not None:
            LTC1_id, LTC1_name = identify_LTC(LTC1, self.morb_names)

            # TODO: This should not be here but when the model is trained
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

    def _find_chains_changing_mode(self, n_cuts=None):
        """ Find the chains whose posterior probability perform a big step through sampling of the steady state.
        This is done by splitting each chain at `n_cuts-1` points and comparing whether there are overlaps between
        the mean value + standard deviation of the rest of the chain before and after each of the cuts.

        :param n_cuts (default: None): Number of equidistant cut made on each chain. If `n_cuts=None`, the number of
        cut is inferred such that are separated by around 50 samples.
        :return changing_chains (list): a list containing the chain indices of those that change mode.
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

    def plot_training_statistics(self, separate_chains=False):
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

            # axes[0].hist(fit["treedepth__"][0,:], bins=100);
            # axes[1].hist(fit["n_leapfrog__"][0,:], bins=100, log=True);
            # axes[2].hist(fit["energy__"][0,:], bins=60);
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

        if (changing_chains := self._find_chains_changing_mode()):
            print(f"Chains {changing_chains} seem to be chaning mode.")
            self.plot_logprob(n_chains=changing_chains, temporal=True)

    def plot_logprob(self, ax=None, separate_chains=False, n_chains=None, sub_mask=None, temporal=False):
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        lps = self._get_var("lp__", n_chains=n_chains, separate_chains=separate_chains, sub_mask=sub_mask)
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

    def plot_parameters(self, separate_chains: bool = False, n_chains=None, sub_mask=None, **kwargs):
        raise Exception("Abstract method, to be implemented by children.")

    def plot_sigmas(self, separate_chains=False, n_chains=None, LTC=None, ax=None, normalised: bool = True,
                    sub_mask=None):
        sigmas = self._get_var("sigma", n_chains=n_chains, separate_chains=separate_chains, sub_mask=sub_mask)

        def plot_sigma(idx, axis, label=None):
            if separate_chains or n_chains is not None:
                color_dict = create_chain_palette(n_chains if n_chains is not None else self.num_chains)
                axis.hist(sigmas[:, idx].T, bins=100, histtype="stepfilled", alpha=0.5,
                          color=color_dict.values())
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
            normalised = False
            LTC_id, LTC_name = identify_LTC(LTC, self.morb_names)
            if ax is None:
                fig, ax = plt.subplots()

            plot_sigma(LTC_id, ax, label="inferred $\sigma_i$")
            ax.set(title=LTC_name, ylabel="Count")

        if self.P_abs is not None:  # and not normalised:
            axes = axes if LTC is None else [ax]
            for i, ax in enumerate(axes):
                axes[i].axvline(self.P_abs[i if LTC is None else LTC_id] / self.M, color="k", label="prevalence")
            axes[0].legend()

        fig.show() if ax is None else None

    def posterior_predictive_check(self, *args, **kwargs):
        raise Exception("Abstract method, to be implemented by children.")

    def diagnose_condition_pair(self, LTC1, LTC2):
        raise Exception("Abstract method, to be implemented by children.")

    def identify_LTC(self, LTC):
        return identify_LTC(LTC, self.morb_names)


class ABCModel(MLTCModel):
    """
    This model constraints the sigmas to be shared among pairs.

    NOTE THAT ALTHOUGH THE CODE ALLOWS TO CHOOSE AMONG TWO STAN API PYTHON PACKAGES, ONLY THE PACKAGE `cmdstanpy` HAS BEEN
PROPERLY TESTED.
    """

    def __init__(self, model_name="MLTC_atomic_hyp_mult", stan_api="pystan", verbose=False):
        super(ABCModel, self).__init__(model_name, stan_api, verbose=verbose)

        self.model_name = model_name
        with io.open(osp.join(MODELS_PATH, model_name + '.stan'), 'rt', encoding='utf-8') as f:
            self.model_code = f.read()
            print(self.model_code) if verbose else None

        self.ana = None
        self.RRs, self.RRs_conf, self.RRs_signif, self.fishers_sig = None, None, None, None
        self.r_mode, self.r_conf, self.r_pvalues, self.r_sig = None, None, None, None

        self.association_df = None
        self.LTC_df = None
        self.sorted_prev_mapping = None

    # def load_fit(self, df: PCCIU_DataFrame, fname: str = None, **sample_kwargs) -> bool:
    #     bool_val = super().load_fit(df, fname, **sample_kwargs)
    # self.compute_analytical()
    # return bool_val

    def _fit_data(self, num_chains: int = 5, num_warmup: int = 200, num_samples: int = 50, **sample_kwargs):
        upper_sigs = self.P_abs / self.M * 1.5
        upper_sigs[upper_sigs > 1] = 1
        # upper_r = (1 - sigs[0]) / sigs[0] / sigs[1]
        N = len(upper_sigs)
        if self.model_name[:10] != "MLTC_assoc":
            upper_r = np.array(
                [[(1 - upper_sigs[i]) / upper_sigs[i] / upper_sigs[j] for j in range(N)] for i in range(N)])
        elif self.model_name == "MLTC_assoc":
            upper_r = np.array([[2 / np.max([upper_sigs[i], upper_sigs[j]]) - 1 for j in range(N)] for i in range(N)])
        elif self.model_name == "MLTC_assoc_prop":
            upper_r = np.array(
                [[(upper_sigs[i] + upper_sigs[j]) / np.max([upper_sigs[i], upper_sigs[j]]) ** 2 - 1 for j in range(N)]
                 for i in range(N)])

        print("Max r: ", upper_r.max())

        super()._fit_data(num_chains=num_chains, num_warmup=num_warmup, num_samples=num_samples,
                          data_kwargs={"P": self.P_abs, "upper_sigma": upper_sigs, "upper_r": upper_r}, **sample_kwargs)

    def _get_prev_sorted_mapping(self, beautify_name_func=None, **beautify_name_kwargs):
        if self.LTC_df is None:
            self.get_conds_dataframe()

        sorted_names = list(self.LTC_df.index)
        mapping = [self.morb_names.index(name) for name in sorted_names if name in self.morb_names]

        names = self.LTC_df.index.to_series().apply(beautify_name_func,
                                                    **beautify_name_kwargs) if beautify_name_func is not None else self.LTC_df.index
        self.LTC_df["name+"] = names + " (" + self.LTC_df["Counts"].astype(int).astype(str) + ")"
        sorted_labels = [self.LTC_df.loc[name, "name+"] for name in sorted_names if name in self.morb_names]

        return mapping, sorted_labels

    def get_conds_dataframe(self):
        LTC_data = np.array([self.P_abs, self.P_abs / self.M, range(len(self.morb_names))]).T
        self.LTC_df = pd.DataFrame(index=self.morb_names, columns=["Counts", "Prevalence", "i"],
                                   data=LTC_data).sort_values("Counts", ascending=False)
        return self.LTC_df

    def compute_RR(self, **kwargs):
        """
        Compute the RR values, its confidence intervals, and significance according to Fisher's exact test.
        As the significance test, we should be using
        [Fisher's exact test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html), as we have many pairs of conditions with very few observations.
        Fisher's
        exact test is computed using scipy.stats.fisher_exact
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).
        :param pval: p-value for computing the 99% confidence intervals.
        :param corrected_pval: p-value for computing significance with Fisher's exact test.
        :return: RRs, RRs_conf, RRs_signif, fishers_signif
        """
        self.RRs, self.fishers_sig, self.RRs_conf = compute_RR(self.X, self.P_abs, **kwargs)
        return self.RRs, self.fishers_sig, self.RRs_conf

    def get_r_stats(self, pvalue, corrected_pvalue):
        """
        The Bonferroni-corrected modifies p-values to *p_val / # of combinations*. This is meant to avoid spurious
        reportings when performing multiple tests of significance. However, in our case we care about each association
        individually and not that much about their collective effect. For this reason, the Bonferroni correction should
        not be applied. As in explained in the [medical article](https://onlinelibrary.wiley.com/doi/10.1111/opo.12131)
        by Richard A. Armstrong (2014):

        > Second, correction has been suggested in situations where an investigator is searching for significant
        associations but without a pre-established hypothesis. However, this use depends on the ‘intention’ of the
        investigator. In an exploratory context, an investigator would not wish to miss a possible effect worthy of
        further study and therefore, a correction would be inappropriate. However, if the objective was to test
        everything in the hope that some comparisons would appear significant and the results were not considered to be
        hypotheses for further study, then a correction should be applied.

        :param pvalue:
        :param corrected_pvalue:
        :return:
        """

        def compute_num_interval(samples, pvalue):
            return [np.percentile(samples, 100 * q, axis=-1) for q in (pvalue / 2, 1 - pvalue / 2)]

        triu = np.triu_indices(n=len(self.morb_names), k=1)
        rs_values = self._get_var("r")
        if self.stan_api == "pystan":
            rs_values[triu[1], triu[0]] = rs_values[triu]
        else:
            rs_values[triu] = rs_values[triu[1], triu[0]]
            rs_values[np.diag_indices(n=len(self.morb_names))] = np.NaN

        r_mean, r_median = rs_values.mean(axis=-1), np.median(rs_values, axis=-1)
        self.r_conf = compute_num_interval(rs_values, pvalue)

        # r_down, r_up = compute_num_interval(rs_values, corrected_pvalue)
        # self.r_sig = np.sign(r_down) == np.sign(r_up)

        r_pvalues = (rs_values > (0 if self.model_name[:10] != "MLTC_assoc" else 1)).mean(axis=-1)
        r_pvalues[r_pvalues > 0.5] = 1 - r_pvalues[r_pvalues > 0.5]
        r_pvalues[np.diag_indices(len(r_pvalues))] = 0.5
        self.r_pvalues = r_pvalues
        self.r_sig = r_pvalues <= corrected_pvalue / 2

        self.r_mode = np.zeros_like(r_mean)
        for i, j in itertools.combinations(range(len(rs_values)), 2):
            mode_range = find_mode_data_groups(rs_values[i, j])
            self.r_mode[i, j] = (mode_range[0] + mode_range[1]) / 2
        self.r_mode[triu[1], triu[0]] = self.r_mode[triu]
        # self.r_mode = rs_values[..., np.argmax(self._get_var("lp__"))]

        return r_mean, r_median, self.r_conf, self.r_sig, self.r_mode

    def get_multimorbidity_coupling(self):
        return ((prevs := self.P_abs[:, None] / self.M) - self._get_var("sigma")) / prevs  # P_eld/sigmas

    def get_results_dataframe(self, pvalue=None, corrected_pvalue=None, LTC=None, LTC2=None):
        if self.association_df is None:
            assert pvalue is not None, str(pvalue)
            corrected_pvalue = pvalue if corrected_pvalue is None else corrected_pvalue

            r_mean, r_median, r_conf, r_sig, r_mode = self.get_r_stats(pvalue, corrected_pvalue)
            if self.model_name[:10] != "MLTC_assoc":
                r_mean += 1
                r_median += 1
                r_mode += 1
                r_conf[0] += 1
                r_conf[1] += 1
            if self.RRs is None:
                self.RRs, self.fishers_sig, self.RRs_conf = compute_RR(self.X, P_abs=self.P_abs, M=self.M, pval=pvalue,
                                                                       corrected_pval=corrected_pvalue)

            res = pd.DataFrame(
                [{"i": i, "j": j, "namei": self.morb_names[i], "namej": self.morb_names[j],
                  "names": f"{self.morb_names[i]}-{self.morb_names[j]}",
                  "Xi": self.P_abs[i], "Xj": self.P_abs[j], "Xij": self.X[i, j],
                  "Pi": self.P_abs[i] / self.M, "Pj": self.P_abs[j] / self.M, "Cij": self.X[i, j] / self.M,
                  "Pi|j": (Pij := self.X[i, j] / self.P_abs[j]), "Pj|i": (Pji := self.X[i, j] / self.P_abs[i]),
                  "maxPi|j": max(Pij, Pji),
                  "namei+": f"{self.morb_names[i]}{self.P_abs[i]}", "namej+": f"{self.morb_names[j]}{self.P_abs[j]}",
                  "ABC": r_mode[i, j], "a+1(num, mean)": r_mean[i, j], "a+1(num, median)": r_median[i, j],
                  "a_conf_down": r_conf[0][i, j], "a_conf_up": r_conf[1][i, j], "a_sig": r_sig[i, j],
                  "RR": self.RRs[i, j], "RR_conf_down": self.RRs_conf[0][i, j], "RR_conf_up": self.RRs_conf[1][i, j],
                  "fisher_sig": self.fishers_sig[i, j]} for (i, j) in
                 itertools.combinations(range(len(r_mean)), 2)])

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

    def plot_parameters(self, separate_chains: bool = False, n_chains=None, sub_mask=None, LTC1=None, LTC2=None,
                        prior: bool = False, **kwargs):
        var_kwargs = {"separate_chains": separate_chains, "n_chains": n_chains, "sub_mask": sub_mask}
        if LTC1 is None and LTC2 is None:
            self.plot_rs(**var_kwargs)
            self.plot_sigmas(**var_kwargs)
            self.plot_priors(**var_kwargs)
        else:
            assert LTC1 is not None and LTC2 is not None
            self.plot_distributions(LTC1, LTC2, prior=False, **var_kwargs)
            self.plot_distributions(LTC1, LTC2, prior=prior, log_scale=True, **var_kwargs)

    def plot_rs(self, separate_chains: bool = False, n_chains=None, sub_mask=None):
        N = len(self.morb_names)
        sigmas = self._get_var('sigma', n_chains=n_chains, separate_chains=separate_chains, sub_mask=sub_mask)
        rs = self._get_var("r", n_chains=n_chains, separate_chains=separate_chains, sub_mask=sub_mask)
        if not separate_chains and n_chains is None and rs.ndim == 3 and sigmas.ndim == 2:
            rs = rs[None, :, ...]
            sigmas = sigmas[None, :, ...]
        for i, j in itertools.product(range(N), range(N)):
            if i == j:
                rs[:, i, j] = 0
            elif i > j:
                rs[:, i, j] = rs[:, j, i]

        n_plots = 5  # N
        figsize = 3 * n_plots
        fig, axes = plt.subplots(n_plots, n_plots, figsize=(figsize, figsize))

        for i, j in itertools.product(range(n_plots), range(n_plots)):
            ax = axes[i][j]
            if i == j:  # sigmas
                for k in range(self.num_chains if separate_chains or n_chains is not None else 1):
                    sns.distplot(sigmas[k, i], kde=False, bins=15, ax=ax, label=str(k), hist_kws={'alpha': 1 / 2})
                # ax.axvline(anas[i,j,0]); #upper_sigma[i])
            elif i > j:  # associations
                for k in range(self.num_chains if separate_chains else 1):
                    sns.distplot(rs[k, i, j], kde=False, hist_kws={'alpha': 1 / 2}, bins=15, ax=ax, label=str(k))
                ax.axvline(self.ana[j, i, 2], color="k", label="analytical") if self.ana is not None else None
                ax.axvline(self.RRs[j, i] - 1, color="grey", label="RR - 1") if self.ana is not None else None
                ax.axvline(0, color="green", label="no association")
                ax.text(0.8, 0.8, f"P0={self.P_abs[i]}\nP1={self.P_abs[j]}\nX={self.X[j, i]}", transform=ax.transAxes,
                        ha="center", va="center")

            ax.set(xlabel=f"{j}: {self.morb_names[j]}") if i == n_plots - 1 else None
            ax.set(ylabel=f"{i}: {self.morb_names[i]}") if j == 0 else None

        fig.show()

    def plot_heatmap(self, var, title=None, ax=None, max_val=None, highlight_top: bool = False, bubbles: bool = False,
                     order=None, groups=None, mirror: bool = False, log=True, beautify_name_func=None,
                     beautify_name_kwargs={}, **plot_comorbidities_kwargs):
        """
        Plot a heatmap of associations between health conditions.

        :param var ("ABC", "RR", "RR_only"): Association type. If "RR_only", it shows associations significant to RR but
        not ABC
        :param title: title of the plot (str)
        :param ax:
        :param max_val:
        :param highlight_top: if the top 3 conditions are highlighted
        :param bubbles: bubbleplot instead of heatmap
        :param order: order of LTCs (list of str)
        :param groups: groupings of morbidities (list of lists of str)
        :param mirror: whether both sides of the triangle are show
        :param plot_comorbidities_kwargs:
        :return:
        """
        assert var in ("ABC", "RR", "RR_only", "Cij", "Pi|j"), str(var)
        assert self.association_df is not None, ("Before running 'plot_heatmap()', you need to generate an association "
                                                 "dataframe via 'get_results_dataframe()'.")
        mirror = mirror if var != "Pi|j" else True

        N = len(self.morb_names)
        self.get_conds_dataframe() if self.LTC_df is None else None
        if order is None:
            mapping, sorted_labels = self._get_prev_sorted_mapping(beautify_name_func=beautify_name_func, **beautify_name_kwargs)
        else:
            assert type(order) == list, str(type(order)) + str(order)
            sorted_labels = [label for label in order if label in self.morb_names]
            mapping = [self.morb_names.index(label) for label in sorted_labels if label in self.morb_names]
        line_kwargs = {"ls": "-", "c": "r"}

        if var in ("ABC", "RR"):
            sig_mask = self.association_df["a_sig" if var == "ABC" else "fisher_sig"] == True
        elif var == "RR_only":
            sig_mask = (self.association_df["a_sig"] == False) & (self.association_df["fisher_sig"] == True)
        elif var not in ("Cij", "Pi|j"):
            raise Exception(f"Plotting variable '{var}' not understood.")
        sig_dat = self.association_df[sig_mask] if var not in ("Cij", "Pi|j") else self.association_df

        vals = np.full_like(self.r_mode, np.NaN)
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
        # print(f"max {var}: {max_val:.2f}", end=" ")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = None

        # Grids
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
                for group in groups:
                    for name in group[::-1]:
                        if name in self.morb_names:
                            LTC_i = mapping.index(self.identify_LTC(name)[0])
                            ax.axvline(LTC_i + 0.5), ax.axhline(N - LTC_i - 1.5)
                            break

        # sns.heatmap (https://seaborn.pydata.org/generated/seaborn.heatmap.html) for useful heatmap functionalities
        plot_comorbidities(vals, sorted_labels, ax=ax, title=f"{var}" if title is None else title, sym=True, log=log,
                           vmin=1 / max_val, vmax=max_val, bubbles=bubbles,
                           **plot_comorbidities_kwargs)  # log=False, vmin=1)#sym=True, norm_kwargs={"vcenter":0})
        # axes[1].set_facecolor("darkgrey");

        if fig is not None:
            fig.tight_layout(), fig.show()
        return ax

    def plot_ABC_vs_RR(self, ax, ABC_type="ABC", errors=False, RR_conf=True, a_conf=True,
                       bar=False, log=True, xlim=None, ylim=None, fig=None, title=None):
        assert self.association_df is not None, "Plotting ABC vs RR requires computing confidence intervals. Run 'ConstrainedModel.get_results_dataframe' first. "
        dato = self.association_df
        if RR_conf is not None:
            dato = dato[dato["fisher_sig"] == RR_conf]
        if a_conf is not None:
            dato = dato[dato["a_sig"] == a_conf]
        print(f"{len(dato)}/{len(self.association_df)} data points")

        if ABC_type in ("mode", "ABC") and errors:
            # MAP = np.argmax(self._get_var("lp__"))
            num_modes, low_conf, high_conf = [], [], []
            for i, row in dato.iterrows():
                r_vals = 1 + self._get_var("r", LTC1=row["namei"], LTC2=row["namej"])
                mode_range = find_mode_data_groups(r_vals)
                # num_modes.append(r_vals[MAP])  # (mode_range[0] + mode_range[1]) / 2)
                num_modes.append((mode_range[0] + mode_range[1]) / 2)
                low_conf.append(np.percentile(r_vals, 1))
                high_conf.append(np.percentile(r_vals, 99))
            ax.errorbar(dato["RR"], dato[ABC_type], color="k", ls="", elinewidth=0.5, capsize=1, alpha=0.2,
                        xerr=dato[["RR_error_down", "RR_error_up"]].T, yerr=dato[["ABC_error_down", "ABC_error_up"]].T)

        col = f"a+1(num, {ABC_type})" if ABC_type != "ABC" else ABC_type
        sns.scatterplot(data=dato.sort_values(by="Pi"), x="RR", y=col, hue="Pi", ax=ax, palette="coolwarm")
        # sns.relplot(data=dato[::-1], x="RR", y="a+1", hue="Pi", col="l", kind="scatter", ax=ax, palette="coolwarm");
        # sns.lineplot(data=dato, x="RR", y=f"a+1(num, {ABC_type})", marker='.', ax=ax, style="")
        lim = [0.9 * min([dato["RR"].min(), dato[col].min()]), 1.1 * max([dato["RR"].max(), dato[col].max()])]
        ax.plot(lim, lim, linestyle="--", c="k")
        ax.axhline(1, linestyle=":", c="green")
        ax.axvline(1, linestyle=":", c="green")

        norm = plt.Normalize(dato['Pi'].min(), dato['Pi'].max())
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        fig.colorbar(sm, label="Prevalence of condition, $P_i$") if bar else None
        # ax.legend(title='$P_2$', loc='upper left');  # , labels=['Hell Yeh', 'Nah Bruh'])
        ax.set(
            ylabel=f"Our association measure, $ABC_{{ij}}${((',' if a_conf else ', NOT') + ' significant') if a_conf is not None else ''}",
            xlabel=f"Relative Risk, $RR_{{ij}}${((',' if RR_conf else ', NOT') + ' significant') if RR_conf is not None else ''}");
        if log:
            ax.set(xscale="log", yscale="log")
            if lim[0] > 1:
                lim[0] = 0.9
            # elif np.isclose(lim[0], 0):
            #     lim[0] = 1e-5
        else:
            lim[0] = -0.1
        ax.set(xlim=lim if xlim is None else xlim, ylim=lim if ylim is None else ylim,
               title=ABC_type if title is None else title)
        ax.grid()

    def plot_multimorbidity_coupling(self, pvalue=None, signif=False, xlim=None, ax=None, beautify_name_func=None,
                       **beautify_name_kwargs):
        couplings = self.get_multimorbidity_coupling()
        if signif:
            couplings[np.percentile(couplings, q=pvalue * 100, axis=1) < 0] = np.NaN
        mapping, sorted_labels = self._get_prev_sorted_mapping(beautify_name_func=beautify_name_func, **beautify_name_kwargs)

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))
        box_kwargs = {} if pvalue is None else {"whis": [100 * pvalue, 100 * (1 - pvalue)]}
        sns.boxplot(data=100 * couplings[mapping].T, orient="h", ax=ax, showfliers=False, **box_kwargs)
        ax.axvline(0, color="r")

        ax.set_yticklabels(sorted_labels)
        ax.set(xlabel="Multimorbidity dependence score (%)")
        ax.set(xlim=xlim) if xlim is not None else None
        ax.grid()

        # fig.show()

        return couplings

    def plot_node_degrees(self, ax, pvalue, title=None, catch_all: bool = False, log: bool = False,
                          sort_by: str = "prev", bootstrapped_RR=None, beautify_name_func=lambda x, **kwargs: x,
                       beautify_name_kwargs={}, **ax_set_kwargs):
        assert sort_by in ("prev", "degree")
        dat = self.association_df

        deg, RR_deg = {}, {}
        dat_sig = dat[dat["a_sig"]].loc[:, ["namei", "namej"]]
        RR_dat_sig = dat[dat["fisher_sig"]].loc[:, ["namei", "namej"]]

        for i, LTC1 in enumerate(self.morb_names):
            if catch_all:
                ABCs = (1 if self.model_name[:10] != "MLTC_assoc" else 0) + self._get_var("r", LTC1=LTC1)

                # triu = np.triu_indices(n=len(self.morb_names), k=1)
                # if self.stan_api == "pystan":
                # ABCs[triu[1], triu[0]] = ABCs[triu]
                # else:
                # ABCs[triu] = ABCs[triu[1], triu[0]]
                # ABCs[np.diag_indices(n=len(self.morb_names))] = np.NaN

                deg[LTC1] = np.delete(ABCs, i, axis=0).mean(axis=0)
                # deg[LTC1] = ABCs.mean(axis=0)  # np.delete(ABCs, i, axis=0).mean(axis=0)

            elif LTC1 in dat_sig.values:
                sig_ABCs = []
                for LTC2 in self.morb_names:
                    if (((dat_sig["namei"] == LTC1) & (dat_sig["namej"] == LTC2)).sum() > 0) or (
                            ((dat_sig["namej"] == LTC1) & (dat_sig["namei"] == LTC2)).sum() > 0):
                        r_values = (1 if self.model_name[:10] != "MLTC_assoc" else 0) + self._get_var("r", LTC1=LTC1,
                                                                                                      LTC2=LTC2)
                        sig_ABCs.append(r_values)  # if not log else np.log(r_values))
                sig_ABCs = np.array(sig_ABCs)
                deg[LTC1] = sig_ABCs.sum(axis=0) if sig_ABCs.ndim > 1 else sig_ABCs
            else:
                deg[LTC1] = np.full_like(deg["Hypertension"],
                                         np.NaN)  # np.zeros_like(deg["Hyperension"] )if not catch_all else np.ones_like(deg["Hypertension"])

            if LTC1 in RR_dat_sig.values:
                LTC1_id = identify_LTC(LTC1, self.morb_names)[0]
                sig_RRs = []
                for LTC2 in self.morb_names:
                    if (((RR_dat_sig["namei"] == LTC1) & (RR_dat_sig["namej"] == LTC2)).sum() > 0) or (
                            ((RR_dat_sig["namej"] == LTC1) & (RR_dat_sig["namei"] == LTC2)).sum() > 0):
                        RR_values = self.RRs[LTC1_id, identify_LTC(LTC2, self.morb_names)[0]]
                        sig_RRs.append(RR_values)  # if not log else np.log(RR_values))
                    else:
                        # pass
                        sig_RRs.append(1 if catch_all else 0)
                RR_deg[LTC1] = sum(sig_RRs) if not catch_all else np.mean(sig_RRs)
            else:
                RR_deg[LTC1] = 0 if not catch_all else 1  # np.NaN

        degs = np.array(list(deg.values()))
        mapping = self._get_prev_sorted_mapping()[0] if sort_by == "prev" else np.median(degs, axis=1).argsort()[::-1]
        sorted_labels = [self.morb_names[el] for el in mapping]

        # sns.boxplot(data=degs[mapping].T, orient="h", ax=ax, showfliers=False,
        #             whis=[100 * pvalue, 100 * (1 - pvalue)], color="lightgrey", log_scale=log)
        data = pd.melt(pd.DataFrame(deg), var_name="LTC", value_name="ABC_deg_i")
        sns.boxplot(data=data, y="LTC", x="ABC_deg_i", orient="h", order=sorted_labels, ax=ax, showfliers=False,
                    whis=[100 * pvalue / 2, 100 * (1 - pvalue / 2)], color="lightgrey", log_scale=log)

        ax.axvline(0 if not catch_all else 1, color="r")  # or log

        RR_marker = "o"
        if bootstrapped_RR is None:
            # RR_mapping = [RR_deg.keys().index(name) for name in degs.keys()[mapping]]
            [ax.plot(RR_deg[name], i, marker=RR_marker, color="blue", markersize=8) for i, name in
             enumerate(sorted_labels)]
        else:
            dat = pd.DataFrame(bootstrapped_RR)
            # [dat.drop(col, axis='columns', inplace=True) for col in dat.columns if
            #  np.percentile(dat[col], pvalue / 2 * 100) <= 1]  # remove points that are not significant
            data = pd.melt(dat, var_name="LTC", value_name="RR_deg_i")

            sns.pointplot(data=data, y="LTC", x="RR_deg_i", order=sorted_labels, ax=ax, orient="h", capsize=0.8,
                          err_kws=dict(linewidth=1), errorbar=("pi", 100 * (1 - pvalue / 2)), color="blue",
                          log_scale=log, linestyles=" ")

        ax.grid(lw=0.2)

        sorted_labels = self._get_prev_sorted_mapping(beautify_labels=beautify_name_func, **beautify_name_kwargs)[1] if sort_by == "prev" else [
            beautify_name_func(label, **beautify_name_kwargs) for label in sorted_labels]
        ax.set_yticklabels(sorted_labels)
        ax.set(xlabel=("Node strengths" if not catch_all else "Average association"), **ax_set_kwargs)
        ax.legend(handles=[mpl.lines.Line2D([0], [0], color='blue', marker=RR_marker, label='RR', ls=""),
                           mpl.patches.Patch(color="lightgrey", label="ABC")], loc='upper right')
        ax.set(title=title) if title is not None else None

        return deg, RR_deg, mapping, sorted_labels

    def plot_priors(self, separate_chains: bool = False, n_chains=None, sub_mask=None):
        var_kwargs = {"separate_chains": separate_chains, "n_chains": n_chains, "sub_mask": sub_mask}
        mus = self._get_var('mu_lognormal_prior', **var_kwargs)
        stds = self._get_var('std_lognormal_prior', **var_kwargs)
        alphas = self._get_var('alpha_beta_prior', **var_kwargs)
        betas = self._get_var('beta_beta_prior', **var_kwargs)

        fig, axes = plt.subplots(2, 2)
        axes[0, 0].hist(mus.T, histtype="stepfilled", alpha=0.5)
        axes[0, 0].set(ylabel="mu")
        axes[0, 1].hist(stds.T, histtype="stepfilled", alpha=0.5)
        axes[0, 1].set(ylabel="std")
        axes[1, 0].hist(alphas.T, histtype="stepfilled", alpha=0.5);
        axes[1, 0].set(ylabel="alpha")
        axes[1, 1].hist(betas.T, histtype="stepfilled", alpha=0.5);
        axes[1, 1].set(ylabel="beta")
        fig.tight_layout();
        fig.show()

    def diagnose_condition_pair(self, LTC1, LTC2):
        self.plot_parameters(LTC1=LTC1, LTC2=LTC2, prior=True)
        self.posterior_predictive_check(LTC1=LTC1, LTC2=LTC2)

    def posterior_predictive_check(self, LTC1, LTC2):
        LTC1_id, LTC1_name = identify_LTC(LTC1, self.morb_names)
        LTC2_id, LTC2_name = identify_LTC(LTC2, self.morb_names)
        assert LTC2_id != LTC1_id
        ind = (LTC1_id, LTC2_id) if LTC1_id < LTC2_id else (LTC2_id, LTC1_id)

        fig, axes = plt.subplots(1, 3, figsize=(20, 4))

        ax = axes[0]
        hist_kwg = {"kde": False, "bins": 25, "ax": ax}
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

    def plot_distributions(self, LTC1, LTC2, log_scale=False, prior=False, separate_chains: bool = False, n_chains=None,
                           sub_mask=None, bins=15, xlim=None):
        LTCs = (LTC1, LTC2)
        fig, axes = plt.subplots(1, 3, figsize=(20, 4))

        ax = axes[0]
        hist_kwg = {"kde": True if not log_scale else False, "log_scale": log_scale, "bins": bins, "ax": ax}
        var_kwg = {"n_chains": n_chains, "sub_mask": sub_mask}
        rsij = self._get_var("r", separate_chains=separate_chains if not prior else False, LTC1=LTC1, LTC2=LTC2,
                             **var_kwg)
        rsij_lim = rsij[np.logical_and(rsij >= xlim[0], rsij <= xlim[1])] if xlim is not None else rsij

        log_add = 1 if log_scale and self.model_name[:10] != "MLTC_assoc" else 0
        if not prior:
            [sns.histplot(rsij_lim[k] + log_add, color=f"C{k}", **hist_kwg) for k in
             range(self.num_chains)] if separate_chains else sns.histplot(rsij + log_add, **hist_kwg)
        else:
            r_priors_ij = self._get_var("r_priors", separate_chains=False, LTC1=LTC1, LTC2=LTC2, **var_kwg)
            if xlim is not None:
                r_priors_ij = r_priors_ij[(r_priors_ij >= xlim[0]) & (r_priors_ij <= xlim[1])]

            sns.histplot(rsij_lim + log_add, label="posterior", **hist_kwg)
            sns.histplot(r_priors_ij + log_add, label="prior", color="C1", **hist_kwg)

        ax.axvline(rsij.mean() + log_add, c="r", label=f"ABC (mean): {rsij.mean():.2}")
        ax.axvline(np.median(rsij) + log_add, c="pink", label=f"ABC (median): {np.median(rsij):.2}")
        # plt.axvline(geometric_mean(rvals), color="pink", label="Geometric mean")

        LTC1_id, LTC1_name = identify_LTC(LTC1, self.morb_names)
        LTC2_id, LTC2_name = identify_LTC(LTC2, self.morb_names)
        assert LTC2_id != LTC1_id
        ind = (LTC1_id, LTC2_id) if LTC1_id < LTC2_id else (LTC2_id, LTC1_id)

        rs_kde = (rsij if xlim is None else rsij[rsij < xlim[1]]) if not log_scale else np.log10(rsij + 1)
        kde = sp.stats.gaussian_kde(rs_kde.flatten())
        x = np.linspace(rs_kde.min(), rs_kde.max(), 100)
        kdex = kde(x)
        kde_mode = x[np.argmax(kdex)] if not log_scale else 10 ** x[np.argmax(kdex)]
        ax.axvline(kde_mode, color="blue", label=f"ABC ({'log' if log_scale else ''}mode): {kde_mode:.2}")

        rs_MAP = rsij[np.argmax(self._get_var("lp__"))]
        ax.axvline(rs_MAP, color="orange", label=f"ABC (MAP): {rs_MAP:.2}")

        if self.RRs is not None:
            ax.axvline(self.RRs[ind], c="grey", label=f"RR: {self.RRs[ind]:.2}")
        if self.ana is not None:
            ana_val = self.ana[ind + (2,)]
            ax.axvline(ana_val + log_add, c="k", label=f"ABC (ana) :{ana_val:.2}")
        ax.axvline(1 - log_add, c="g", label="no association")
        # ax.axvline(upper_r, color="k")

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
                sigma_l = self._get_var('sigma', separate_chains=separate_chains, LTC1=LTCs[l], **var_kwg)
                [sns.histplot(sigma_l[k], color=f"C{k}", **hist_kwg) for k in
                 range(self.num_chains)] if separate_chains else sns.histplot(sigma_l, **hist_kwg)
            else:
                sigma_l = self._get_var('sigma', separate_chains=False, LTC1=LTCs[l], **var_kwg)
                sigma_prior_l = self._get_var('sigma_prior_post', separate_chains=False, LTC1=LTCs[l], **var_kwg)
                sns.histplot(sigma_l, label="posterior", **hist_kwg)
                sns.histplot(sigma_prior_l, label="prior", color="C1", **hist_kwg)

            LTC_id, LTC_name = identify_LTC(LTCs[l], self.morb_names)
            ax.axvline(self.P_abs[LTC_id] / self.M, c="k", label=f"Prevalence: {self.P_abs[LTC_id] / self.M:.2}")
            ax.set(xlabel=f"{LTC_name}, $\\sigma_{{{LTC_id}}}$")
            ax.legend()

        fig.show()

    def compute_RR_normal_params(self, LTC1, LTC2):
        indices = tuple(identify_LTC(cond, self.morb_names)[0] for cond in (LTC1, LTC2))

        if self.fishers_sig[indices]:
            m = np.log(self.RRs[indices])
            std = 1 / self.X[indices] - 1 / self.M + 1 / self.P_abs[indices[0]] / self.P_abs[
                indices[1]] - 1 / self.M ** 2
            return m, std, True
        else:
            return None, None, False


def train_group(index, df: PCCIU_DataFrame, grouping, model_type, num_warmup: int = 500, num_chains: int = 3,
                num_samples: int = 1000, positive_levels: int = 2, model_name=None, stan_api="cmdstanpy") -> MLTCModel:
    assert issubclass(model_type, MLTCModel)

    dfs, labels, fnames = df.get_groups(grouping)

    model_kwargs = {} if model_name is None else {"model_name": model_name}
    model = model_type(verbose=False, stan_api=stan_api, **model_kwargs)

    assert grouping != "Age-Sex" or (type(index) == tuple and len(index) == 2)
    df = dfs[index] if grouping != "Age-Sex" else dfs[2 * index[0] + index[1]]
    fname = fnames[index] if grouping != "Age-Sex" else fnames[index[0]][index[1]]

    model.load_fit(df, fname + "0mu" + model.create_file_name(num_warmup), num_chains=num_chains,  # + "0mu"
                   num_samples=num_samples, num_warmup=num_warmup, random_seed=1)
    model.plot_training_statistics(separate_chains=False)
    model.plot_logprob(separate_chains=True, temporal=True)

    return model


if __name__ == '__main__':
    STAN_API = "cmdstanpy"
    if STAN_API == "pystan":
        import stan

        print("Compiling models with pystan version", stan.__version__)
    elif STAN_API == "cmdstanpy":
        import cmdstanpy

        cmdstanpy.show_versions()

    # M = 4e4
    dfo = PCCIU_DataFrame(demo_cols=[])  # , nrows=M)
    # mdel = train_age_group(8, dfo, num_warmup=50000, positive_levels=3, model_type=CooccurrenceModel)
    # mdel = train_age_group(9, dfo, num_chains=3, model_type=AtomicModel)

    # mdel = train_group(9, dfo, "Age", num_warmup=500, num_samples=20000, num_chains=1, model_type=AtomicModelHyperpriors)
    # mdel = train_group(9, dfo, "Age", num_warmup=500, num_samples=2000, num_chains=1, model_type=AtomicModelHyperpriors, model_name="MLTC_atomic_hyp_mult")
    # mdel = train_group(4, dfo, "Age85", num_warmup=500, num_samples=2000, num_chains=1, model_type=AtomicModelHyperpriors, model_name="MLTC_atomic_hyp_mult")
    # mdel = train_group(1, dfo, "Sex", num_warmup=500, num_samples=500, num_chains=3, model_type=AtomicModelHyperpriors)
    # mdel = train_group((9, 0), dfo, "Age-Sex", num_warmup=500, num_samples=20000, num_chains=1, model_type=AtomicModelHyperpriors)
    mdel = train_group((9, 0), dfo, "Age-Sex", num_warmup=1000, num_samples=1000, num_chains=10,
                       model_type=ABCModel, model_name="MLTC_atomic_hyp_mult")

    # model_file = 'MLTC_double-sigmalim_no_prior'
    # mdel = Model(
    # mdel.load_fit(dfo, fname=model_file+"_M1751841")
    # vary_warmup(dfo, mdel)
    # vary_jitter(dfo, mdel)

    # model = AtomicModel()
    # model.load_fit(dfo)
