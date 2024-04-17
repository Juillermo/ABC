"""
Author: Guillermo Romero Moreno (Guillermo.RomeroMoreno@ed.ac.uk)
Date: 9/2/2022

This file contains the basic functions for loading the dataset and obtaining information from it.
"""

import os.path as osp
import time
import re

import numpy as np
import pandas as pd

from ABC.utils import PROJECT_DIR

DATA_PATH = osp.abspath(osp.join(PROJECT_DIR, 'PCCIU/processed_data/clean_PCCIU_Multimorbidity.tsv'))


def load_dataset(morb_cols=None, **read_csv_kwargs):
    """
    This function loads the dataset, which needs to be in long-form --- i.e. rows corresponding to patients and
    columns corresponding to conditions, with binary values. We also have a column with "Sex" (categorical) and a column
    with "Age" (integer).
    For a data subsample, include e.g. `nrows=1e5` as kwarg.

    :param morb_cols: list with the column names of the dataset containing condition diagnoses.
    :param read_csv_kwargs: kwargs for the `pd.read_csv` function.
    :return: data (pd.DataFrame with the dataset in long-form), morb_names (a list with the names of the columns related
     to diagnoses of conditions)
    """

    print("Loading dataset...")
    ini_time = time.time()

    # CHANGE THE LINES BELOW TO ADAPT THEM TO YOUR DATASET
    data = pd.read_csv(DATA_PATH, delimiter="\t", index_col=0, **read_csv_kwargs)
    data["Sex"] = data["Sex"].apply(lambda x: "Men" if x == 1 else ("Women" if x == 2 else "Other"))
    morb_names = list(data.columns[11:51]) if morb_cols is None else morb_cols

    data["n_morb"] = data[morb_names].sum(axis=1)
    print(f"Dataset loaded (elapsed time: {time.time() - ini_time:.2f}s)")
    print("Length of dataset:", len(data))

    return data, morb_names


def stratify_dataset(data, stratification_variable):
    """
    This function stratifies the dataset along a stratification variable.

    :param data: pandas DataFrame with the dataset in long-form.
    :param stratification_variable: variable to use for stratifying. The options are "Age" (10 years age groups), "Age85"
     (20 years age groups), "Sex", "Age-Sex" (sex and 10 years age groups), "Age-Sex85" (sex and 20 years age groups).
    :return dfs: list of pandas DataFrames for each strata
    :return labels: list of names for each stratum to use in plots
    :return fnames: list of names for each stratum to use to load and save files
    """

    # CHANGE THE LINES BELOW TO ADAPT THEM TO YOUR DATASET
    assert stratification_variable in ("Age", "Age85", "Sex", "Age-Sex", "Age-Sex85")

    age_groups = (0, 25, 45, 65, 85, 101) if stratification_variable in ("Age85", "Age-Sex85") else np.arange(0, 101,
                                                                                                              10)
    age_labels = [f"{age_groups[i]} $\leq$ Age $<$ {age_groups[i + 1]}" for i in range(len(age_groups) - 1)]
    sex_labels = ["Men", "Women"]

    fname = stratification_variable + ("10" if stratification_variable in ('Age', 'Age-Sex') else '')
    if stratification_variable in ("Age", "Age85"):
        masks = [(data["Age"] >= age_groups[i]) & (data["Age"] < age_groups[i + 1]) for i in range(len(age_groups) - 1)]
        labels = age_labels
        fnames = [fname + f"-{i}" for i in range(len(labels))]
    elif stratification_variable == "Sex":
        masks = data["Sex"] == "Men"
        labels = sex_labels
        fnames = [fname + f"-{i}" for i in range(len(labels))]
    elif stratification_variable in ("Age-Sex", "Age-Sex85"):
        masks = [
            [(data["Age"] >= age_groups[i]) & (data["Age"] < age_groups[i + 1]) & (data["Sex"] == j) for j in
             ("Men", "Women")]
            for i in range(len(age_groups) - 1)]
        masks = [m for l in masks for m in l]
        labels = [[f"{label} ({label_s})" for label_s in sex_labels] for label in age_labels]
        labels = [m for l in labels for m in l]
        fnames = [[fname + f"-{i}-{j}" for j in range(len(sex_labels))] for i in range(len(age_labels))]
    else:
        raise Exception(f"Group '{stratification_variable}' not understood.")

    masks = masks if type(masks) == list else (masks, ~masks)
    dfs = [data[m] for m in masks]

    return dfs, labels, fnames


def beautify_name(name: str, sep: str = " ", short: bool = False) -> str:
    """
    Function to beautify names of LTCs for tables and figures.
    
    :param name: name to be beautified
    :param sep: separator to use between words
    :param short: for a shorter version of the names, useful in figures
    :return: beautified name
    """
    beautify_dict = {
        "ActiveAsthma": "Asthma (currently treated)",
        "CHD": "Coronary heart disease (CHD)" if not short else "Coronary heart disease",
        "RheumatoidArthritisEtc": "Rheumatoid arthritis, other inflammatory polyarthropathies & systematic connective tissue disorders" if not short else "Rheumatoid arthritis",
        "COPD": "Chronic obstructive pulmonary disease (COPD)" if not short else "COPD",
        "AnxietyEtc": "Anxiety & other neurotic, stress related & somatoform disorders" if not short else "Anxiety related disorders",
        "IrritableBowelSyndrome": "Irritable bowel syndrome (IBS)" if not short else "Irritable bowel syndrome",
        "AnyCancer_Last5Yrs": "New diagnosis of cancer in last five years" if not short else "Any cancer last 5 years",
        "Other Psychoactive Misuse": "Other psychoactive substance misuse",
        "StrokeTIA": "Stroke & transient ischaemic attack (TIA)" if not short else "Stroke & TIA",
        "CKD": "Chronic kidney disease (CKD)" if not short else "Chronic kidney disease",
        "Diverticular": "Diverticular disease of intestine" if not short else "Diverticular disease",
        "AtrialFib": "Atrial fibrillation (AF)",
        "Prostate": "Prostate disorders",
        "Epilepsy": "Epilepsy (currently treated)",
        "SchizophreniaBipolar": "Schizophrenia (and related non-organic psychosis) or bipolar disorder" if not short else "Schizophrenia or bipolar",
        "PsoriasisEczema": "Psoriasis or eczema",
        "Blindness": "Blindness & low vision",
        "AnorexiaBulimia": "Anorexia or bulimia",
        "Parkinsons": "Parkinson’s disease",
        "ViralHepatitis": "Viral Hepatitis",
    }

    return beautify_dict.get(name, sep.join(
        [word if i == 0 else word.lower() for i, word in enumerate(re.findall('[A-Z]+[a-z]*|\d+', name))]))


def beautify_index(table: pd.DataFrame, **beautify_name_kwargs) -> pd.DataFrame:
    """
    Takes a table with index as name pairs separated by a dash and beautify each name of the pair.

    :param table: table with index as name pairs separated by a dash and beautify each name of the pair
    :param beautify_name_kwargs: additional keyword arguments to pass to the `beautify_name` function
    :return: table with the index as *beautified* name pairs, still separated by a dash
    """
    table.index = [" - ".join([beautify_name(nam, **beautify_name_kwargs) for nam in re.findall('[^-]+', name)]) for
                   name in list(table.index)]
    return table


if __name__ == "__main__":
    from ABC.utils import MLTC_count

    df, names = load_dataset(nrows=1e5)
    adj = np.array(MLTC_count(df, names), dtype=int)
