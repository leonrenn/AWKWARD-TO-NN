"""
File that contains utilities to process the data.
"""

import os
import random
from typing import Dict, List, Tuple

import awkward as ak
import numpy as np
import pandas as pd
import torch
import uproot as ur
from torch.utils.data import Dataset


def get_array_from_pMSSM_point(path: str) -> pd.DataFrame:
    """
    Get an awkward array from a pMSSM point.
    1. Check if the file exists.
    2. Check if the file is a root file.
    3. Open the file and get the tree and return the array.

    Args:
        path (str): Path to the root file.

    Returns:
        pd.DataFrame: The awkward array.
    """
    # 1. Check if the file exists.
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found!")
    # 2. Check if the file is a root file.
    if not path.endswith(".root"):
        raise TypeError(f"File {path} is not a root file!")
    # 3. Open the file and get the tree and return the array.
    return ur.open(path)["ntuple"].arrays(library="pd")


def from_df_2_df(df: pd.DataFrame,
                 remove_fields: List[str],
                 normalize: bool) -> Tuple[pd.DataFrame,
                                           Dict[str, float]]:
    """
    Convert awkward array (in dataframe format) to pandas dataframe
    that can be used for further processing.

    1. Normalization factors.
    2. Remove fields.
    3. Remove all columns that just contain zeros.
    4. Iterate over all columns and pad them to the same length.
    5. Normalize the data (necessary for the classifier).

    Example of dataframe:

    Event | MET | Jet | el_pt | ... |
    ---------------------------------
    1     | 12   | 5   |  [12, 23, -1] | ... |
    2     | 23   | 2   |  [12, 23, 34] | ... |

    Args:
        df (pd.DataFrame): The dataframe that was loaded from uproot.
        remove_fields (List[str]): List of fields that should be removed.
        normalize (bool): Normalize the data.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: The dataframe and
            the normalization factors.

    """
    # 1. Normalization factors.
    if normalize is True:
        norm_factors: Dict[str, float] = {}
    else:
        norm_factors = None

    # 2. Remove fields.
    try:
        df.drop(remove_fields, axis=1, inplace=True)
    except KeyError:
        print("Could not remove all fields.")

    # 3. Remove all columns that just contain zeros.
    df = df.loc[:, (df != 0).any(axis=0)]

    # 4. Iterate over all columns and pad them to the same length.
    for col in df.columns:
        if isinstance(df[col][0], ak.highlevel.Array):
            # get lenght for every entry if it is a list
            try:
                df[f"{col}_len"] = df[col].apply(
                    lambda x: len(x) if isinstance(x, list) else x)
                max_len = df[f"{col}_len"].max()
                if max_len == 0:
                    # remove column if it is empty
                    df.drop([col, f"{col}_len"], axis=1, inplace=True)
                else:
                    # make all lists the same size and fill up with 0
                    df[col] = df[col].apply(
                        lambda x: x + [0]*(max_len-len(x))
                        if isinstance(x, list) else x)
            except Exception:
                print(f"Could not convert {col} to list.")

    # 5. Normalize the data (necessary for the classifier).
    if normalize is True:
        for col in df.columns:
            if isinstance(df[col][0], list):
                norm_factors[col] = df[col].apply(lambda x: max(
                    np.absolute(x)) if len(x) > 0 else 0).max()
                df[col] = df[col].apply(lambda x: x/norm_factors[col])
            else:
                norm_factors[col] = df[col].abs().max()
                df[col] = df[col].apply(lambda x: x/norm_factors[col])

    return df, norm_factors


def sr_acc_rej(path_data: str,
               path_info: str) -> pd.DataFrame:
    """
    Dataframe with all signal regions in pMSSMFactory with entries of either
    accteped or rejected events.

    0. All signal regions.
    1. Empty dataframe.
    2. Check if directory exists.
    3. Iterate over all files.
        1. Check if file is a root file.
        1. Check if file exists in info directory.
            1. Open root file.
            2. Get arrays from root file.
            3. Convert awkward array to pandas dataframe and append to
               dataframe.
    4. Assign new column names.
    5. Remove empty columns.
    6. Accepted or rejected events.

    Example of dataframe:

    Event | SR_0 | SR_1 |
    ---------------------
    1     | 1    | 0    |
    2     | 0    | 1    |
    3     | 0    | 0    |

    Args:
        path_data (str): Path to the directory with the root files.
        path_info (str): Path to the directory with the info files.

    Returns:
        pd.DataFrame: Dataframe with all signal regions in pMSSMFactory with
            entries of either accteped or rejected events.

    """
    # 0. All signal regions.
    signal_regions: List[str] = []
    # 1. Empty dataframe.
    df = pd.DataFrame()
    # 2. Check if directory exists.
    if os.path.isdir(path_data) is False:
        raise ValueError("Directory does not exist.")
    else:
        data_files = os.listdir(path_data)
    if os.path.isdir(path_info) is False:
        raise ValueError("Directory does not exist.")
    else:
        info_files = os.listdir(path_info)
    # 3. Iterate over all files.
    for file in data_files:
        # 1. Check if file is a root file.
        if file.endswith(".root"):
            # 1. Check if file exists in info directory.
            if f"{file.strip('.root')}.info" in info_files:
                srs = pd.read_csv(
                    path_info + f"{file.strip('.root')}.info")["SR"].to_list()
                # 1. Open root file.
                root_file: ur.ReadOnlyFile = ur.open(path_data + file)
                # 2. Get arrays from root file.
                ttree_arrays = root_file["ntuple"].arrays(srs)
                signal_regions += [f"{file.strip('.root')}@{sr}" for sr in srs]
                # 3. Convert awkward array to pandas dataframea and append to
                # dataframe.
                df = pd.concat([df, ak.to_dataframe(ttree_arrays)], axis=1)
    # 4. Assign new column names.
    df.columns = signal_regions
    # 5. Remove empty columns.
    df = df.loc[:, (df != 0).any(axis=0)]
    # 6. Accepted or rejected ints.
    df = df.astype(bool)
    return df


def sample_events_signal_regions(df_acc_rej: pd.DataFrame,
                                 cut_acc_events: int) -> Dict[str, Dict]:
    """
    Generate a sample of events with equal amount of accepted and rejected
    events for each signal region. The sample is generated by randomly
    selecting events from the accepted and rejected events.
    The equal distribution of accepted and rejected events is necessary for
    the classifier.

    0. Set seed.
    1. Mask for signal regions with more accepted events than threshold.
    2. Selected signal regions from mask.
    3. Selected signal regions in dataframe.
    4. Divide into accepted and rejetced indices.
    5. Draw a sample from accepted and rejected indices.

    Args:
        df_acc_rej (pd.DataFrame): Dataframe with all signal regions in
            pMSSMFactory with entries of either accteped or rejected events.
        cut_acc_events (int): Threshold for the number of accepted events.

    Returns:
        Dict[str, Dict]: Dictionary with signal regions as keys and a
            dictionary with accepted and rejected indices as values.
    """
    # 0. Set seed.
    random.seed(42)

    # 1. Mask for signal regions with more accepted events than threshold.
    mask = df_acc_rej.sum() > cut_acc_events

    # 2. Selected signal regions from mask.
    sel_sr = np.array(df_acc_rej.columns)[mask]

    # 3. Selected signal regions in dataframe.
    df_acc_rej_sel = df_acc_rej[sel_sr]

    # 4. Divide into accepted and rejetced indices.
    acc_rej_index = {}
    for signal_region in df_acc_rej_sel.columns:
        acc_rej_index[signal_region] = {"acc": [],
                                        "rej": []}
        acc_rej_index[signal_region]["acc"] = \
            df_acc_rej_sel[signal_region][df_acc_rej_sel[signal_region]
                                          ].index.tolist()

        acc_rej_index[signal_region]["rej"] = \
            df_acc_rej_sel[signal_region][~df_acc_rej_sel[signal_region]
                                          ].index.tolist()

    # 5. Draw a sample from the accpeted and rejected indices.
    sample_accepted_index = {}
    for signal_region in acc_rej_index.keys():
        sample_accepted_index[signal_region] = {"acc": [],
                                                "rej": []}
        sample_accepted_index[signal_region]["acc"] = random.choices(
            acc_rej_index[signal_region]["acc"], k=cut_acc_events)
        sample_accepted_index[signal_region]["rej"] = random.choices(
            acc_rej_index[signal_region]["rej"], k=cut_acc_events)

    return sample_accepted_index


def parameterize_df_with_sel_events(df: pd.DataFrame,
                                    sample_indices: Dict[str, Dict]) -> Tuple:
    """
    Parameterizing the events with the selected events from the
    sample_events_signal_regions function.

    0. Make signal regions usable as floats.
    1. Zip the singal region names with the values to get a dict.
    2. Add signal region and accepted column.
    3. Initialize empty dataframe.
    4. Iterate over signal regions and add the accepted and rejected events
        to the dataframe.

    Example of dataframe:
    Event | MET | Jet | el_pt | ... | SR | Accepted |
    -------------------------------------------------
    1     | 12   | 5   |  [12, 23, -1] | ... | SR_0 | True |
    2     | 23   | 2   |  [12, 23, 34] | ... | SR_0 | False |
    .
    .
    .
    n     | 34   | 1   |  [12, 23, 45] | ... | SR_k | False |

    Args:
        df (pd.DataFrame): Dataframe with events.
        sample_indices (Dict[str, Dict]): Dictionary with signal regions as
            keys and a dictionary with accepted and rejected indices as values.

    Returns:
        Tuple: Tuple with dataframe with parameterized events and a dictionary
            with the signal regions as keys and the signal region values as
            values.
    """

    # 0. Make signal regions usable as floats.
    lenght_srs = len(sample_indices.keys())
    signal_regions = np.linspace(0, 1, lenght_srs)

    # 1. Zip the singal region names with the values to get a dict.
    srs_values = dict(zip(sample_indices.keys(), signal_regions))

    # 2. Add signal region and accepted columns to dataframe.
    df["signal_region"] = 0.0
    df["accepted"] = 0

    # 3. Initialize empty dataframe.
    df_complete = pd.DataFrame()

    # 4. Iterate over signal regions and add the accepted and rejected events
    # to the dataframe.
    for signal_region in sample_indices.keys():

        df_acc = df.iloc[sample_indices[signal_region]["acc"]]
        df_acc["signal_region"] = srs_values[signal_region]
        df_acc["accepted"] = 1

        df_rej = df.iloc[sample_indices[signal_region]["rej"]]
        df_rej["signal_region"] = srs_values[signal_region]
        # df_rej["accepted"] = False <- already false

        df_complete = pd.concat([df_complete, df_acc, df_rej], axis=0)

    return df_complete, srs_values


class DatasetMLP(Dataset):
    def __init__(self,
                 df_complete: pd.DataFrame) -> None:
        super().__init__()

        # 1. Initialize dataframe.
        self.df_complete = df_complete

        # 2. Iterate over columns and expand list columns to multiple columns.
        for col in self.df_complete.columns:
            if isinstance(self.df_complete[col].values[0], np.ndarray):
                for i in range(len(self.df_complete[col].values[0])):
                    self.df_complete[col + str(i)] = (self.df_complete[col]
                                                      .apply(lambda x: x[i]))
                self.df_complete = self.df_complete.drop(col, axis=1)

    def __len__(self) -> int:
        return len(self.df_complete)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Target for classification.
        y = torch.tensor(
            self.df_complete["accepted"].values, dtype=torch.float32)
        # 2. Features for classification.
        x = torch.tensor(self.df_complete.drop(
            ["accepted"], axis=1).values, dtype=torch.float32)
        return x, y
