#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import glob
import os
import torch
import matplotlib.pyplot as plt

# Read / write data locations
IN_DIR = "../data/raw"
OUT_DIR = "../data/processed"

# Trajectory length threshold
N = 500


def prepare_input_data(dfs_selected):
    """Format dfs as input data tensor of shape (# trajectories, N, 2)"""
    x = []
    for i in range(len(dfs_selected)):
        xi = np.array([dfs_selected[i]['latitude'], dfs_selected[i]['longitude']]).T
        x.append(xi)
    x = np.array(x)

    x = torch.tensor(x)
    return x


def select_trajectories(data, N):
    """Selects dfs of trajectories where len(trajectory) >= N, truncate dfs to N"""
    files_selected = []
    dfs_selected = []
    for file, df in data.items():
        if len(df) >= N:
            files_selected.append(file)
            dfs_selected.append(df[0:N])
    return files_selected, dfs_selected


def read_csvs(files):
    """Reads in csv files to pandas DataFrames, returns dictionary mapping file to DataFrame."""
    data = {}
    for file in files:
        # Skip row with index 1, contains the column units
        df = pd.read_csv(file, skiprows=[1])
        df = df.dropna(how='any', subset=['latitude', 'longitude'])
        df = df.drop_duplicates(subset=['profile_id'], keep='first')
        data[file] = df
    return data


def visualize_trajectory_lens_dist():
    """Helper function to visualize trajectory lens distribution to determine N threshold."""
    lens = [len(df) for df in data.values()]
    print(sorted(lens))
    plt.hist(lens)
    plt.show()  


def main():
    files = glob.glob(os.path.join(IN_DIR, "*.csv"))
    data = read_csvs(files)
    
    files_selected, dfs_selected = select_trajectories(data, N)
    # print(files_selected)

    x = prepare_input_data(dfs_selected)
    # print(x.shape)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    torch.save(x, os.path.join(OUT_DIR, "x.pt"))


if __name__ == '__main__':
    main()