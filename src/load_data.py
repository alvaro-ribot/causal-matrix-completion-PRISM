import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#load PRISM dataset and return preprocessed matrix
def load_matrix(seed = 1234, killer_threshold = 1):

    #load cv dataset
    X = pd.read_csv('primary-screen-replicate-collapsed-logfold-change.csv', index_col=0)

    # remove cells *_FAILED_STR
    failed = [f for f in X.index if 'FAILED_STR' in f]
    X.drop(failed, inplace = True)

    # we remove columns that have more than "bad" nans
    good_cols = []
    bad = 5
    na_columns = X.isna().sum(axis = 0) # number of nan per column
    for i, v in enumerate(na_columns):
        if v < bad: good_cols.append(i)
            
    x = X[X.columns[good_cols]].dropna(axis=0).to_numpy()

    # shuffle data
    random.seed(seed)
    np.random.seed(seed)
    perm_rows = np.random.permutation(x.shape[0])
    perm_cols = np.random.permutation(x.shape[1])

    x = x[perm_rows, :]
    x = x[:, perm_cols]

    # delete drugs that kill more than "killer_threshold"% of cells
    if killer_threshold != 1:
        killers = []
        killers = np.sum(np.sign(x) == -1, axis = 0) >= x.shape[0]*killer_threshold 
        killers = [d for d in range(x.shape[1]) if killers[d]]

        x = np.delete(x, killers, 1)

    n = np.min(x.shape)

    x = x[:n, :n] # to have a square matrix
    x = x.T 
    # rows: drugs
    # cols: cancer cell-line

    return x


# mask of observed entries, W[i,j] = 1 iff (i,j) is observed
def observed_entries(x, exponent = 1.7, squared = False, percentage = 0.8):
    n = len(x)
    if squared:
        train = int(n*percentage)
        W = np.ones(x.shape)
        W[train:, train:] = 0
    else:
        W = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                W[i,j] = i + 1 < n**exponent/(j+1)
    
    return W






# LUNG CASE

def load_lung(seed, killer_threshold = 1):
     #load cv dataset
    X = pd.read_csv('primary-screen-replicate-collapsed-logfold-change.csv', index_col=0)

    # remove cells *_FAILED_STR
    failed = [f for f in X.index if 'FAILED_STR' in f]
    X.drop(failed, inplace = True)

    # we remove columns that have more than "bad" nans
    good_cols = []
    bad = 5
    na_columns = X.isna().sum(axis = 0) # number of nan per column
    for i, v in enumerate(na_columns):
        if v < bad: good_cols.append(i)
            
    X = X[X.columns[good_cols]].dropna(axis=0)

    # lung cell lines
    cline = pd.read_csv('primary-screen-cell-line-info.csv', index_col=0)
    lung_cells = cline[cline['primary_tissue'] == 'lung'].index

    x = X.loc[lung_cells].to_numpy()


    # shuffle data
    random.seed(seed)
    np.random.seed(seed)
    perm_rows = np.random.permutation(x.shape[0])
    perm_cols = np.random.permutation(x.shape[1])

    x = x[perm_rows, :]
    x = x[:, perm_cols]

    # delete drugs that kill more than "killer_threshold"% of cells
    if killer_threshold != 1:
        killers = []
        killers = np.sum(np.sign(x) == -1, axis = 0) >= x.shape[0]*killer_threshold 
        killers = [d for d in range(x.shape[1]) if killers[d]]

        x = np.delete(x, killers, 1)

    n = np.min(x.shape)

    x = x[:n, :n] # to have a square matrix
    x = x.T 
    # rows: drugs
    # cols: cancer cell-line

    return x