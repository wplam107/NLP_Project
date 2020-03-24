import numpy as nu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# EDA functions
def df_info(df, target, mv=0.2):
    '''
    General info on features and target
    '''
    total = len(df)
    miss_sums = df.isna().sum()
    indices = miss_sums.index
    miss = [ idx for idx in indices if miss_sums[idx] != 0]
    no_miss = [ idx for idx in indices if miss_sums[idx] == 0]

    clss = len(df[target].unique())
    cl_val = [ f'{idx}' + f' - {df[target].value_counts()[idx]}' for idx in df[target].unique() ]
    cl_val_norm = [ f'{idx}' + f' - {round(100*df[target].value_counts(normalize=True)[idx], 2)}%' for idx in df[target].unique() ]

    print(f'Total Observations: {total}')
    print(f'Target Variable: {target}')
    print(f'Classes: {clss}')
    print(f'Imbalance: {", ".join(cl_val)}')
    print(f'Imbalance Ratio: {", ".join(cl_val_norm)}\n')
    print(f'No missing values: {", ".join(no_miss)}\n')
    print('Values Missing:')
    print('---------------')
    for idx in miss:
        if miss_sums[idx]/total > mv:
            print(f'{idx}: {miss_sums[idx]} ({round(100*miss_sums[idx]/total, 2)}%) ***')
        else:
            print(f'{idx}: {miss_sums[idx]} ({round(100*miss_sums[idx]/total, 2)}%)')

def chi_sq(df, feature='', target=''):
    '''
    Chi-Squared test for single dummy feature, uses alpha = 0.05 and ddof = 1
    Parameters
    ----------
    feature : str
        feature column to inspect as str
    target : str
        target variable
    '''
    a = len(df.loc[(df[feature].isna() == True) & (df[target] == 1)])
    b = len(df.loc[(df[feature].isna() == True) & (df[target] == 0)])
    c = len(df.loc[(df[feature].isna() == False) & (df[target] == 1)])
    d = len(df.loc[(df[feature].isna() == False) & (df[target] == 0)])
    total = a + b + c + d
    
    e_a = (a + b)*(a + c) / total
    e_b = (a + b)*(b + d) / total
    e_c = (c + d)*(a + c) / total
    e_d = (c + d)*(b + d) / total
    
    chi_2, p = stats.chisquare([a, b, c, d], [e_a, e_b, e_c, e_d], ddof=1)
    chi_2 = round(chi_2, 4)
    p = round(p, 4)
    if chi_2 > 3.8415:
        print('Reject Null Hypothesis')
    else:
        print('Cannot Reject Null Hypothesis')
    print(f'Chi-Squared: {chi_2}')
    print(f'p-value: {p}')