import numpy as nu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data cleaning/engineering functions


# EDA functions
def df_info(df, target, mv=0.2):
    total = len(df)
    miss_sums = df.isna().sum()
    indices = miss_sums.index
    miss = [ idx for idx in indices if miss_sums[idx] != 0]
    no_miss = [ idx for idx in indices if miss_sums[idx] == 0]

    clss = len(df[target].unique())
    cl_val = [ f'{idx}' + f' - {df[target].value_counts()[idx]}' for idx in df[target].unique() ]

    print(f'Target variable: {target}')
    print('----------------')
    print(f'Classes: {clss}')
    print(f'Imbalance: {", ".join(cl_val)}\n')
    print(f'No missing values: {", ".join(no_miss)}\n')
    print('Values Missing:')
    print('---------------')
    for idx in miss:
        if miss_sums[idx]/total > mv:
            print(f'{idx}: {miss_sums[idx]} ({round(100*miss_sums[idx]/total, 2)}%) ***')
        else:
            print(f'{idx}: {miss_sums[idx]} ({round(100*miss_sums[idx]/total, 2)}%)')

