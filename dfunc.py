import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# EDA Functions
def df_info(df, target, mv=0.2):
    '''
    General info on features and target
    '''
    total = len(df)
    miss_sums = df.isna().sum()
    indices = miss_sums.index
    miss = [ idx for idx in indices if miss_sums[idx] != 0 ]
    no_miss = [ idx for idx in indices if miss_sums[idx] == 0 ]

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

    r_a = round(100 * a / (a + b), 2)
    r_b = round(100 * b / (b + a), 2)
    r_c = round(100 * c / (c + d), 2)
    r_d = round(100 * d / (c + d), 2)

    chi_2, p = stats.chisquare([a, b, c, d], [e_a, e_b, e_c, e_d], ddof=1)
    chi_2 = round(chi_2, 4)
    p = round(p, 4)
    if chi_2 > 3.8415:
        print('Reject Null Hypothesis')
    else:
        print('Cannot Reject Null Hypothesis')
    print(f'Chi-Squared: {chi_2}')
    print(f'p-value: {p}\n')
    print(f'Target True when NaN: {r_a}%')
    print(f'Target False when NaN: {r_b}%')
    print(f'Target True when non-null: {r_c}%')
    print(f'Target False when non-null: {r_d}%')

# Data Engineering Functions
def feat_to_dum(df, feature, s_value=np.nan, pref=None):
    '''
    Function to convert NaN values to specified value
    Parameters
    ----------
    feature : str
        feature to dummy
    s_value : str, int, float
        value to inpute in case of NaN
    pref : str
        prefix for dummy variables 
    '''
    condition = df[feature].isna() == True
    df[feature] = np.where(condition, s_value, df[feature])
    df[feature] = df[feature].astype('category')
    dummies = pd.get_dummies(df[feature], prefix=pref, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=feature, inplace=True)

    print(f'Feature Dummied and Dropped: {feature}')
    return df