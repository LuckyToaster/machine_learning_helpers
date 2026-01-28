import fireducks.pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from scipy.stats import zscore
from missingno import matrix

def nan_table(df):
    return pd.DataFrame({ 
        'NaN Count': df.isna().sum(), 
        'NaN %': df.isna().mean() * 100 
    }).sort_values(by='NaN Count', ascending=False).reset_index().rename(columns={'index': 'Feature'}) 


def nan_visualization(df):
    plt.style.use('dark_background')
    matrix(df, sparkline=False)
    plt.title('NaN values Visualization', fontsize=16)
    plt.subplots_adjust(top=0.6)
    plt.show()


def drop_duplicate_rows(df):
    df_deduplicated = df.drop_duplicates(keep='first').reset_index()
    print(f'N duplicate rows: {len(df) - len(df_deduplicated)}')
    return df_deduplicated

def drop_zscores(df, z=3):
    z_scores = df.select_dtypes(include=[np.number]).apply(zscore)
    outlier_indices = df[ (np.abs(z_scores) > z).any(axis=1) ].index
    print(f'removed {len(outlier_indices)} from train')
    return df.drop(index=outlier_indices)


# https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
def drop_dup_cols_by_name(df):
    return df.loc[:,~df.columns.duplicated()].copy()

def drop_duplicate_cols(train, test):
    to_drop = []
    cols = train.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if train[cols[i]].equals(train[cols[j]]):
                to_drop.append(cols[j])
    
    to_drop = list(set(to_drop))
    print(f'Duplicate columns: {to_drop}')
    return train.drop(columns=to_drop), test.drop(columns=to_drop)


def one_hot_encoding(train, test):
    encoder = OneHotEncoder(handle_unknown='warn', sparse_output=False)
    encoder.set_output(transform="pandas")
    return encoder.fit_transform(train), encoder.transform(test)
    
def ordinal_encoding(train, test):
    encoder = OrdinalEncoder()
    encoder.set_output(transform='pandas')
    return encoder.fit_transform(train), encoder.transform(test)

def robust_scaling(train, test):
    rs = RobustScaler()
    rs.set_output(transform='pandas')
    return rs.fit_transform(train), rs.transform(test)

def multiple_imputation(train, test, seed, max_iter=10):
    imputer = IterativeImputer(random_state=seed, max_iter=max_iter)
    return imputer.fit_transform(train), imputer.transform(test)

