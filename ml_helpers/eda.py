import matplotlib.pyplot as plt
import fireducks.pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import itertools
import torch
from math import log

from torchmetrics.functional.nominal import theils_u_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kurtosistest, skewtest, normaltest
from sklearn.feature_selection import mutual_info_regression

from .data_cleaning import ordinal_encoding


# only continuous data
def skewness_kurtosis_df(dataframe, verbose=False):
    if verbose:
        print('Skewness:\n\tStatistic (z-score): Positive means right skew, negative means left skew\n\tP-Value < 0.05 = statistically significant skewness') 
        print('Kurtosis:\n\tStatistic (z-score):\n\t\tPositive = leptokurtic (heavy tails)\n\t\t0 = normal distribution\n\t\tNegative = platykurtic\n\tP-Value < 0.05 = statistically significant skewness')

    cols = dataframe.columns.tolist()
    skewnesses = [ skewtest(dataframe[col]) for col in cols ]
    kurtosises = [ kurtosistest(dataframe[col]) for col in cols ]
    return pd.DataFrame([
        {
            'feature': col, 
            'skew_z': skewnesses[i].statistic, 
            'skew_p': skewnesses[i].pvalue,
            'kurtosis_z': kurtosises[i].statistic,
            'kurtosis_p': kurtosises[i].pvalue
        } for i, col in enumerate(cols)
    ])


# only continuous data
def dagostino_normality_df(dataframe):
    stats, pvals = normaltest(dataframe, axis=0)
    return pd.DataFrame({
        'feature': dataframe.columns,
        'stat': stats,
        'pval': pvals,
        'is_normal': pvals > 0.05
    }).sort_values(by='stat', ascending=True)


# mutual info gives correlation score of lines, curves and squiggly lines
def mutual_info_df(dataframe, target):
    X = dataframe.select_dtypes(include=np.number).drop(columns=[target])
    y = dataframe[target]
    return pd.DataFrame({
        'Feature': X.columns,
        'MI Score': mutual_info_regression(X, y)
    }).sort_values(by='MI Score', ascending=False)


def vif_df(df):
    vif = pd.DataFrame()
    vif['feature'] = df.columns
    vif['VIF'] = [ variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif


def plot_hist_ecdf(dataframe, feature_name, nbins=None):
    plt.style.use('dark_background')

    if nbins is None:
        nbins = 1 + int(log(len(dataframe), 2)) # Sturge's Rule

    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle(f'{feature_name} Histogram with eCDF', fontsize=16)
    ax2 = ax1.twinx()
    sns.histplot(data=dataframe, x=feature_name, bins=nbins, ax=ax1)
    sns.ecdfplot(data=dataframe, x=feature_name, color="red", lw=2, ax=ax2)
    plt.show()


def count_plot(dataframe, cat_feat_name):
    plt.style.use('dark_background')
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 5))
    sns.countplot(x=dataframe[cat_feat_name]);
    plt.show()


def plot_theils_u_matrix(df):
    plt.style.use('dark_background')
    df, _ = ordinal_encoding(df, df)
    cols = df.columns.tolist()
    matrix = torch.tensor(df.values)
    corr_matrix = pd.DataFrame(theils_u_matrix(matrix).numpy(), columns=cols, index=cols)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(corr_matrix, linewidth=1, annot=True, fmt=".3f")
    ax.set_title("Theil's U Matrix", fontsize=16, pad=20)
    plt.show()


def scatter_plot(dataframe, feature1_name, feature2_name, point_size=2):
    plt.style.use('dark_background')
    _, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x=dataframe[feature1_name], y=dataframe[feature2_name], c='blue', s=point_size)
    ax.set_xlabel(feature1_name)
    ax.set_ylabel(feature2_name)
    plt.show()


def scatter_plot_3d(dataframe, feat1, feat2, feat3, point_size=1.5):
    fig = go.Figure()
    fig = px.scatter_3d(
        x=dataframe[feat1],
        y=dataframe[feat2],
        z=dataframe[feat3],
        title="Mi 3D plot",
        labels={ "x": feat1, "y": feat2, "z": feat3 }
    )
    fig.update_traces(marker=dict(size=point_size)) 
    fig.show()



# will generate (n_feats^2 - n_feats) / 2 scatter plots
# takes numeric data only
def feature_combinations_scatter_plot(df):
    numeric_features = df.columns.tolist()
    pairs = itertools.combinations(numeric_features, 2)
    for pair in (pairs):
        sns.lmplot(data=df, x=pair[0], y=pair[1],
        line_kws={'color': 'black'},
        aspect=1 )
        plt.show();
