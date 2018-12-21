#!/bin/env python3

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", palette="pastel")


def get_descriptive_plots(entry_data, out=False):
    # Classification var
    X_no_cat = pd.get_dummies(entry_data.drop('type', axis=1))

    # Calculate correlation
    corr = X_no_cat.corr()

    # Descriptive statistics
    fig, axs = plt.subplots(figsize=(18, 10), nrows=3, ncols=2)
    plt.subplots_adjust(hspace=0.3)

    sns.countplot(x="type", data=entry_data, ax=axs[0, 0])
    sns.countplot(y="type", hue="color", data=entry_data, ax=axs[0, 1])
    sns.boxplot(x="rotting_flesh", y="type", data=entry_data, ax=axs[1, 0])
    sns.boxplot(x="bone_length", y="type", data=entry_data, ax=axs[1, 1])
    sns.boxplot(x="hair_length", y="type", data=entry_data, ax=axs[2, 0])
    sns.boxplot(x="has_soul", y="type", data=entry_data, ax=axs[2, 1])

    if out:
        plt.savefig('./desc.png')
        print("Data description plot saved in desc.png")
    else:
        plt.show()

    fig, axs = plt.subplots(figsize=(18, 10), nrows=1, ncols=2)
    plt.subplots_adjust(hspace=0.3)
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, ax=axs[0])
    pca = PCA().fit(X_no_cat.values)
    sns.lineplot(x=range(len(pca.explained_variance_ratio_)),
                 y=np.cumsum(pca.explained_variance_ratio_), ax=axs[1])\
        .set_xticks(range(len(pca.explained_variance_ratio_)))
    plt.xlabel('Components number')
    plt.ylabel('Cumulative explained variance')

    if out:
        plt.savefig('./correlation.png')
        print("Correlation plot saved in correlation.png")
    else:
        plt.show()
