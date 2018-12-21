# coding: utf8
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# General correlation map
def get_cormat_map_gen(df, output=None):
    cormat = df.corr()
    sns.heatmap(cormat, vmax=.85, square=True)
    if output is None:
        plt.show()
    else:
        plt.savefig(output)


def get_cormat_map_most_close(df, n, var, output=None):
    cormat = df.corr()
    cols = cormat.nlargest(n, var)[var].index
    largcorrmat = np.corrcoef(df[cols].values.T)
    sns.heatmap(largcorrmat, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 10}, yticklabels=cols.values,
                xticklabels=cols.values)
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
