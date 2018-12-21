#!/bin/env python3

import pandas as pd
from sklearn import model_selection
from scipy.stats import ttest_ind

from .optimize import get_models_list


def student_test(entry_data, out=False, optimize=False,
                 decimals=4, iterations=10000):
    models_list = get_models_list()
    Y_flat = entry_data['type'].values.flatten()
    X = entry_data.drop('type', axis=1)
    X_no_cat = pd.get_dummies(X)

    # Save results and names
    results = pd.DataFrame(columns=('name', 'value'))
    results.set_index('name')

    # Test loop
    for name, model, params in models_list:
        kfold = model_selection.RepeatedStratifiedKFold(n_splits=10,
                                                        n_repeats=iterations)

        cv_results = model_selection.cross_val_score(model, X_no_cat.values,
                                                     Y_flat, cv=kfold,
                                                     n_jobs=-1,
                                                     scoring='accuracy')

        results = results.append({'name': name,
                                  'value': cv_results},
                                 ignore_index=True)

    if out:
        results.to_csv('./mean_models.csv', sep=',')
        print(
            'Means by models saved in mean_models.csv')
    else:
        print(results)

    student_results = pd.DataFrame(
        index=results.name.values, columns=results.name.values, dtype='float')
    for i in range(results.shape[0]):
        for j in range(results.shape[0]):
            student_results.set_value(results.iloc[j, 0],
                                      results.iloc[i, 0],
                                      ttest_ind(results.iloc[i, 1],
                                                results.iloc[j, 1]).pvalue)

    if out:
        tmp = student_results.round(decimals)
        tmp.to_csv('./student_pvalues.csv')


def predict_values(model, input_file, out=False):
    X_test = pd.read_csv(input_file, header=0, index_col=0)
    X_test = pd.get_dummies(X_test)
    Y_test = model.predict(X_test)
    XY_test = pd.DataFrame({'id': X_test.index.values, 'class': Y_test})
    XY_test.set_index("id", inplace=True)

    if out:
        XY_test.to_csv('./sample_submission.csv', sep=",")
        print('Predictions saved in sample_submission.csv')
    else:
        print(XY_test)
