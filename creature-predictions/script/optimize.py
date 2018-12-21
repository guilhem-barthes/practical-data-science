#!/bin/env python3

import pandas as pd
import numpy as np
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sns.set(style="ticks", palette="pastel")


# list of the different models we want to try
def get_models_list():
    return (
        ('SVC', SVC(),
         {
            'C': [0.01, 0.1, 1.0, 10],
             'kernel': ['poly', 'rbf', 'linear', 'sigmoid'],
            'degree': [x for x in range(1, 6)],
            'coef0': [-5, 0, 5],
            'tol': [np.float_(1.0e-16),
                    np.float_(1.0e-12),
                    np.float_(1.0e-8),
                    np.float_(1.0e-4),
                    np.float_(1.0e-2)],
        }),
        # ('Linear SVC', LinearSVC(),
        #  [{
        #      'C': [0.01, 0.1, 1.0, 10],
        #      'penalty': ['l1'],
        #      'loss': ['squared_hinge'],
        #      'dual': [False],
        #      'multi_class': ['ovr', 'crammer_singer'],
        #      'tol': [np.float_(1.0e-16),
        #              np.float_(1.0e-12),
        #              np.float_(1.0e-8),
        #              np.float_(1.0e-4),
        #              np.float_(1.0e-2)],
        #  },
        #     {'penalty': ['l2'],
        #      'C': [0.01, 0.1, 1.0, 10],
        #      'loss': ['squared_hinge'],
        #      'dual': [False, True],
        #      'multi_class': ['ovr', 'crammer_singer'],
        #      'tol': [np.float_(1.0e-16),
        #              np.float_(1.0e-12),
        #              np.float_(1.0e-8),
        #              np.float_(1.0e-4),
        #              np.float_(1.0e-2)],
        #      }, {
        #      'penalty': ['l2'],
        #      'C': [0.01, 0.1, 1.0, 10],
        #      'loss': ['hinge', ],
        #      'dual': [True],
        #      'multi_class': ['ovr', 'crammer_singer'],
        #      'tol': [np.float_(1.0e-16),
        #              np.float_(1.0e-12),
        #              np.float_(1.0e-8),
        #              np.float_(1.0e-4),
        #              np.float_(1.0e-2)],
        #  }]),
        # ('Nu SVC', NuSVC(),
        #  {
        #     'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
        #      'kernel': ['poly', 'rbf', 'linear', 'sigmoid'],
        #     'degree': [x for x in range(1, 6)],
        #     'tol': [np.float_(1.0e-16),
        #             np.float_(1.0e-12),
        #             np.float_(1.0e-8),
        #             np.float_(1.0e-4),
        #             np.float_(1.0e-2)],
        # }),
        # ('Decision Tree Classifier', DecisionTreeClassifier(),
        #  {
        #     'criterion': ['gini', 'entropy'],
        #      'splitter': ['best', 'random'],
        #     'max_depth': [None, 1, 5, 9],
        #     'max_features': ['auto', 'sqrt', 'log2', None],
        #     'min_samples_split': [x for x in range(1, 10)]
        # }),
        # ('Logistic Regression CV', LogisticRegressionCV(),
        #  {
        #     'Cs': [x for x in range(10)],
        #      'fit_intercept': [True, False],
        #     'dual': [True, False],
        #     'solver': ['newton-cg', 'sag', 'saga', 'lbfgs']
        # }),
        # ('Naïve Bayes Gaussian NB', GaussianNB(), {}),
        # ('KNeighbors Classifier', KNeighborsClassifier(),
        #  {
        #     'n_neighbors': [x for x in range(1, 12)],
        #      'weights': ['uniform', 'distance'],
        #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        # }),
        # ('LDA', LinearDiscriminantAnalysis(), [{
        #     'solver': ['svd'],
        #     'tol': [np.float_(1.0e-16),
        #             np.float_(1.0e-12),
        #             np.float_(1.0e-8),
        #             np.float_(1.0e-4),
        #             np.float_(1.0e-2)],
        #     'n_components': [x for x in range(1, 11)]
        # },
        #     {
        #     'solver': ['lsqr'],
        #     'shrinkage': [None, 'auto', 0.1, 0.5, 0.9],
        #     'n_components': [x for x in range(1, 11)]
        # }])
    )


def get_model_comparison_plot(entry_data, models_list, output_file=None):
    # Classification var
    Y_flat = entry_data['type'].values.flatten()
    X_no_cat = pd.get_dummies(entry_data.drop('type', axis=1))

    # Save results and names
    results = pd.DataFrame(columns=('name', 'value'))

    # Test loop
    for name, model, params in models_list:
        kfold = model_selection.RepeatedStratifiedKFold(n_splits=10,
                                                        n_repeats=1000)
        grid = model
        if params is not None:
            grid = model_selection.GridSearchCV(model, params, cv=kfold,
                                                scoring="accuracy")
            grid.fit(X_no_cat.values, Y_flat)
        cv_results = model_selection.cross_val_score(grid, X_no_cat.values,
                                                     Y_flat,
                                                     scoring='accuracy')

        if params is not None:
            name = name + ' ' + str(grid.best_params_)

        for value in cv_results:
            results = results.append({'name': name, 'value': value},
                                     ignore_index=True)
        print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

    sns.boxplot(x="value", y="name", data=results)
    plt.show()

    if output_file is not None:
        results.to_csv(output_file)


def get_final_model(args, entry_data):
    final_model = None
    models_list = get_models_list()
    if args.tune != "None" and len(args.tune) >= 1:
        for model in models_list:
            if args.tune in model:
                final_model = tune_hyperparameters(entry_data, model,
                                                   iterations=args.it)
                print("Best parameters :", final_model.best_params_)
                print("Best score :", final_model.best_score_)

    return final_model


def tune_hyperparameters(entry_data, model_data, iterations=1000):
    Y_flat = entry_data['type'].values.flatten()
    X = entry_data.drop('type', axis=1)
    X_no_cat = pd.get_dummies(X)

    kfold = model_selection.RepeatedStratifiedKFold(n_splits=10,
                                                    n_repeats=iterations)

    model_name, model, params = model_data
    grid = model_selection.GridSearchCV(model, params, cv=kfold, n_jobs=-1,
                                        scoring="accuracy")
    grid.fit(X_no_cat.values, Y_flat)

    return grid
