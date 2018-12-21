#!/bin/env python3
# coding: utf8

import pandas as pd
import argparse

from script.visualization import get_descriptive_plots
from script.calculation import student_test, predict_values
from script.optimize import get_final_model


def main_calculations():
    # Argument parser
    parser = argparse.ArgumentParser(description="Select which operation you \
                                    want to do")
    parser.add_argument('--desc', type=bool, default=True,
                        choices=[False, True], help='Describe the dataset')
    parser.add_argument('--student', type=bool, default=True,
                        help="Compare models with Student test")
    parser.add_argument('--tune', type=str, default="LDA", help="Optimize the\
                        model. Set None to not optimize the model")
    parser.add_argument('--predict', type=bool, default=True,
                        choices=[False, True], help='Predict train var')
    parser.add_argument('--decimals', type=int, default=4,
                        help='Number of decimals in the output file')
    parser.add_argument('--it', type=int, default=10000,
                        help='K-fold iterations number')
    parser.add_argument('--save', default=True,  choices=[False, True],
                        type=bool, help='Show results(False)/save them (True)')

    args = parser.parse_args()

    # Load data and merge them
    sizes = pd.read_csv('./input/sizes.csv', index_col=0, header=0, sep=",")
    colors = pd.read_csv('./input/colors.csv', index_col=0, header=0, sep=",")
    types = pd.read_csv('./input/types.csv', index_col=0, header=0, sep=",")
    entry_data = sizes.join(colors, how="outer")
    entry_data = entry_data.join(types, how="inner")

    if args.desc:
        get_descriptive_plots(entry_data, out=args.save)

    if args.student:
        student_test(entry_data, out=args.save, decimals=args.decimals,
                     iterations=args.it)

    final_model = get_final_model(args, entry_data)

    # Load test datas and predict classes
    if args.predict and final_model is not None:
        predict_values(final_model, './test.csv', out=args.save)
    elif args.predict and args.tune == "None":
        print("Predictions work only with tuned models (--tune)")


if __name__ == '__main__':
    main_calculations()
