#!/usr/bin/env python3
# coding: utf8

import pandas as pd
from script.graphs import get_cormat_map_gen, get_cormat_map_most_close
from script.preprocess import to_ranges, fill_na_simple, fill_na_crossed
from script.model import get_model


def main_calculations():
    # Loading both datasets : train and test
    train_data = pd.read_csv("./input/train.csv")
    test_x = pd.read_csv('./input/test.csv')
    train_data.set_index("Id")
    test_x.set_index("Id")

    # Get correlations between vars
    get_cormat_map_gen(train_data)
    get_cormat_map_most_close(train_data, 10, 'SalePrice')

    # Drop columns too correlated with other ones
    # TODO: use a stronger method
    cols_to_drop = ['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd']
    test_x.drop(cols_to_drop, axis="columns", inplace=True)
    train_data.drop(cols_to_drop, axis="columns", inplace=True)

    # Preprocessing : replace ordinal values and fill NAs
    to_ranges(test_x)
    to_ranges(train_data)
    fill_na_simple(test_x)
    fill_na_simple(train_data)
    fill_na_crossed(train_data, test_x)

    # Split train dataframe
    train_x = train_data.loc[:, :'SaleCondition']
    train_y = train_data.loc[:, 'SalePrice':]

    # Concat train and test, get dummies and separe them again
    all_rows_x = pd.concat([train_x, test_x], keys=["train", "test"])
    all_rows_x_dummies = pd.get_dummies(all_rows_x)
    train_x_dummies = all_rows_x_dummies.loc['train'].set_index("Id")
    test_x_dummies = all_rows_x_dummies.loc['test'].set_index("Id")

    # Train, predict and save results
    model = get_model(train_x_dummies, train_y)
    test_y = model.predict(test_x_dummies)
    output = pd.DataFrame(data={"SalePrice": test_y}, index=test_x.Id)
    output.to_csv("./output.csv", index_label="Id")


if __name__ == '__main__':
    main_calculations()
