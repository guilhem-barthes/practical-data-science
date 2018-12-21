#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
values = [
    ["MSSubClass", "Categorical"],
    ["MSZoning", "Categorical"],
    ["LotFrontage", "Numerical"],
    ["LotArea", "Numerical"],
    ["Street", "Categorical"],
    ["Alley", "Categorical"],
    ["LotShape", "Ordinal"],
    ["LandContour", "Categorical"],
    ["Utilities", "Ordinal"],
    ["LotConfig", "Categorical"],
    ["LandSlope", "Ordinal"],
    ["Neighborhood", "Categorical"],
    ["Condition1", "Categorical"],
    ["Condition2", "Categorical"],
    ["BldgType", "Categorical"],
    ["HouseStyle", "Categorical"],
    ["OverallQual", "Numerical"],
    ["OverallCond", "Numerical"],
    ["YearBuilt", "Numerical"],
    ["YearRemodAdd", "Numerical"],
    ["RoofStyle", "Categorical"],
    ["RoofMatl", "Categorical"],
    ["Exterior1st", "Categorical"],
    ["Exterior2nd", "Categorical"],
    ["MasVnrType", "Categorical"],
    ["MasVnrArea", "Numerical"],
    ["ExterQual", "Ordinal"],
    ["ExterCond", "Ordinal"],
    ["Foundation", "Categorical"],
    ["BsmtQual", "Ordinal"],
    ["BsmtCond", "Ordinal"],
    ["BsmtExposure", "Ordinal"],
    ["BsmtFinType1", "Ordinal"],
    ["BsmtFinSF1", "Numerical"],
    ["BsmtFinType2", "Ordinal"],
    ["BsmtFinSF2", "Numerical"],
    ["BsmtUnfSF", "Numerical"],
    ["TotalBsmtSF", "Numerical"],
    ["Heating", "Categorical"],
    ["HeatingQC", "Ordinal"],
    ["CentralAir", "Ordinal"],
    ["Electrical", "Categorical"],
    ["1stFlrSF", "Numerical"],
    ["2ndFlrSF", "Numerical"],
    ["LowQualFinSF", "Numerical"],
    ["GrLivArea", "Numerical"],
    ["BsmtFullBath", "Numerical"],
    ["BsmtHalfBath", "Numerical"],
    ["FullBath", "Numerical"],
    ["HalfBath", "Numerical"],
    ["BedroomAbvGr", "Numerical"],
    ["KitchenAbvGr", "Numerical"],
    ["KitchenQual", "Numerical"],
    ["TotRmsAbvGrd", "Numerical"],
    ["Functional", "Ordinal"],
    ["Fireplaces", "Numerical"],
    ["FireplaceQu", "Ordinal"],
    ["GarageType", "Categorical"],
    ["GarageYrBlt", "Numerical"],
    ["GarageFinish", "Ordinal"],
    ["GarageCars", "Numerical"],
    ["GarageArea", "Numerical"],
    ["GarageQual", "Ordinal"],
    ["GarageCond", "Ordinal"],
    ["PavedDrive", "Ordinal"],
    ["WoodDeckSF", "Numerical"],
    ["OpenPorchSF", "Numerical"],
    ["EnclosedPorch", "Numerical"],
    ["3SsnPorch", "Numerical"],
    ["ScreenPorch", "Numerical"],
    ["PoolArea", "Numerical"],
    ["PoolQC", "Ordinal"],
    ["Fence", "Ordinal"],
    ["MiscFeature", "Categorical"],
    ["MiscVal", "Numerical"],
    ["MoSold", "Numerical"],
    ["YrSold", "Numerical"],
    ["SaleType", "Categorical"],
    ["SaleCondition", "Categorical"]
]
df = pd.DataFrame(values)
df.transpose()
df.columns = ["Name", "Type"]
sns.countplot(y="Type",  data=df)
plt.show()