
def to_ranges(df):
        # Replace NA values by a numerical range
        # BsmtExposure
        values = ["No", "Mn", "Av", "Gd"]
        subst = [x for x in range(1, len(values) + 1)]
        df.BsmtExposure.replace(values, subst, inplace=True)

        # LotShape
        values = ["IR3", "IR2", "IR1", "Reg"]
        subst = [x for x in range(len(values))]
        df.LotShape.replace(values, subst, inplace=True)

        # LandSlope
        values = ["Sev", "Mod", "Gtl"]
        subst = [x for x in range(len(values))]
        df.LandSlope.replace(values, subst, inplace=True)

        # BsmtFinType1 & BsmtFinType2
        values = ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
        subst = [x for x in range(1, len(values) + 1)]
        df.BsmtFinType1.replace(values, subst, inplace=True)
        df.BsmtFinType2.replace(values, subst, inplace=True)

        # CentralAir
        values = ["N", "Y"]
        subst = [0, 1]
        df.CentralAir.replace(values, subst, inplace=True)

        # Utilities
        values = ["ELO", "NoSeWa", "NoSewr", "AllPub"]
        subst = [x for x in range(len(values))]
        df.Utilities.replace(values, subst, inplace=True)

        # Functional
        values = ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]
        subst = [x for x in range(len(values))]
        df.Functional.replace(values, subst, inplace=True)

        # GarageFinish
        values = ["Unf", "RFn", "Fin"]
        subst = [x for x in range(1, len(values) + 1)]
        df.GarageFinish.replace(values, subst, inplace=True)

        # PavedDrive
        values = ["N", "P", "Y"]
        subst = [x for x in range(0, len(values))]
        df.PavedDrive.replace(values, subst, inplace=True)

        # PoolQC
        values = ["Fa", "TA", "Gd", "Ex"]
        subst = [x for x in range(1, len(values) + 1)]
        df.PoolQC.replace(values, subst, inplace=True)

        # Cas de Fence
        values = ["MnWw", "GdWo", "MnPrv", "GdPrv"]
        subst = [x for x in range(1, len(values) + 1)]
        df.Fence.replace(values, subst, inplace=True)

        # General case : from Po(or) to Ex(cellent)
        values = ["Po", "Fa", "TA", "Gd", "Ex"]
        subst = [x for x in range(1, len(values) + 1)]
        df.replace(values, subst, inplace=True)


def fill_na_simple(df):
    # Replace NA values when they have a meaning
    columns_na_meaning = {
        "BsmtQual": 0,
        "BsmtCond": 0,
        "BsmtExposure": 0,
        "BsmtFinType1": 0,
        "BsmtFinType2": 0,
        "BsmtFinSF1": 0,
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 0,
        "TotalBsmtSF": 0,
        'GarageCars': 0,
        'GarageArea': 0,
        "MasVnrType": "None",
        "MasVnrArea": 0,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "FireplaceQu": 0,
        "GarageType": 0,
        "GarageFinish": 0,
        "GarageQual": 0,
        "GarageCond": 0,
        "PoolQC": 0,
        "Fence": 0,
        "MiscFeature": 0
    }

    df.fillna(columns_na_meaning, inplace=True)


def fill_na_crossed(train_data, test_data):
    # Replace NA by the mode when they don't have a signification
    test_data['MSZoning'] = test_data['MSZoning'].fillna(
        train_data['MSZoning'].mode()[0])
    train_data['MSZoning'] = train_data['MSZoning'].fillna(
        train_data['MSZoning'].mode()[0])
    train_data['KitchenQual'] = train_data['KitchenQual'].fillna(
        train_data['KitchenQual'].mode()[0])
    test_data['KitchenQual'] = test_data['KitchenQual'].fillna(
        train_data['KitchenQual'].mode()[0])
    train_data['Alley'] = train_data['Alley'].fillna(
        train_data['Alley'].mode()[0])
    test_data['Alley'] = test_data['Alley'].fillna(
        train_data['Alley'].mode()[0])
    train_data['Utilities'] = train_data['Utilities'].fillna(
        train_data['Utilities'].mode()[0])
    test_data['Utilities'] = test_data['Utilities'].fillna(
        train_data['Utilities'].mode()[0])
    train_data['Electrical'] = train_data['Electrical'].fillna(
        train_data['Electrical'].mode()[0])
    test_data['Electrical'] = test_data['Electrical'].fillna(
        train_data['Electrical'].mode()[0])
    # TODO: could be replaced by interpolation
    train_data['LotFrontage'] = train_data['LotFrontage'].fillna(
        train_data['LotFrontage'].mode()[0])
    test_data['LotFrontage'] = test_data['LotFrontage'].fillna(
        train_data['LotFrontage'].mode()[0])
    train_data['Functional'] = train_data['Functional'].fillna(
        train_data['Functional'].mode()[0])
    test_data['Functional'] = test_data['Functional'].fillna(
        train_data['Functional'].mode()[0])
    train_data['SaleType'] = train_data['SaleType'].fillna(
        train_data['SaleType'].mode()[0])
    test_data['SaleType'] = test_data['SaleType'].fillna(
        train_data['SaleType'].mode()[0])
    train_data['Exterior1st'] = train_data['Exterior1st'].fillna(
        train_data['Exterior1st'].mode()[0])
    test_data['Exterior1st'] = test_data['Exterior1st'].fillna(
        train_data['Exterior1st'].mode()[0])
    train_data['Exterior2nd'] = train_data['Exterior2nd'].fillna(
        train_data['Exterior2nd'].mode()[0])
    test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna(
        train_data['Exterior2nd'].mode()[0])
    train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(
        train_data['GarageYrBlt'].mode()[0])
    test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(
        train_data['GarageYrBlt'].mode()[0])
