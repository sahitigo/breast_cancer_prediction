import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def apply_eda(df, drop_null='yes', fill_null='mean', columns_to_drop=None):
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    df_concatenated = None
    print("DataFrame shape:")
    print(df.shape, '\n', '\n')

    print("Checking for null values:")
    null_values_count = df.isna().sum().sum()
    print("Total Null values count:", null_values_count)
    df_num = None
    df_cat = None
    df = drop_null_columns(df)

    if null_values_count != 0:
        user_input = drop_null

        if user_input == 'yes':
            df = df.dropna()
            df = remove_non_numeric_values(df)
            df = check_and_update_numeric_cols(df)
            df_num = df.select_dtypes(include='number')
            df_cat = df.select_dtypes(include='object')
            print(df.corr())

        else:
            fill_null_values = fill_null
            df = remove_non_numeric_values(df)
            df = check_and_update_numeric_cols(df)
            df_num = df.select_dtypes(include='number')
            df_cat = df.select_dtypes(include='object')
            print(df.corr())

            for col in df_num.columns:
                if fill_null_values == 'mean':
                    df_num[col].fillna(df_num[col].mean(), inplace=True)

                elif fill_null_values == 'median':
                    df_num[col].fillna(df_num[col].median(), inplace=True)

                elif fill_null_values == 'ffill':
                    df_num[col].fillna(method='ffill', inplace=True)

                elif fill_null_values == 'bfill':
                    df_num[col].fillna(method='bfill', inplace=True)

            for col in df_cat.columns:
                df_cat[col].fillna(df_cat[col].mode().iloc[0], inplace=True)
    else:
        df = check_and_update_numeric_cols(df)
        df_num = df.select_dtypes(include='number')
        df_cat = df.select_dtypes(include='object')

    print(df.corr())

    vif_data = calculate_vif(df_num)
    print("Features and their VIF values:")
    print(vif_data)

    while vif_data["VIF"].max() > 10:
        max_vif_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        df_num.drop(columns=max_vif_feature, inplace=True)
        vif_data = calculate_vif(df_num)

    print("Features after removing features with VIF >= 10:")
    print(df_num.columns)

    vif_remaining = calculate_vif(df_num)
    print("VIF of remaining features:")
    print(vif_remaining)

    plot_categorical(df_cat)
    plot_numerical(df_num)
    plot_boxplots(df_num)
    # plot_correlation(correlation_matrix)
    plot_vif(vif_data)

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df_cat).toarray()
    df_encoded = pd.DataFrame(encoded_data, columns=[f"{col}_{category}" for col, categories in zip(df_cat.columns, encoder.categories_) for category in categories])

    if len(df_encoded.columns) == 0:
        df_concatenated = df_num
    else:
        df_concatenated = pd.concat([df_num, df_encoded], axis=1)

    return df_concatenated

def plot_categorical(df_cat, palette='Set1'):
    for col in df_cat.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_cat, x=col, palette=palette)
        plt.title(f"Count Plot - {col}")
        plt.xticks(rotation=90)
        plt.show()

def plot_numerical(df_num):
    for col in df_num.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_num[col], kde=True, element="step", palette='Set1')
        plt.title(f"Distribution Plot - {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.show()

def plot_boxplots(df_num):
    for col in df_num.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_num, y=col, palette='Set1')
        plt.title(f"Box Plot - {col}")
        plt.ylabel(col)
        plt.show()

def plot_vif(vif_data):
    vif_data_sorted = vif_data.sort_values(by='VIF', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='VIF', y='Feature', data=vif_data_sorted, palette='Set1')
    plt.title("Variance Inflation Factor (VIF)")
    plt.xlabel("VIF")
    plt.ylabel("Feature")
    plt.show()

def check_and_update_numeric_cols(df):
    for column in df.select_dtypes(include='object'):
        converted = pd.to_numeric(df[column], errors='ignore')
        if np.issubdtype(converted.dtype, np.number):
            df.loc[:, column] = converted
    return df

def remove_non_numeric_values(df):
    drop_values = ["#VALUE!", "NA", "NaN", "na", "-", "nan"]
    drop_condition = ~df.isin(drop_values).any(axis=1)
    df = df[drop_condition]
    return df

def drop_null_columns(df):
    row_size = df.shape[0]
    null_threshold_percent = 50
    has_null_flag = 0

    for col in df.columns:
        null_values_count_col = df[col].isna().sum()
        null_value_percent = (null_values_count_col / row_size) * 100

        if null_value_percent >= null_threshold_percent:
            df.drop(columns=[col], axis=1, inplace=True)
            has_null_flag = 1
    if has_null_flag == 1:
        print("Dataframe shape after dropping null columns:", df.shape)
    return df