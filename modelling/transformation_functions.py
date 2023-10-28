import pandas as pd


def sort_ascending(dataframe, column_name):
    sorted_df = dataframe.sort_values(by=column_name, ascending=True)
    return sorted_df.index.tolist()


def sort_descending(dataframe, column_name):
    sorted_df = dataframe.sort_values(by=column_name, ascending=False)
    return sorted_df.index.tolist()


def filter_by_threshold(dataframe, column_name, threshold):
    sorted_df = dataframe[dataframe[column_name] > threshold]
    return sorted_df.index.tolist()


def combined_transform(dataframe, column_name, threshold):
    df = dataframe[dataframe[column_name] > threshold]
    sorted_df = df.sort_values(by=column_name, ascending=True)
    return sorted_df.index.tolist()
