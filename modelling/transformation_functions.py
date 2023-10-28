def sort_ascending(dataframe, column_name):
    return dataframe.sort_values(by=column_name, ascending=True)

def sort_descending(dataframe, column_name):
    return dataframe.sort_values(by=column_name, ascending=False)

def filter_by_threshold(dataframe, column_name, threshold):
    return dataframe[dataframe[column_name] > threshold]


def combined_transform(dataframe, column_name, threshold):
    df = dataframe[dataframe[column_name] > threshold]
    return df.sort_values(by=column_name, ascending=True)
