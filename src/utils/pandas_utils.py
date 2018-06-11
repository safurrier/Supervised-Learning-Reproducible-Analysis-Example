import numpy as np
import pandas

def find_null_columns(df):
    """
    Return a list of columns with null values

    Args:
    df - dataframe - Dataframe to check columns of

    Returns:
    list of null columns
    """
    return df.columns[df.isnull().any()].tolist()


def null_column_report(df, total=True, percent=True, ):
    """
    Print each null column column in a dataframe that is null as well as percent null

    Args:
    df - pandas dataframe
    total - boolean - Flag to indicate whether to print total null records per column
    percent - boolean - Flag to indicate whether to print percent of column that is null

    Returns:
    None
    """
    null_columns = find_null_columns(df)
    for col in null_columns:
        total_null_records = df[col].isnull().sum()
        print('Column:')
        print(col)
        if total:
            print('Total Nulls:')
            print(total_null_records)
        if percent:
            print('Percent Null:')
            print(round(total_null_records/df.shape[0], 2))
            print()



def column_comparison(series, col1, col2, comparison='equal', pos_return_val=1, neg_return_val=0):
    """
    Apply to a dataframe row to return a binary feature depending on equality or inequality

    E.g. df.apply(lambda s: column_match(s, 'day_of_week', 'day_of_sale'), axis=1) to for matching the two.
    Result is series of positive_return_vals and neg_return_vals. Defaults to
    """
    if comparison == 'equal':
        if series[col1] == series[col2]:
            return pos_return_val
        else:
            return neg_return_val
    if comparison == 'inequal':
        if series[col1] != series[col2]:
            return pos_return_val
        else:
            return neg_return_val

def dummies_from_bins(df, col, bins, bin_labels, col_prefix):
    """
    Given a dataframe and column to create binary features from bins, return dummy columns of said bins
    concatenated onto the end of the df
    """
    # cut the column values into bins. the labels provided are the returned values
    # bins must increase monotonically
    binned_values = pandas.cut(df[col],
                           bins=bins,
                           labels=bin_labels)

    # Create dummy variables and add prefix to col label
    dummies_cols = pandas.get_dummies(binned_values).add_prefix(col_prefix)

    # Concatenate onto end of original df
    df = pandas.concat([df, dummies_cols], axis=1)
    return df


def bin_apply(s, feature_col, min_val, max_val,binary=False):
    """
    Apply function to pandas df with axis=1 to evaluate row values and return value or binary response
    If binary=True, response values are 1 if present 0 otherwise
    Else returns the original value or a NaN
    E.g.:
    df.apply(lambda s: bin_feature_binary(s, 'hazard_rank', 0, 3), axis=1) to create a binary feature that returns
    1 if hazard group is between 0-3 and 0 if otherwise

    """
    if (s[feature_col] >= min_val) & (s[feature_col] <= max_val):
        if binary:
            return 1
        else:
            return s[feature_col]
    else:
        if binary:
            return 0
        else:
            return np.nan

def bin_df_feature(df, feature_col, min_val, max_val, binary=False):
    """
    Given a dataframe, feature column (series), bin edges, return a new series whose values are those that fit within the
    bin edges. Optionally denote if binary response (1 if present, 0 otherwise)
    """
    if binary:
        return df.apply(lambda s: bin_apply(s, feature_col, min_val, max_val, binary=True), axis=1)
    else:
        return df.apply(lambda s: bin_apply(s, feature_col, min_val, max_val, binary=False), axis=1)

def binary_feature(df, feat_col, value, binary_feature_col_name=None, concat=False):
    """
    Given a dataframe, feature column name and value to check, return a series of binary responses 1 and 0
    1 if the value in the feature column to check is present, 0 if otherwise
    binary_feature
    """
    # If binary_feature_col_name is none use this instead
    if not binary_feature_col_name:
        binary_feature_col_name = feat_col+'_is_'+str(value)

    def is_value_present(s, value):
        """
        Given a series and a value, return a binary feature 1 if present and 0 if otherwise
        """
        if s[feat_col] == value:
            return 1
        else:
            return 0
    # Return binary feature series
    binary_feature = df.apply(lambda s: is_value_present(s, value), axis=1)
    # Set series name
    binary_feature.name = binary_feature_col_name
    if concat:
        return pandas.concat([df, binary_feature], axis=1)
    return binary_feature

def scale_feature(df, feat_col, scale, value, scaled_feature_col_name=None, concat=False):
    """
    Given a dataframe, feature column name and value to check, return a scaled response
    If the value is present, multiply it by the scale multiplier. Can be used to increase or decrease
    importance of binary features
    """
    # If weighted_feature_col_name is none use this instead
    if not scaled_feature_col_name:
        scaled_feature_col_name = feat_col+'_weighted'

    def scale_value(s, value):
        """
        Given a series and a value, return a binary feature 1 if present and 0 if otherwise
        """
        if s[feat_col] == value:
            return s[feat_col] * scale
        else:
            return s[feat_col]
    # Return weighted feature series
    scaled_feature = df.apply(lambda s: scale_value(s, value), axis=1)
    # Set series name
    scaled_feature.name = weighted_feature_col_name
    if concat:
        return pandas.concat([df, scaled_feature], axis=1)
    return scaled_feature

def pct_change_from_col(df, anchor_col, diff_col):
    """ Given two columns of values, compute the percent change
        in vectorized manner

    Parameters
    ----------
    df : DataFrame
        Dataframe of data
    anchor_col : str
        The column name of from which to compute the percent differenc **from**
    diff_col : str
        The column name of from which to compute the percent differenc **to**
    Returns
    -------
    Series
        A Series of the percent change from the anchor column to the difference
        column

    Example
    -------
    df['Pct_Change_Jan_to_Feb'] = pct_change(df, 'Sales_Jan', 'Sales_Feb')

    """
    return (df[anchor_col] - df[diff_col]) / df[anchor_col]

def is_numeric_non_categoric(series, positive_val=1, negative_val=0, cardinality_limit = 1):
    """
    Check a series to see if it is numeric and has values other two designated
    positive and negative states (default is 0 and 1). Also check for a cardinality limit.
    Will return False if number of unique values is not above this (default is 1)

    Parameters
    ----------
    series : Series
        A pandas series to check
    positive_val : int
        The positive value if series were to be binary
    negative_val : int
        The negative value if the series were to be binary
    cardinality_limit : int
        A limit to the cardinality of the series. Anything below this limit will return False.
        For use when there may be integer transformed categories
            (e.g. series Fruit has categories 'Apple', 'Orange', 'Banana' mapped to 1, 2, 3)

    Returns
    -------
    Boolean
        True if column appears to be numeric, non binary with cardinality above the specified limit
    """
    # The first condition checks if all values are real numbers

    # Check if there are other values in a column besides the designated positive or negative class
    # If the size of the returned array is > 0, it's not binary
    try:
        return (np.isreal(series).all()) \
    & (np.setdiff1d(series.unique(), np.array([positive_val,negative_val])).size != 0) \
    & (len(series.unique()) > cardinality_limit)
    # Type Error occurs when float is compared to string, which would constitute non binary
    # Since this would only occur if there is not a match between the positive/negative values
    # Unique values of the column
    except TypeError:
        return False
