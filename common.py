"""
commonly used functions and variables
"""
import numpy as np
import pandas as pd


def rename_columns(df):
    """
    set '0', '1', '2', ...
    as dataframe header
    """
    # create column name array
    num_atts = len(df.columns)
    new_columns = []
    for i in range(0, num_atts):
        # new_columns.append(str(i))
        new_columns.append(colToExcel(i))

    # first row will be removed upon setting column names, so we save it and later append it
    first_row = list(df.columns)

    # rename
    df = df.set_axis(new_columns, axis=1, inplace=False)

    return df

def colToExcel(col):
    """
    convert df columns to excel-like values:
    1 -> A
    2 -> B
    27 -> AA
    """
    excelCol = str()
    div = col + 1
    while div:
        (div, mod) = divmod(div-1, 26) # will return (x, 0 .. 25)
        excelCol = chr(mod + 65) + excelCol

    return excelCol

def convert_to_numeric(df):
    """
    convert all entries to numeric
    essentially - we're interested in distribution of data not data itself
    """
    for column in df.columns:
        # get unique values
        u = df[column].unique()
        # gen a mapping series
        m = pd.Series(range(len(u)), u)
        # encode
        df[column] = df[column].map(m)
    return df

def swap(list, pos1, pos2):
    x = list[pos2]
    list[pos2] = list[pos1]
    list[pos1] = x
    return list

def intersection(list1, list2):
    return sorted(list(set(list1) & set(list2)))

def union(list1, list2):
    return sorted(list(set(list1) | set(list2)))

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def cartesian_product_multi(*dfs):
    """
    source:
    https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
    """
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    return pd.DataFrame(
        np.column_stack([df.values[idx[:,i]] for i,df in enumerate(dfs)]))

def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0

    # check if exceeds list itself
    if x > arr[high] or x < arr[low]:
        return -1

    while low <= high:
        mid = (high + low) // 2

        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid

    # reached here, return closeset element > x
    return low

def mergesort(it_dict):
    """perform merge sort on iterators (based on key)"""

    sorted_pairs = sorted(it_dict.items(), key=lambda x: x[1])
    res = {}

    # O(N) - need to fix this later on
    for pair in sorted_pairs:
        key = pair[0]
        value = pair[1]
        res[key] = value

    return res