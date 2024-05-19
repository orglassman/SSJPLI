import os
import pickle
import re

from scipy.stats import entropy
import numpy as np
import pandas as pd

COVERAGES = [.25, .5, .75, .9, .99]


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
        (div, mod) = divmod(div - 1, 26)  # will return (x, 0 .. 25)
        excelCol = chr(mod + 65) + excelCol

    return excelCol


def freedman_diaconis_binning(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    n = len(x)

    width = 2 * (q3 - q1) / np.cbrt(n)
    if width:
        num_bins = int(np.ceil((np.max(x) - np.min(x)) / width))
    else:
        num_bins = 1
    binned = pd.cut(x, bins=num_bins, labels=False)
    return binned


def preprocess(df):
    """
    1. remove na
    2. bin floats
    3. encode labels

    CONVERT TO STRINGS NOT INTEGERS
    """
    df = df.dropna()

    for column in df.columns:
        # get unique values
        if df.dtypes[column] == 'float64':
            x = freedman_diaconis_binning(df[column])
            df[column] = x

        u = df[column].unique()
        # gen mapping series
        m = pd.Series([str(x) for x in range(len(u))], u)
        # encode
        mapped = df[column].map(m)

        df = df.assign(**{column: mapped})
    return df


def swap(l, pos1, pos2):
    x = l[pos2]
    l[pos2] = l[pos1]
    l[pos1] = x
    return l


def intersection(list1, list2):
    return sorted(list(set(list1) & set(list2)))


def union(list1, list2):
    return sorted(list(set(list1) | set(list2)))


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cartesian_product_multi(*dfs):
    """
    source:
    https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
    """
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    return pd.DataFrame(
        np.column_stack([df.values[idx[:, i]] for i, df in enumerate(dfs)]))


def binary_search(arr, x):
    low = 0
    high = len(arr) - 1

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


def min_set_cover(x, subsets):
    """
    x - list of elements to be covered by subsets
    subsets - list of subsets
    """
    covered_elements = {element: False for element in x}

    res = []
    for element in x:

        # skip already covered elements
        if covered_elements[element]:
            continue

        # search backwards (larger subsets are at end of list)
        for s in reversed(subsets):

            # element in s - potential candidate
            if element in s:

                # if s contains elements not in x, then s is dismissed
                skip = False
                for other_element in s:
                    # skip if there's foreign element or an already covered element
                    if (other_element not in x) or covered_elements[other_element]:
                        skip = True
                        break
                if skip:
                    continue
                # s does not contain elements not in x - continue
                covered_elements[element] = True
                res.append(','.join(s))

                # check if subset covers other elements
                for other_element in x:
                    if other_element in s:
                        covered_elements[other_element] = True

                break

    return res


def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


def deflatten(t, levels):
    if not sum(levels) == len(t):
        raise ValueError

    reshaped = []
    prev = 0
    for level in levels:
        if level == 1:
            reshaped.append(t[prev])
        else:
            reshaped.append(t[prev: prev + level])
        prev = prev + level

    return tuple(reshaped)


def randomize_query(omega, size):
    return np.random.choice(omega, size=size, replace=False)


def dump_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
        return b


def load_dir(path):
    dfs = {c: {} for c in COVERAGES}
    pattern = "coverage_(\d+\.\d+)_query_size_(\d+)"
    for file in os.listdir(path):
        fullpath = path + '\\' + file
        if os.path.isdir(fullpath):
            continue
        match = re.search(pattern, file)
        if match:
            coverage = float(match.group(1))
            query_size = int(match.group(2))
            dfs[coverage][query_size] = load_pickle(fullpath)
        else:
            print(f"-W- File {fullpath}")

    return dfs


def entropy_csv(path, q):
    """compute H(q) for input CSV"""
    p = sorted(q)
    df = preprocess(pd.read_csv(path, usecols=p))
    dist = df.value_counts(normalize=True).values
    H = entropy(dist, base=2)
    return H


def binary_entropy(q):
    if q != 0 and q != 1:
        q_bar = 1 - q
        return -q * np.log2(q) - q_bar * np.log2(q_bar)

    return 0


def DKL(P, Q):
    res = 0
    for x in P.keys():
        if not Q[x]:
            raise "support of Q smaller than P"

        p = P[x]
        q = Q[x]

        res += p * np.log2(p / q)

    return res


def sort_by_key(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[0])}


def sort_by_value(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
