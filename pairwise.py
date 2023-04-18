from argparse import ArgumentParser
from random import choices

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyitlib.discrete_random_variable import information_mutual

from common import cartesian_product
from sklearn.metrics import mutual_info_score


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-out_dir', help='path to output directory')
    parser.add_argument('-card_a', default=8, help='A cardinality')
    parser.add_argument('-card_b', help='B cardinality')

    args = parser.parse_args()
    return args


#   ######################  ####### ############### #############   ####
#   PAIRWISE SSJ TESTING    ####### ############### #### ######  #######
#   ######################  ####### ############### #############   ####

def init_tids(card_a, card_b, N):
    TA = {k: [] for k in range(card_a)}
    TB = {k: [] for k in range(card_b)}
    for i in range(N):
        a = np.random.choice(card_a)
        b = np.random.choice(card_b)

        TA[a].append(i)
        TB[b].append(i)

    TA = clear_tid(TA)
    TB = clear_tid(TB)
    return TA, TB


def clear_tid(tid):
    """
    scan all letters. remove empty letters (no rows in data)
    """
    res = []
    for k in tid.keys():
        if not tid[k]:
            res.append(k)

    for i in res:
        del tid[i]

    return tid


def join(TA, TB):
    res = {}
    for ka, va in TA.items():
        for kb, vb in TB.items():
            intersection = sorted(list(set(va).intersection(set(vb))))
            if intersection:
                key = (ka, kb)
                res[key] = intersection

    return res


def ground_truth_entropy(TAB):
    """precise calculation"""
    lens = [len(x) for x in TAB.values()]
    N = sum(lens)

    dist = [x / N for x in lens]
    res = 0
    for x in dist:
        res -= x * np.log2(x)

    return res


def get_distributions(TA, TB):
    lens1 = [len(x) for x in TA.values()]
    lens2 = [len(x) for x in TB.values()]
    N = sum(lens1)
    N2 = sum(lens2)
    assert (N == N2)

    dist1 = {k: v for k, v in zip(TA.keys(), lens1)}
    dist2 = {k: v for k, v in zip(TB.keys(), lens2)}
    return dist1, dist2


def sample(distributions):
    a = choices(list(distributions[0].keys()), weights=list(distributions[0].values()))[0]
    b = choices(list(distributions[1].keys()), weights=list(distributions[1].values()))[0]
    return a, b


def update_distributions(distributions, a, b, L):
    distributions[0][a] -= L
    if not distributions[0][a]:
        del distributions[0][a]

    distributions[1][b] -= L
    if not distributions[1][b]:
        del distributions[1][b]

    return distributions


def SSJ(TA, TB, coverage):
    """join TA TB by adaptive sampling"""
    distributions = get_distributions(TA, TB)

    N = sum([len(x) for x in TA.values()])
    N2 = sum([len(x) for x in TB.values()])
    assert N == N2

    effective_N = 0

    TAB = {}  # output
    visited = []
    sampled_empty = []  # AxB \ DAB
    nsamples = 0
    target_N = N * coverage

    while effective_N < target_N:
        nsamples += + 1

        # generate sample
        a, b = sample(distributions)

        if (a, b) in visited:
            continue
        visited.append((a, b))

        # intersect Ia Ib
        Ia = TA[a]
        Ib = TB[b]
        intersection = sorted(list(set(Ia).intersection(set(Ib))))
        L = len(intersection)

        if not L:
            sampled_empty.append((a, b))
        else:
            TAB[(a, b)] = intersection

        # adaptive sampling step
        distributions = update_distributions(distributions, a, b, L)
        if not distributions:
            break

        effective_N += L

    return TAB, nsamples


def pairwise_experiment():
    args = parse_args()
    out_dir = args.out_dir
    card_a = int(args.card_a)
    card_b = int(args.card_b)
    N = int(args.N)

    # generate product set
    product_set = cartesian_product(np.arange(card_a), np.arange(card_b))

    # init TIDs
    TA, TB = init_tids(card_a, card_b, N)

    # join TIDs
    TAB = join(TA, TB)

    print(f'-I- Product set of size {len(product_set)}; Active domain of size {len(TAB)}')

    HAB = ground_truth_entropy(TAB)

    TAB_s, nsamples = SSJ(TA, TB, coverage=0.8)
    HAB_s = ground_truth_entropy(TAB_s)


#   ######################  ####### ############### #############   ####
#   MARCH 2023  ### ######  ####### ############### #############   ####
#   ######################  ####### ############### #############   ####

def build_df_max_rho_max_I(alpha, beta):
    """remove maximum entries without reducing cardinalities"""
    product_set = cartesian_product(np.arange(alpha), np.arange(beta))
    df_product = pd.DataFrame(product_set, columns=['X', 'Y'])
    df_product.sort_values(by='Y')
    data_clusters = df_product.groupby('Y')
    df_tuples = []
    for d in data_clusters:
        cluster = d[1]
        y = cluster['Y'].iloc[0]
        j = y % alpha
        x = cluster['X'].iloc[j]
        pair = (x, y)
        df_tuples.append(pair)

    # create data
    df = pd.DataFrame(df_tuples, columns=['X', 'Y'])
    return df


def build_df(alpha, beta, rho):
    """remove $rho$ number of entries from product set"""
    product_set = cartesian_product(np.arange(alpha), np.arange(beta))
    df_product = pd.DataFrame(product_set, columns=['X', 'Y'])
    df_product.sort_values(by='Y')
    drop_indices = np.random.choice(df_product.index, rho, replace=False)
    df_subset = df_product.drop(drop_indices)
    df_subset.reset_index()
    return df_subset


def I_exact_max_rho_min_I(alpha, beta):
    """I(X;Y) for maximum entry removal minimum I regime"""
    A = (alpha - 1) / beta * np.log(beta)
    B = (1 + beta - alpha) / beta * np.log(beta / (1 + beta - alpha))
    return A + B


def I_exact_max_rho_max_I(alpha, beta: object):
    """I(X;Y) for maximum entry removal maximum I regime"""
    q = beta % alpha
    A = np.log(beta / (int(beta / alpha) + 1))
    B = np.log(beta / int(beta / alpha))
    return q / alpha * A + (alpha - q) / alpha * B


def I_upper_bound_max_I(alpha, beta):
    """U > I(X;Y) for maximum entry removal maximum I regime"""
    q = beta % alpha
    A = np.log(alpha)
    B = np.log(beta / (beta / alpha - 1))
    return q / alpha * A + (alpha - q) / alpha * B


def I_lower_bound_max_I(alpha, beta):
    """L < I(X;Y) for maximum entry removal maximum I regime"""
    q = beta % alpha
    A = np.log(beta / (beta / alpha + 1))
    B = np.log(alpha)
    return q / alpha * A + (alpha - q) / alpha * B


def I_exact(df):
    """explicitly evaluate I(X;Y)"""
    Px = df['X'].value_counts(normalize=True)
    Py = df['Y'].value_counts(normalize=True)
    Pxy = df.value_counts(normalize=True)

    res = 0
    for pair in Pxy.index:
        x = pair[0]
        y = pair[1]

        joint = Pxy[pair]
        marginal_x = Px[x]
        marginal_y = Py[y]

        res += joint * np.log(joint / (marginal_x * marginal_y))

    return res


def I_single_entry(alpha, beta):
    """I(X;Y) where single entry removed from product set"""
    A = (beta - 1) / (alpha * beta - 1) * np.log((alpha * beta - 1) / (alpha * (beta - 1)))
    B = (alpha - 1) / (alpha * beta - 1) * np.log((alpha * beta - 1) / (beta * (alpha - 1)))
    C = (alpha * beta - alpha - beta + 1) / (alpha * beta - 1) * np.log((alpha * beta - 1) / (alpha * beta))

    return A + B + C


def plot_all(**kwargs):
    legend = []
    plt.figure()
    for k, v in kwargs.items():
        legend.append(k)
        plt.plot(v)

    plt.legend(legend)
    plt.show()


def pairwise_MI_experiment():
    """for maximum entry removal regime, evaluate I(X;Y). fixed beta, varying alpha"""
    args = parse_args()
    out_dir = args.out_dir
    beta = int(args.card_b)

    Us = []  # U >= I1s
    I1s = []  # I(X;Y) for max rho max I regime
    Ls = []  # L <= I1s
    I2s = []  # I(X;Y) for max rho min I regime
    I3s = []  # I(X;Y) for single entry removal (min rho)

    for alpha in range(2, beta):

        Is = []  # I(X;Y)
        max_rho = alpha * beta - beta

        for rho in range(1, max_rho):
            df = build_df(alpha, beta, rho)
            Is.append(I_exact(df))

        Us = [I_upper_bound_max_I(alpha, beta)] * len(Is)
        I1s = [I_exact_max_rho_max_I(alpha, beta)] * len(Is)
        Ls = [I_lower_bound_max_I(alpha, beta)] * len(Is)
        I2s = [I_exact_max_rho_min_I(alpha, beta)] * len(Is)
        I3s = [I_single_entry(alpha, beta)] * len(Is)

        print('hello')

    print('finish')


def I_general_regime():
    args = parse_args()
    alpha = int(args.card_a)
    beta = int(args.card_b)
    N = alpha * beta

    Is = []
    for rho in range(1, N):
        df = pd.DataFrame(cartesian_product(np.arange(alpha), np.arange(beta)), columns=['X', 'Y'])
        drop_indices = np.random.choice(df.index, rho, replace=False)
        df = df.drop(drop_indices)
        df.reset_index()

        Is.append(I_exact(df))

    plt.plot(list(range(1, N)), Is)


if __name__ == '__main__':
    pairwise_MI_experiment()
