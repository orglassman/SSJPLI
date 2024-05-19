import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_file', help='path to input CSV')
    parser.add_argument('-q_sizes', help='integers separated by commas')
    args = parser.parse_args()
    return args


def plot_data(q_sizes, data):
    large = 22
    params = {'axes.titlesize': large,
              'legend.fontsize': large,
              'figure.figsize': (16, 10),
              'axes.labelsize': large,
              'axes.titlesize': large,
              'xtick.labelsize': large,
              'ytick.labelsize': large,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.tick_params(size=20)
    plt.style.use('seaborn-paper')


    names = list(data.keys())
    fig, ax = plt.subplots(1, 3)

    for i, dataset_name in enumerate(names):
        actives = data[dataset_name][0]
        pss = data[dataset_name][1]
        actuals = data[dataset_name][2]

        avg_actives = {k: np.average(v) for k, v in actives.items()}
        avg_pss = {k: np.average(v) for k, v in pss.items()}
        avg_actuals = {k: np.average(v) for k, v in actuals.items()}

        ax[i].set_yscale('log')
        ax[i].plot(q_sizes, avg_actives.values(), linewidth=3, markersize=7, marker='o', label='active')
        ax[i].plot(q_sizes, avg_pss.values(), linewidth=3, markersize=7, marker='^', label='product')
        ax[i].plot(q_sizes, avg_actuals.values(), linewidth=3, markersize=7, marker='D', label='actual')
        ax[i].grid(axis='y')
    pass


def run_product_sets(in_file, q_sizes):
    df = pd.read_csv(in_file)

    pss = {}  # worst
    actives = {}  # active domain of current X
    actuals = {}  # what NLJ actually encounters pairwise
    for l in q_sizes:
        tmp_pss = []
        tmp_actives = []
        tmp_actuals = []
        for i in range(100):
            q = np.random.choice(df.columns, l, replace=False)
            print(f'-I- Query {q}')
            active = len(df[q].value_counts())
            print(f'-I- Active domain size {active}')
            tmp_actives.append(active)

            # get product set
            ps_size = 1
            for x in q:
                ps_size *= len(df[x].value_counts())
            print(f'-I- Product set {ps_size}')
            tmp_pss.append(ps_size)

            # get actual NLJ product set
            first = q[0]
            up_til_now = [first]
            total_operations = 0
            for x in q[1:]:
                actual_product = len(df[up_til_now].value_counts()) * len(df[x].value_counts())
                total_operations += actual_product
                up_til_now.append(x)
            print(f'-I- Total operations for pairwise NLJ {total_operations}')
            tmp_actuals.append(total_operations)

        actives[l] = tmp_actives
        pss[l] = tmp_pss
        actuals[l] = tmp_actuals
    return [actives, pss, actuals]


if __name__ == '__main__':
    files = {
        'covertype': "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Covertype\\covtype_discrete.csv",
        'connect4': "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Connect4\\connect4.csv",
        'mushroom': "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Mushroom\\mushroom.csv"
    }
    q_sizes = [2, 4, 6, 10, 15]
    to_plot = {}
    for name, file in files.items():
        print(f'-I- Running dataset {name}')
        to_plot[name] = run_product_sets(file, q_sizes)
    plot_data(q_sizes, to_plot)