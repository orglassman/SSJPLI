import os
import re
from argparse import ArgumentParser

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from common import load_pickle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-in_dir', help='path to directory containing .pkl files')
    args = parser.parse_args()
    return args


def plot_distribution_over_time(letters, distributions):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for ax in axs.flatten():
        ax.grid()
    axs[0, 0].bar(letters, distributions[0])
    axs[0, 1].bar(letters, distributions[1])
    axs[1, 0].bar(letters, distributions[2])
    axs[1, 1].bar(letters, distributions[3])

    axs[1, 1].tick_params(axis='x', rotation=90, labelsize=24)
    axs[1, 0].tick_params(axis='x', rotation=90, labelsize=24)

    axs[0, 0].set_title(r'$\P^{(0)}$', fontsize=35)
    axs[0, 1].set_title(r'$\P^{(1)}$', fontsize=35)
    axs[1, 0].set_title(r'$\P^{(2)}$', fontsize=35)
    axs[1, 1].set_title(r'$\P^{(3)}$', fontsize=35)


def get_dist_from_ssj_distributions_dict(ssj_distributions_dict):
    tot = sum(ssj_distributions_dict.values())
    dist = [x / tot for x in ssj_distributions_dict.values()]
    return dist


def config_mpl():
    plt.rcParams.update({'figure.autolayout': True})
    mpl.style.use('seaborn-paper')
    # mpl.rcParams['']

    # ax.plot(query_sizes, y, linewidth=4, marker='o')


def get_fig_ax():
    fig, ax = plt.subplots()
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.grid(axis='y')
    return fig, ax


def get_run_params(files):
    pattern = r'alpha_(\d+).+beta_(\d+).+bin_num_(\d+).+coverage_(\d+\.\d+)'
    alphas = {}
    betas = {}
    bins = {}
    coverages = {}
    for file in files:
        res = re.search(pattern, file)
        if not res:
            print(f'-W- Check {file}')
            continue
        alpha = int(res.groups(0)[0])
        beta = int(res.groups(0)[1])
        bin_num = int(res.groups(0)[2])
        coverage = float(res.groups(0)[3])
        alphas[alpha] = 1
        betas[beta] = 1
        bins[bin_num] = 1
        coverages[coverage] = 1

    res = [
        list(alphas.keys()),
        list(betas.keys()),
        list(bins.keys()),
        list(coverages.keys())
    ]

    return res


def load_dfs(dir, coverages):
    files = os.listdir(dir)

    pattern = r'coverage_(\d+\.\d+).+query_size_(\d+)'
    dfs = {c: {} for c in coverages}
    for file in files:
        res = re.search(pattern, file)
        if not res:
            print(f'-W- Check {file}')
            continue

        coverage = float(res.groups(0)[0])
        query_size = int(res.groups(0)[1])

        dfs[coverage][query_size] = load_pickle(dir + os.sep + file)

    return dfs


if __name__ == '__main__':
    print('hello')
