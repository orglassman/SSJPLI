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


def plot_5_datasets_coverage():
    coverages = [.25, .5, .75, .9, .99]

    adult_dir = ''
    connect4_dir = ''
    covertype_dir = ''
    letter_dir = ''
    mushroom_dir = ''

    adult_dfs = load_dfs(adult_dir, coverages)
    connect4_dfs = load_dfs(connect4_dir, coverages)
    covertype_dfs = load_dfs(covertype_dir, coverages)
    letter_dfs = load_dfs(letter_dir, coverages)
    mushroom_dfs = load_dfs(mushroom_dir, coverages)

    fig, axs = plt.subplots(1, 3, sharex=True, figsize=[20, 6])
    for ax in axs.flatten():
        ax.grid(axis='y')
        ax.tick_params(labelsize=20)

    axs[0].plot(coverages, [np.average(adult_dfs[c][4]['t_ssj_ratio']) for c in coverages], linewidth=2, marker='o',
                label='Adult')
    axs[0].plot(coverages, [np.average(connect4_dfs[c][4]['t_ssj_ratio']) for c in coverages], linewidth=2, marker='^',
                label='Connect-4')
    axs[0].plot(coverages, [np.average(covertype_dfs[c][4]['t_ssj_ratio']) for c in coverages], linewidth=2, marker='v',
                label='Covertype')
    axs[0].plot(coverages, [np.average(letter_dfs[c][4]['t_ssj_ratio']) for c in coverages], linewidth=2, marker='*',
                label='Letter')
    axs[0].plot(coverages, [np.average(mushroom_dfs[c][4]['t_ssj_ratio']) for c in coverages], linewidth=2, marker='p',
                label='Mushroom')

    axs[1].plot(coverages, [np.average(adult_dfs[c][4]['h_ratio']) for c in coverages], linewidth=2, marker='o',
                label='Adult')
    axs[1].plot(coverages, [np.average(connect4_dfs[c][4]['h_ratio']) for c in coverages], linewidth=2, marker='^',
                label='Connect-4')
    axs[1].plot(coverages, [np.average(covertype_dfs[c][4]['h_ratio']) for c in coverages], linewidth=2, marker='v',
                label='Covertype')
    axs[1].plot(coverages, [np.average(letter_dfs[c][4]['h_ratio']) for c in coverages], linewidth=2, marker='*',
                label='Letter')
    axs[1].plot(coverages, [np.average(mushroom_dfs[c][4]['h_ratio']) for c in coverages], linewidth=2, marker='p',
                label='Mushroom')

    axs[2].plot(coverages, [np.average((adult_dfs[c][4]['H_true'] - adult_dfs[c][4]['H_ssj']) ** 2) for c in coverages],
                linewidth=2, marker='o', label='Adult')
    axs[2].plot(coverages,
                [np.average((connect4_dfs[c][4]['H_true'] - connect4_dfs[c][4]['H_ssj']) ** 2) for c in coverages],
                linewidth=2, marker='^', label='Connect-4')
    axs[2].plot(coverages,
                [np.average((covertype_dfs[c][4]['H_true'] - covertype_dfs[c][4]['H_ssj']) ** 2) for c in coverages],
                linewidth=2, marker='v', label='Covertype')
    axs[2].plot(coverages,
                [np.average((letter_dfs[c][4]['H_true'] - letter_dfs[c][4]['H_ssj']) ** 2) for c in coverages],
                linewidth=2, marker='*', label='Letter')
    axs[2].plot(coverages,
                [np.average((mushroom_dfs[c][4]['H_true'] - mushroom_dfs[c][4]['H_ssj']) ** 2) for c in coverages],
                linewidth=2, marker='p', label='Mushroom')

    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), fancybox=True, shadow=True, ncol=5, fontsize=15)
    axs[1].set_xlabel('coverage', fontsize=20)
    axs[0].set_title(r'$t\;/\;t_{PLI}$', fontsize=20)
    axs[1].set_title(r'$H_{s}\;/\;H$', fontsize=20)
    axs[2].set_title('MSE', fontsize=20)
    plt.tight_layout()


def plot_5_datasets_varying_q():
    query_sizes = [2, 4, 6, 10, 15]
    coverages = [.25, .5, .75, .9, .99]

    connect4_dir = ''
    covertype_dir = ''
    letter_dir = ''
    mushroom_dir = ''

    connect4_dfs = load_dfs(connect4_dir, coverages)
    covertype_dfs = load_dfs(covertype_dir, coverages)
    letter_dfs = load_dfs(letter_dir, coverages)
    mushroom_dfs = load_dfs(mushroom_dir, coverages)

    fig, axs = plt.subplots(1, 3, sharex=True, figsize=[20, 6])
    for ax in axs.flatten():
        ax.grid(axis='y')
        ax.tick_params(labelsize=20)

    axs[0].plot(query_sizes, [np.average(connect4_dfs[.9][q]['t_ssj_ratio']) for q in query_sizes], linewidth=2,
                marker='^',
                label='Connect-4')
    axs[0].plot(query_sizes, [np.average(covertype_dfs[.9][q]['t_ssj_ratio']) for q in query_sizes], linewidth=2,
                marker='v',
                label='Covertype')
    axs[0].plot(query_sizes, [np.average(letter_dfs[.9][q]['t_ssj_ratio']) for q in query_sizes], linewidth=2,
                marker='*',
                label='Letter')
    axs[0].plot(query_sizes, [np.average(mushroom_dfs[.9][q]['t_ssj_ratio']) for q in query_sizes], linewidth=2,
                marker='p',
                label='Mushroom')

    axs[1].plot(query_sizes, [np.average(connect4_dfs[.9][q]['h_ratio']) for q in query_sizes], linewidth=2, marker='^',
                label='Connect-4')
    axs[1].plot(query_sizes, [np.average(covertype_dfs[.9][q]['h_ratio']) for q in query_sizes], linewidth=2,
                marker='v',
                label='Covertype')
    axs[1].plot(query_sizes, [np.average(letter_dfs[.9][q]['h_ratio']) for q in query_sizes], linewidth=2, marker='*',
                label='Letter')
    axs[1].plot(query_sizes, [np.average(mushroom_dfs[.9][q]['h_ratio']) for q in query_sizes], linewidth=2, marker='p',
                label='Mushroom')

    axs[2].plot(query_sizes,
                [np.average((connect4_dfs[.9][q]['H_true'] - connect4_dfs[.9][q]['H_ssj']) ** 2) for q in query_sizes],
                linewidth=2, marker='^', label='Connect-4')
    axs[2].plot(query_sizes,
                [np.average((covertype_dfs[.9][q]['H_true'] - covertype_dfs[.9][q]['H_ssj']) ** 2) for q in
                 query_sizes],
                linewidth=2, marker='v', label='Covertype')
    axs[2].plot(query_sizes,
                [np.average((letter_dfs[.9][q]['H_true'] - letter_dfs[.9][q]['H_ssj']) ** 2) for q in query_sizes],
                linewidth=2, marker='*', label='Letter')
    axs[2].plot(query_sizes,
                [np.average((mushroom_dfs[.9][q]['H_true'] - mushroom_dfs[.9][q]['H_ssj']) ** 2) for q in query_sizes],
                linewidth=2, marker='p', label='Mushroom')

    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), fancybox=True, shadow=True, ncol=5, fontsize=15)
    axs[1].set_xlabel('query size', fontsize=20)
    axs[0].set_title(r'$t\;/\;t_{PLI}$', fontsize=20)
    axs[1].set_title(r'$H_{s}\;/\;H$', fontsize=20)
    axs[2].set_title('MSE', fontsize=20)
    plt.tight_layout()


def plot_adult_q2():
    adult_q2_dir = ''
    adult_q2_dfs = load_dfs(adult_q2_dir)
    coverages = [.25, .5, .75, .9, .99]

    fig, axs = plt.subplots(1, 3, figsize=[20, 6])
    for ax in axs.flatten():
        ax.grid(axis='y')
        ax.tick_params(labelsize=20)
    axs[0].plot(coverages, [np.average(adult_q2_dfs[c][2]['t_ssj_ratio']) for c in coverages], linewidth=2, marker='o',
                label='SSJ')
    axs[0].plot(coverages, coverages, linewidth=2, color='r', label=r'$y=x$')
    axs[1].plot([np.average(adult_q2_dfs[c][2]['t_ssj_ratio']) for c in coverages],
                [np.average(adult_q2_dfs[c][2]['h_ratio']) for c in coverages], linewidth=2, marker='o')
    axs[2].plot([np.average(adult_q2_dfs[c][2]['h_ratio']) for c in coverages],
                [np.average(adult_q2_dfs[c][2]['MSE']) for c in coverages], linewidth=2, marker='o')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), fancybox=True, shadow=True, ncol=5, fontsize=15)
    axs[0].set_xlabel('coverage', fontsize=20)
    axs[1].set_xlabel(r'$t\;/\;t_{PLI}$', fontsize=20)
    axs[2].set_xlabel(r'$H_{s}\;/\;H$', fontsize=20)
    axs[0].set_title(r'$t\;/\;t_{PLI}$', fontsize=20)
    axs[1].set_title(r'$H_{s}\;/\;H$', fontsize=20)
    axs[2].set_title('MSE', fontsize=20)
    plt.tight_layout()


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
