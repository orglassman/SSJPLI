import os
import re
from argparse import ArgumentParser

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from common import load_pickle, binary_entropy


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


def get_U_binary_entropy(df):
    A = df['rho'] * df['H_ssj']
    B = (1 - df['rho']) * np.log2(df['product_set'])
    C = df['rho'].apply(binary_entropy)
    return np.average(A + B + C)


def load_dfs(dir, coverages=None, mode='ssj'):
    if coverages is None:
        coverages = [.25, .5, .75, .9, .99]
    files = os.listdir(dir)

    pattern = None
    dfs = {c: {} for c in coverages}
    if mode == 'ssj':
        pattern = r'coverage_(\d+\.\d+).+query_size_(\d+)'
    elif mode == 'isj':
        pattern = r'precision_(\d+\.\d+).+query_size_(\d+)'

    for file in files:
        res = re.search(pattern, file)
        if not res:
            print(f'-W- Check {file}')
            continue

        coverage = float(res.groups(0)[0])
        query_size = int(res.groups(0)[1])
        dfs[coverage][query_size] = load_pickle(dir + os.sep + file)

    return dfs


def make_net_vs_q_plot():
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['axes.labelsize'] = 15
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    # plt.figure(figsize=(12, 5), dpi=300, facecolor='w')

    from common import load_dir
    adult_dir = 'C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\real data analysis\\Adult'
    connect4_dir = 'C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\real data analysis\\Connect4'
    covertype_dir = 'C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\real data analysis\\Covertype'
    letter_dir = 'C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\real data analysis\\Letter'
    mushroom_dir = 'C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\real data analysis\\Mushroom'

    adult_dfs = load_dir(adult_dir)
    connect4_dfs = load_dir(connect4_dir)
    covertype_dfs = load_dir(covertype_dir)
    letter_dfs = load_dir(letter_dir)
    mushroom_dfs = load_dir(mushroom_dir)

    adult_qs = sorted(list(adult_dfs[.99].keys()))
    connect4_qs = sorted(list(connect4_dfs[.99].keys()))
    covertype_qs = sorted(list(covertype_dfs[.99].keys()))
    letter_qs = sorted(list(letter_dfs[.99].keys()))
    mushroom_qs = sorted(list(mushroom_dfs[.99].keys()))

    adult_avgs = [np.average(adult_dfs[.99][q]['t_ssj_ratio']) for q in adult_qs]
    connect4_avgs = [np.average(connect4_dfs[.99][q]['t_ssj_ratio']) for q in connect4_qs]
    covertype_avgs = [np.average(covertype_dfs[.99][q]['t_ssj_ratio']) for q in covertype_qs]
    letter_avgs = [np.average(letter_dfs[.99][q]['t_ssj_ratio']) for q in letter_qs]
    mushroom_avgs = [np.average(mushroom_dfs[.99][q]['t_ssj_ratio']) for q in mushroom_qs]

    # plot
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)

    axs = [ax1, ax2, ax3, ax4, ax5]

    for ax in axs:
        ax.tick_params(labelsize=15)

    for ax in axs:
        ax.grid(axis='y')

    ax1.plot(adult_qs, adult_avgs, marker='o', linewidth=4, markersize=10, color='tab:blue')
    ax2.plot(connect4_qs, connect4_avgs, marker='o', linewidth=4, markersize=10, color='tab:orange')
    ax3.plot(covertype_qs, covertype_avgs, marker='o', linewidth=4, markersize=10, color='tab:green')
    ax4.plot(letter_qs, letter_avgs, marker='o', linewidth=4, markersize=10, color='tab:red')
    ax5.plot(mushroom_qs, mushroom_avgs, marker='o', linewidth=4, markersize=10, color='tab:purple')

    # labels and titles
    ax1.set_title('Adult', fontsize=15)
    ax2.set_title('Connect-4', fontsize=15)
    ax3.set_title('Covertype', fontsize=15)
    ax4.set_title('Letter', fontsize=15)
    ax5.set_title('Mushroom', fontsize=15)

    ax1.set_ylabel(r'$t/t_{PLI}$', fontsize=15)
    ax4.set_ylabel(r'$t/t_{PLI}$', fontsize=15)
    ax2.set_xlabel('query size', fontsize=15)
    plt.text(-0.1, -0.1, 'query size', ha='center', va='center', fontsize=15, transform=plt.gca().transAxes)

if __name__ == '__main__':
    print('hello')
