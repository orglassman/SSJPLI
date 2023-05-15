import os
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from common import load_pickle

alphas = np.arange(10, 50, 10)
betas = np.arange(10, 50, 5)
coverages = [.25, .5, .75, .9, .99]
bins = [7, 16, 19, 23]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-in_dir', help='path to directory containing .pkl files')
    args = parser.parse_args()
    return args


def get_ax():
    plt.figure()
    ax = plt.axes()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    return ax


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


def load_dfs(dir):
    files = os.listdir(dir)
    alphas, betas, bins, coverages = get_run_params(files)

    pattern = r'alpha_(\d+).+beta_(\d+).+bin_num_(\d+).+coverage_(\d+\.\d+)'
    dfs = {alpha: {beta: {b: {} for b in bins} for beta in betas} for alpha in alphas}
    for file in files:
        res = re.search(pattern, file)
        if not res:
            print(f'-W- Check {file}')
            continue
        alpha = int(res.groups(0)[0])
        beta = int(res.groups(0)[1])
        bin_num = int(res.groups(0)[2])
        coverage = float(res.groups(0)[3])
        dfs[alpha][beta][bin_num][coverage] = load_pickle(dir + os.sep + file)

    return dfs


def plot_graph_1(dfs):
    ax = get_ax()

    records_7 = [np.average(dfs[c][7]['N']) for c in coverages]
    records_16 = [np.average(dfs[c][16]['N']) for c in coverages]
    records_19 = [np.average(dfs[c][19]['N']) for c in coverages]
    records_23 = [np.average(dfs[c][23]['N']) for c in coverages]
    records_7_std = [np.std(dfs[c][7]['N']) for c in coverages]
    records_16_std = [np.std(dfs[c][16]['N']) for c in coverages]
    records_19_std = [np.std(dfs[c][19]['N']) for c in coverages]
    records_23_std = [np.std(dfs[c][23]['N']) for c in coverages]
    records_7_y0 = [x - y for x, y in zip(records_7, records_7_std)]
    records_7_y1 = [x + y for x, y in zip(records_7, records_7_std)]
    records_16_y0 = [x - y for x, y in zip(records_16, records_16_std)]
    records_16_y1 = [x + y for x, y in zip(records_16, records_16_std)]
    records_19_y0 = [x - y for x, y in zip(records_19, records_19_std)]
    records_19_y1 = [x + y for x, y in zip(records_19, records_19_std)]
    records_23_y0 = [x - y for x, y in zip(records_23, records_23_std)]
    records_23_y1 = [x + y for x, y in zip(records_23, records_23_std)]

    ax.plot(coverages, records_7, linewidth=3, marker='o')
    ax.plot(coverages, records_7, linewidth=3, marker='o', color='C0', label=r'H=2.16')
    ax.plot(coverages, records_16, linewidth=3, marker='o', color='C1', label=r'H=5.146')
    ax.plot(coverages, records_19, linewidth=3, marker='o', color='C2', label=r'H=6.146')
    ax.plot(coverages, records_23, linewidth=3, marker='o', color='C3', label=r'H=7.476')
    plt.legend(fontsize=35)

    ax.fill_between(coverages, records_7_y0, records_7_y1, color='lightsteelblue')
    ax.fill_between(coverages, records_16_y0, records_16_y1, color='antiquewhite')
    ax.fill_between(coverages, records_19_y0, records_19_y1, color='gainsboro')
    ax.fill_between(coverages, records_23_y0, records_23_y1, color='thistle')
    plt.legend(fontsize=35)


def plot_graph_2(dfs):
    ax = get_ax()
    h_abs = [2.16, 5.146, 6.146, 7.476]
    records_hs = [np.average(dfs[.25][b]['records']) for b in bins]
    records_hs_stds = [np.std(dfs[.25][b]['records']) for b in bins]
    records_hs_y0 = [x - y for x, y in zip(records_hs, records_hs_stds)]
    records_hs_y1 = [x + y for x, y in zip(records_hs, records_hs_stds)]

    ax.plot(h_abs, records_hs, marker='o', linewidth=3, color='C0')
    ax.fill_between(h_abs, records_hs_y0, records_hs_y1, color='thistle')

    plt.xlabel(r'$H(AB)\;[bit/symbol]$', fontsize=40)
    plt.ylabel('records', fontsize=40)


def plot_graph_3(dfs):
    ax = get_ax()
    t_ssjs = [dfs[c][16]['t'].iloc[31] for c in coverages]
    t_explicits = [dfs[c][16]['t_explicit'].iloc[31] for c in coverages]
    t_traverses = [dfs[c][16]['t_traverse'].iloc[31] for c in coverages]
    ax.plot(coverages, t_ssjs, linewidth=3, marker='o', color='C0', label='SSJ')
    ax.plot(coverages, t_explicits, linewidth=3, marker='o', color='C1', label='explicit')
    ax.plot(coverages, t_traverses, linewidth=3, marker='o', color='C2', label='traverse')
    plt.xlabel('coverage', fontsize=40)
    plt.ylabel('time [sec]', fontsize=40)
    plt.legend(fontsize=35)


def plot_graph_3_stds(dfs):
    ax = get_ax()
    t_ssjs = np.array([np.average(dfs[c][16]['t']) for c in coverages])
    t_ssjs_std = np.array([np.std(dfs[c][16]['t']) for c in coverages])
    t_explicits = np.array([np.average(dfs[c][16]['t_explicit']) for c in coverages])
    t_explicits_std = np.array([np.std(dfs[c][16]['t_explicit']) for c in coverages])
    t_traverses = np.array([np.average(dfs[c][16]['t_traverse']) for c in coverages])
    t_traverses_std = np.array([np.std(dfs[c][16]['t_traverse']) for c in coverages])

    t_ssjs_y0 = t_ssjs - t_ssjs_std
    t_ssjs_y1 = t_ssjs + t_ssjs_std
    t_explicits_y0 = t_explicits - t_explicits_std
    t_explicits_y1 = t_explicits + t_explicits_std
    t_traverses_y0 = t_traverses - t_traverses_std
    t_traverses_y1 = t_traverses + t_traverses_std
    ax.fill_between(coverages, t_ssjs_y0, t_ssjs_y1, color='lightsteelblue')
    ax.fill_between(coverages, t_explicits_y0, t_explicits_y1, color='antiquewhite')
    ax.fill_between(coverages, t_traverses_y0, t_traverses_y1, color='gainsboro')
    ax.plot(coverages, t_ssjs, marker='o', linewidth=3, color='C0', label='SSJ')
    ax.plot(coverages, t_explicits, marker='o', linewidth=3, color='C1', label='explicit')
    ax.plot(coverages, t_traverses, marker='o', linewidth=3, color='C2', label='traverse')
    plt.legend(fontsize=35)
    plt.ylabel('time [sec]', fontsize=40)
    plt.xlabel('coverage', fontsize=40)


def plot_graph_4(dfs):
    ax = get_ax()
    h_ssjs = [np.average(dfs[c][16]['H']) for c in coverages]
    h_trues = [np.average(dfs[c][16]['H_true']) for c in coverages]
    h_qs = [np.average(dfs[c][16]['HQ']) for c in coverages]
    h_ssjs_std = [np.std(dfs[c][16]['H']) for c in coverages]
    h_trues_std = [np.std(dfs[c][16]['H_true']) for c in coverages]
    h_qs_std = [np.std(dfs[c][16]['HQ']) for c in coverages]
    h_ssjs_y0 = [x - y for x, y in zip(h_ssjs, h_ssjs_std)]
    h_ssjs_y1 = [x + y for x, y in zip(h_ssjs, h_ssjs_std)]
    h_trues_y0 = [x - y for x, y in zip(h_trues, h_trues_std)]
    h_trues_y1 = [x + y for x, y in zip(h_trues, h_trues_std)]
    h_qs_y0 = [x - y for x, y in zip(h_qs, h_qs_std)]
    h_qs_y1 = [x + y for x, y in zip(h_qs, h_qs_std)]

    ax.plot(coverages, h_ssjs, marker='o', linewidth=3, color='C0', zorder=10, label='sampled')
    ax.plot(coverages, h_trues, marker='o', linewidth=3, color='C1', label='true')
    ax.plot(coverages, h_ssjs, marker='o', linewidth=3, color='C2', label='missampled')
    plt.legend(fontsize=35)

    ax.fill_between(coverages, h_ssjs_y0, h_ssjs_y1, zorder=10, color='lightsteelblue')
    ax.fill_between(coverages, h_trues_y0, h_trues_y1, color='antiquewhite')
    ax.fill_between(coverages, h_qs_y0, h_qs_y1, color='gainsboro')

    plt.xlabel('coverage', fontsize=40)
    plt.ylabel(r'$H\;[bit/symbol]$', fontsize=40)


def plot_graph_5(dfs):
    alphas = [10, 20, 30, 40]
    betas = [10, 15, 20, 25, 30, 35, 40, 45]
    dfs = {al: {be: {b: {} for b in bins} for be in betas} for al in alphas}
    pattern = 'alpha_(\d+)_beta_(\d+).+bin_num_(\d+)_HAB_avg_(\d+\.\d+)_coverage_(\d+\.\d+)'

    pass


def main():
    args = parse_args()
    in_dir = args.in_dir
    dfs = load_dfs(in_dir)
    print('hello world')


if __name__ == '__main__':
    main()
