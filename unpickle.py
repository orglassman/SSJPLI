import os
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from common import load_pickle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-in_dir', help='path to directory containing pickle files')
    args = parser.parse_args()
    return args


def plot_single_bin_N_coverage(dfs, label):
    coverages = [.25, .5, .75, .9, .99]
    print(f'For {label}: {len(dfs[.25])} dataset instances')
    averages = [np.average(dfs[c]['N']) for c in coverages]
    plt.plot(coverages, averages, linestyle="-", marker="o", label=label)


def parse_files(dir):
    bins = [7, 16, 19, 23]
    coverages = [.25, .5, .75, .9, .99]
    files = os.listdir(dir)

    dfs = {b: {c: None for c in coverages} for b in bins}

    pattern = r'bin_num_(\d+)_HAB_avg_(\d+\.\d+)_coverage_(\d+\.\d+)'

    for file in files:
        res = re.search(pattern, file)
        if not res:
            print(f'-W- Check {file}')
            continue

        bin_num = int(res.groups(0)[0])
        HAB_avg = round(float(res.groups(0)[1]), 3)
        coverage = float(res.groups(0)[2])
        fullpath = dir + os.sep + file
        dfs[bin_num][coverage] = load_pickle(fullpath)

    return dfs


def plot_N_coverage_per_bin(bin_files, H_values):
    # generate labels for legend
    labels = {}
    for b, h in H_values.items():
        label = r'$H(AB)=${0}, bin={1}'.format(h, b)
        labels[b] = label

    plt.figure()
    # for bin_num, files in bin_files.items():
    #     plot_single_bin_N_coverage(files, labels[bin_num])

    bins_of_interest = [7, 16, 19, 23]
    for bin_num in bins_of_interest:
        plot_single_bin_N_coverage(bin_files[bin_num], labels[bin_num])

    plt.xlabel('coverage', fontsize=15)
    plt.ylabel('samples', fontsize=15)
    plt.title('Number of Samples vs. Coverage', fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    print('hello')


def unpickle_main():
    args = parse_args()
    in_dir = args.in_dir

    bin_files, H_avg = parse_files(in_dir)
    plot_N_coverage_per_bin(bin_files, H_avg)

    return 0


if __name__ == '__main__':
    unpickle_main()
