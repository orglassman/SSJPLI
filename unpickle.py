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

def analyze_single_dataset_single_bin(target_bin, target_dataset):
    paths = [
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_7_HAB_avg_2.160964047443681_coverage_0.5.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_7_HAB_avg_2.160964047443681_coverage_0.9.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_7_HAB_avg_2.160964047443681_coverage_0.25.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_7_HAB_avg_2.160964047443681_coverage_0.75.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_7_HAB_avg_2.160964047443681_coverage_0.99.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_16_HAB_avg_5.1467310096163565_coverage_0.5.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_16_HAB_avg_5.1467310096163565_coverage_0.9.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_16_HAB_avg_5.1467310096163565_coverage_0.25.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_16_HAB_avg_5.1467310096163565_coverage_0.75.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_16_HAB_avg_5.1467310096163565_coverage_0.99.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_19_HAB_avg_6.146298065791048_coverage_0.5.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_19_HAB_avg_6.146298065791048_coverage_0.9.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_19_HAB_avg_6.146298065791048_coverage_0.25.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_19_HAB_avg_6.146298065791048_coverage_0.75.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_19_HAB_avg_6.146298065791048_coverage_0.99.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_23_HAB_avg_7.476752128611231_coverage_0.5.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_23_HAB_avg_7.476752128611231_coverage_0.9.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_23_HAB_avg_7.476752128611231_coverage_0.25.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_23_HAB_avg_7.476752128611231_coverage_0.75.pkl",
        "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Outputs\\strata analysis\\alpha_10_beta_20_ssj_no_growth\\n_samples_vs_I__bin_num_23_HAB_avg_7.476752128611231_coverage_0.99.pkl"]
    bins = [7, 16, 19, 23]
    coverages = [.25, .5, .75, .9, .99]
    pattern = r'bin_num_(\d+)_HAB_avg_(\d+\.\d+)_coverage_(\d+\.\d+)'
    dfs = {b: {} for b in bins}
    for path in paths:
        res = re.search(pattern, path)
        if not res:
            print(f'-W- Check {path}')
            continue
        bin_num = int(res.groups(0)[0])
        HAB_avg = round(float(res.groups(0)[1]), 3)
        coverage = float(res.groups(0)[2])
        dfs[bin_num][coverage] = load_pickle(path)

    features = ['H', 'H_true', 'HQ', 'HQUN', 'MISS', 'EMPTY', 'I', 'N', 't', 'rho', 'records']
    H_points = []
    H_true_points = []
    HQ_points = []
    HQUN_points = []
    miss_points = []
    empty_points = []
    I_points = []
    N_points = []
    t_points = []
    rho_points = []
    for c in coverages:
        H_points.append(dfs[target_bin][c]['H'].iloc[target_dataset])
        H_true_points.append(dfs[target_bin][c]['H_true'].iloc[target_dataset])
        HQ_points.append(dfs[target_bin][c]['HQ'].iloc[target_dataset])
        HQUN_points.append(dfs[target_bin][c]['HQUN'].iloc[target_dataset])
        miss_points.append(dfs[target_bin][c]['MISS'].iloc[target_dataset])
        empty_points.append(dfs[target_bin][c]['EMPTY'].iloc[target_dataset])
        I_points.append(dfs[target_bin][c]['I'].iloc[target_dataset])
        N_points.append(dfs[target_bin][c]['N'].iloc[target_dataset])
        t_points.append(dfs[target_bin][c]['t'].iloc[target_dataset])
        rho_points.append(dfs[target_bin][c]['rho'].iloc[target_dataset])

    Us = [x + np.log2(y) for x, y in zip(H_points, miss_points)]

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

# def isolate_single_dataset_bin(bin_num, dataset_num):
#     for c, df in dfs.items():
#         U6s[c] = df['U6'].iloc[0]
#     U6s = sort_by_key(U6s)

if __name__ == '__main__':
    # unpickle_main()
    analyze_single_dataset_single_bin(target_bin=23, target_dataset=31)
    print('hello')