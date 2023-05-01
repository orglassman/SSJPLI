#!/usr/bin/env python

import os.path
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr

from common import sort_by_key
from dataset import DataSet
from sampler import SyntheticSampler


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-out_dir', help='Path to output directory')
    parser.add_argument('-alpha', help='Cardinality of A', default=5)
    parser.add_argument('-beta', help='Cardinality of B', default=10)
    parser.add_argument('-L', help='Number of repetitions per value of k', default=10)
    parser.add_argument('-R', help='Number of repetitions per experiment (analysis section for lowering variance)',
                        default=1000),
    parser.add_argument('-mode', help='SSJ/IS', default='SSJ')
    parser.add_argument('-bins', help='Analyze specific bins. Separate with commas e.g.: 7,16,19,23', default=None)
    args = parser.parse_args()
    return args


def parse_bins(bin_str):
    if bin_str:
        return [int(x) for x in bin_str.split(',')]

    return None


class DatasetContainer:
    def __init__(self, out_dir, alpha=5, beta=5, L=20, R=1000, mode='SSJ', target_bins=None):
        """
        alpha - cardinality A
        beta - cardinality B
        L - repetitions per value of k
        where
        k - number of entries to be removed, integer between [1, alpha*beta-2]


        R - repetition factor for analysis part. number of repetitions for aggregate results (LPF!)
        """
        self.out_dir = out_dir
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.R = R
        self.datasets = {}
        self.num_datasets = 0
        self.mode = mode
        self.target_bins = target_bins

        print(f'-I- New container initialized with (alpha,beta)=({alpha},{beta}) and L={L}. Sampling method {mode}')

    def generate(self):
        dataset = DataSet(alpha=self.alpha, beta=self.beta)
        for k in range(1, dataset.get_N() - 2):
            cumulative = 0
            while cumulative < self.L:
                dataset.drop(k)
                # if self.cardinality_reduced(dataset):
                #     dataset.reset()
                #     continue

                HAB = dataset.HAB()
                new = dataset.clone()

                if HAB in self.datasets.keys():
                    self.datasets[HAB].append(new)
                else:
                    self.datasets[HAB] = [new]

                dataset.reset()
                cumulative += 1

        self.num_datasets = sum([len(x) for x in self.datasets.values()])
        print(f'-I- Total {self.num_datasets} dataset instances generated successfully')

    def bin(self):
        """sturge's rule for binning"""
        entropies = list(self.datasets.keys())
        sturge = int(np.ceil(np.log2(len(entropies) + 1)))
        freedman_diaconis_width = int(np.ceil(2 * iqr(entropies) / np.cbrt(len(entropies))))
        freedman_diaconis_num = int(np.ceil((max(entropies) - min(entropies)) / freedman_diaconis_width))

        bins = np.linspace(0, max(entropies), num=sturge * 3)
        indices = np.digitize(entropies, bins)

        self.strata = {b: [] for b in range(1, len(bins) + 1)}

        for HAB, idx in zip(entropies, indices):
            self.strata[idx] += self.datasets[HAB]

        for bin, stratum in self.strata.items():
            print(f'-I- Bin {bin}: {len(stratum)} records')

    def sort(self):
        """sort by I(A;B) within each bin"""
        for bin_num in self.strata.keys():
            self.strata[bin_num] = sorted(self.strata[bin_num])

        print('-I- Strata sorted successfully')

    def dump_df(self, df, **kwargs):
        out_file = f'{self.out_dir}{os.sep}'
        for k, v in kwargs.items():
            out_file += f'{k}_{v}_'
        out_file = out_file.rstrip('_')
        out_file += '.pkl'
        with open(out_file, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'-I- Check {out_file}')

    def stratum_varying_coverage(self, stratum, dump=False):
        """analyze varying coverage in every sub population"""
        datasets = self.strata[stratum]
        E_HAB = np.round(np.average([x.HAB() for x in datasets]), decimals=3)
        if not bool(datasets):
            return

        coverages = [0.25, 0.5, 0.75, 0.9, 0.99]
        for coverage in coverages:
            Hs = []
            H_baselines = []

            HQs = []
            HQUNs = []
            MISSs = []
            EMPTYs = []
            NSs = []
            rhos = []
            times = []
            Is = []
            records = []
            t_explicits = []
            t_traverses = []
            for i, dataset in enumerate(datasets):
                sampler = SyntheticSampler(dataset, mode=self.mode)

                H_aggregate = 0
                HQ_aggregate = 0
                HQUN_aggregate = 0
                MISS_aggregate = 0
                EMPTY_aggregate = 0
                I_aggregate = 0
                Ns_aggregate = 0
                time_aggregate = 0
                rho_aggregate = 0
                t_explicit_aggregate = 0
                t_traverse_aggregate = 0
                # average over R repetitions
                for i in range(self.R):
                    res_data = sampler.entropy(coverage=coverage)
                    bounds = sampler.get_bounds(res_data)
                    t_explicit = sampler.explicit_entropy()
                    t_traverse = sampler.traverse_entropy()

                    H_aggregate += res_data['H']
                    HQ_aggregate += bounds['HQ']
                    HQUN_aggregate += bounds['HQUN']
                    MISS_aggregate += bounds['MISS']
                    EMPTY_aggregate += bounds['EMPTY']
                    I_aggregate += dataset.I
                    Ns_aggregate += res_data['num_samples']
                    time_aggregate += res_data['time']
                    rho_aggregate += res_data['rho']
                    t_explicit_aggregate += t_explicit
                    t_traverse_aggregate += t_traverse

                H_average = H_aggregate / self.R
                HQ_average = HQ_aggregate / self.R
                HQUN_average = HQUN_aggregate / self.R
                MISS_average = MISS_aggregate / self.R
                EMPTY_average = EMPTY_aggregate / self.R
                I_average = I_aggregate / self.R
                Ns_average = Ns_aggregate / self.R
                time_average = time_aggregate / self.R
                rho_average = rho_aggregate / self.R
                records_num_dataset = dataset.get_N()
                t_explicit_average = t_explicit_aggregate / self.R
                t_traverse_average = t_traverse_aggregate / self.R

                Hs.append(H_average)
                HQs.append(HQ_average)
                HQUNs.append(HQUN_average)
                MISSs.append(MISS_average)
                EMPTYs.append(EMPTY_average)
                H_baselines.append(dataset.HAB())
                Is.append(I_average)
                NSs.append(Ns_average)
                times.append(time_average)
                rhos.append(rho_average)
                records.append(records_num_dataset)
                t_explicits.append(t_explicit_average)
                t_traverses.append(t_traverse_average)

            data = dict(H=Hs, H_true=H_baselines, t_explicit=t_explicits, t_traverse=t_traverses, HQ=HQs, HQUN=HQUNs,
                        MISS=MISSs, EMPTY=EMPTYs,
                        I=Is, N=NSs, t=times,
                        rho=rhos, records=records)

            data_df = pd.DataFrame(data)
            if dump:
                self.dump_df(data_df,
                             **{'alpha': self.alpha, 'beta': self.beta, 'varying_cardinality': '', 'bin_num': stratum,
                                'HAB_avg': E_HAB, 'coverage': coverage})

        print(f'-I- Check {self.out_dir}')

    def strata_varying_coverage(self):
        if self.target_bins:
            for target_bin in self.target_bins:
                self.stratum_varying_coverage(target_bin, dump=True)
        else:
            for stratum in self.strata.keys():
                num_datasets = len(self.strata[stratum])
                if num_datasets < 30:
                    continue

                self.stratum_varying_coverage(stratum, dump=True)

        print(f'-I- Strata coverage analysis completed successfully. Check {self.out_dir}')


def container_main():
    args = parse_args()
    alpha = int(args.alpha)
    beta = int(args.beta)
    L = int(args.L)
    R = int(args.R)
    target_bins = parse_bins(args.bins)
    mode = args.mode
    out_dir = args.out_dir

    container = DatasetContainer(alpha=alpha, beta=beta, L=L, R=R, out_dir=out_dir, mode=mode, target_bins=target_bins)
    container.generate()
    container.bin()
    container.sort()
    container.strata_varying_coverage()

    print('hello')


def vary_cardinality_measure_time():
    args = parse_args()
    max_alpha = int(args.alpha)
    max_beta = int(args.beta)
    L = int(args.L)
    R = int(args.R)
    target_bins = parse_bins(args.bins)
    mode = args.mode
    out_dir = args.out_dir

    alphas = np.arange(10, max_alpha, 10)
    betas = np.arange(10, max_alpha, 5)
    for alpha in alphas:
        for beta in betas:
            container = DatasetContainer(alpha=alpha, beta=beta, L=L, R=R, out_dir=out_dir, mode=mode,
                                         target_bins=target_bins)
            container.generate()
            container.bin()
            container.sort()
            container.strata_varying_coverage()

    print('hello')


if __name__ == '__main__':
    # container_main()
    vary_cardinality_measure_time()
