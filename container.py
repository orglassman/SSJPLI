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
    parser.add_argument('-R', help='Number of repetitions per experiment (analysis section for lowering variance)', default=1000),
    parser.add_argument('-mode', help='SSJ/IS', default='SSJ')
    args = parser.parse_args()
    return args


class DatasetContainer:
    def __init__(self, out_dir, alpha=5, beta=5, L=20, R=1000, mode='SSJ'):
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

    def stratum_overview(self, stratum):
        """analyze sampling within bin"""
        datasets = self.strata[stratum]

        if not bool(datasets):
            return

        Is = [d.I for d in datasets]
        Hs = [d.HAB() for d in datasets]

        HNs = []  # normalized
        HUNs = []  # unnormalized
        NSs = []
        Sigmas = []
        Rhos = []
        times = []
        U1s = []  # HN + HQN (both normalized)
        U2s = []  # HN + HQUN
        U3s = []  # HUN + HQN
        U4s = []  # HUN + HQUN (should equal H(X))
        U5s = []  # new bound
        U6s = []  # new bound
        for dataset in datasets:
            sampler = SyntheticSampler(dataset)
            res_data = sampler.entropy_ssj()

            HNs.append(res_data['HN'])
            HUNs.append(res_data['HUN'])
            NSs.append(res_data['num_samples'])
            Sigmas.append(res_data['sigma'])
            Rhos.append(res_data['rho'])
            times.append(res_data['time'])
            U1, U2, U3, U4, U5, U6 = sampler.get_bounds(res_data)
            U1s.append(U1)
            U2s.append(U2)
            U3s.append(U3)
            U4s.append(U4)
            U5s.append(U5)
            U6s.append(U6)

        data_dict = {
            'HAB': Hs,
            'IAB': Is,
            'N_samples': NSs,
            'time': times,
            'sigma': Sigmas,
            'rho': Rhos,
            'HAB_N': HNs,
            'HAB_UN': HUNs,
            'HN + HQN': U1s,
            'HN + HQUN': U2s,
            'HUN + HQN': U3s,
            'HUN + HQUN': U4s,
            'U5': U5s,
            'U6': U6s
        }
        df = pd.DataFrame(data_dict)
        df.sort_values(by='IAB')
        return df

    def strata_overview(self):
        dfs = {}
        for stratum in self.strata.keys():
            dfs[stratum] = self.stratum_overview(stratum)

        print('-I- Strata overview completed successfully')
        return dfs

    def stratum_varying_coverage(self, stratum, dump=False):
        """analyze varying coverage in every sub population"""
        datasets = self.strata[stratum]
        E_HAB = np.average([x.HAB() for x in datasets])
        if not bool(datasets):
            return

        coverages = [0.25, 0.5, 0.75, 0.9, 0.99]
        for coverage in coverages:
            Hs = []
            H_baselines = []

            U1s = []
            U2s = []
            U3s = []
            U4s = []
            U5s = []
            U6s = []


            NSs = []
            rhos = []
            times = []
            Is = []
            records = []
            for i, dataset in enumerate(datasets):
                sampler = SyntheticSampler(dataset, mode=self.mode)

                H_aggregate = 0
                U1_aggregate = 0
                U2_aggregate = 0
                U3_aggregate = 0
                U4_aggregate = 0
                U5_aggregate = 0
                U6_aggregate = 0
                I_aggregate = 0
                Ns_aggregate = 0
                time_aggregate = 0
                rho_aggregate = 0

                # average over R repetitions
                for i in range(self.R):
                    res_data = sampler.entropy(coverage=coverage)
                    bounds = sampler.get_bounds(res_data)

                    H_aggregate += res_data['H']
                    U1_aggregate += bounds['U1']
                    U2_aggregate += bounds['U2']
                    U3_aggregate += bounds['U3']
                    U4_aggregate += bounds['U4']
                    U5_aggregate += bounds['U5']
                    U6_aggregate += bounds['U6']
                    I_aggregate += dataset.I
                    Ns_aggregate += res_data['num_samples']
                    time_aggregate += res_data['time']
                    rho_aggregate += res_data['rho']

                H_average = H_aggregate / self.R
                U1_average = U1_aggregate / self.R
                U2_average = U2_aggregate / self.R
                U3_average = U3_aggregate / self.R
                U4_average = U4_aggregate / self.R
                U5_average = U5_aggregate / self.R
                U6_average = U6_aggregate / self.R
                I_average = I_aggregate / self.R
                Ns_average = Ns_aggregate / self.R
                time_average = time_aggregate / self.R
                rho_average = rho_aggregate / self.R
                records_num_dataset = dataset.get_N()

                Hs.append(H_average)
                U1s.append(U1_average)
                U2s.append(U2_average)
                U3s.append(U3_average)
                U4s.append(U4_average)
                U5s.append(U5_average)
                U6s.append(U6_average)
                H_baselines.append(dataset.HAB())
                Is.append(I_average)
                NSs.append(Ns_average)
                times.append(time_average)
                rhos.append(rho_average)
                records.append(records_num_dataset)

            data = dict(H=Hs, H_true=H_baselines, U1=U1s, U2=U2s, U3=U3s, U4=U4s, U5=U5s, U6=U6s, I=Is, N=NSs, t=times,
                        rho=rhos, records=records)

            data_df = pd.DataFrame(data)
            if dump:
                self.dump_df(data_df, **{'n_samples_vs_I': '', 'bin_num':stratum, 'HAB_avg': E_HAB, 'coverage': coverage})

        print(f'-I- Check {self.out_dir}')

    def strata_varying_coverage(self):
        for stratum in self.strata.keys():
            num_datasets = len(self.strata[stratum])
            if num_datasets < 30:
                continue

            self.stratum_varying_coverage(stratum, dump=True)

        print(f'-I- Strata coverage analysis completed successfully. Check {self.out_dir}')

    def plot_varying_coverage(self, coverage, data):
        return
        # df = pd.DataFrame(data)
        #
        # sturge = int(np.ceil(np.log2(len(df)) + 1))
        # bins = np.linspace(0, max(df['I']), sturge)
        # bin_ids = np.digitize(df['I'], bins=bins)
        #
        # df['bin'] = bin_ids
        # groups = df.groupby(by='bin')
        #
        # print('hello')


def container_main():
    args = parse_args()
    alpha = int(args.alpha)
    beta = int(args.beta)
    L = int(args.L)
    R = int(args.R)
    mode = args.mode
    out_dir = args.out_dir

    container = DatasetContainer(alpha=alpha, beta=beta, L=L, R=R, out_dir=out_dir, mode=mode)
    container.generate()
    container.bin()
    container.sort()
    container.strata_varying_coverage()

    print('hello')


if __name__ == '__main__':
    container_main()
