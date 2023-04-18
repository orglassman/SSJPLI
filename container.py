import os.path
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from dataset import DataSet
from sampler import SyntheticSequentialSampler


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-out_dir', help='Path to output directory')
    parser.add_argument('-alpha', help='Cardinality of A', default=5)
    parser.add_argument('-beta', help='Cardinality of B', default=10)
    parser.add_argument('-L', help='Number of repetitions per value of k', default=10)
    args = parser.parse_args()
    return args


def dump_outputs(dfs, alpha, beta, out_dir):
    sep = os.path.sep
    for stratum, df in dfs.items():
        out_file = f'{out_dir}{sep}alpha_{alpha}_beta_{beta}_bin_{stratum}_records_{len(df)}.pkl'

        with open(out_file, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'-I- Check {out_dir}')


class DatasetContainer:
    def __init__(self, alpha=5, beta=5, L=20):
        """
        alpha - cardinality A
        beta - cardinality B
        L - repetitions per value of k
        where
        k - number of entries to be removed, integer between [1, alpha*beta-2]
        """
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.datasets = {}

        print(f'-I- New container initialized with (alpha,beta)=({alpha},{beta}) and L={L}')

    def generate(self):
        dataset = DataSet(alpha=self.alpha, beta=self.beta)
        for k in range(1, dataset.get_N() - 1):
            for l in range(self.L):
                dataset.drop(k)
                HAB = dataset.HAB()
                new = dataset.clone()

                if HAB in self.datasets.keys():
                    self.datasets[HAB].append(new)
                else:
                    self.datasets[HAB] = [new]

                dataset.reset()

        print('-I- Datasets generated successfully')

    def bin(self):
        """sturge's rule for binning"""
        entropies = list(self.datasets.keys())
        num_bins = int(np.ceil(np.log2(len(entropies) + 1)))
        bins = np.linspace(0, max(entropies), num=num_bins)
        indices = np.digitize(entropies, bins)

        self.strata = {b: [] for b in range(1, len(bins) + 1)}

        for HAB, idx in zip(entropies, indices):
            self.strata[idx] += self.datasets[HAB]

        for bin, stratum in self.strata.items():
            print(f'-I- Bin {bin}: {len(stratum)} records')

    def sort(self):
        """sort by I(A;B) within each bin"""
        for bin in self.strata.keys():
            self.strata[bin] = sorted(self.strata[bin])

        print('-I- Strata sorted successfully')

    def stratum_analysis(self, bin):
        """analyze sampling within bin"""
        datasets = self.strata[bin]
        Is = [d.I for d in datasets]
        Hs = [d.HAB() for d in datasets]

        HNs = []  # normalized
        HUNs = []  # unnormalized
        U1s = []  # HN + HQN (both normalized)
        U2s = []  # HN + HQUN
        U3s = []  # HUN + HQN
        U4s = []  # HUN + HQUN (should equal H(X))
        U5s = []  # new bound
        U6s = []  # new bound

        for dataset in datasets:
            sampler = SyntheticSequentialSampler(dataset)
            res_data = sampler.entropy_ssj()

            HNs.append(res_data['H_normalized'])
            HUNs.append(res_data['H_unnormalized'])

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

    def strata_analysis(self):
        dfs = {}
        for bin in self.strata.keys():
            dfs[bin] = self.stratum_analysis(bin)

        print('-I- Strata analysis completed successfully')
        return dfs


def container_main():
    args = parse_args()
    alpha = int(args.alpha)
    beta = int(args.beta)
    L = int(args.L)
    out_dir = args.out_dir

    container = DatasetContainer(alpha=alpha, beta=beta, L=L)
    container.generate()
    container.bin()
    container.sort()
    dfs = container.strata_analysis()

    # pickle dataframes
    dump_outputs(dfs, alpha, beta, out_dir)

    print('hello')


if __name__ == '__main__':
    container_main()
