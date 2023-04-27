import itertools
from argparse import ArgumentParser
import time
from random import choices

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from common import binary_entropy, sort_by_key


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-in_file', help='path to input CSV')
    parser.add_argument('-X', help='input query. example "A,B"')

    args = parser.parse_args()
    return args


def H_dict(d, base=2):
    """calculate entropy from dictionary where
    keys = symbols
    values = number of occurrences
    """
    N = sum(d.values())
    res = 0
    for v in d.values():
        q = v / N
        res -= q * np.emath.logn(base, q)

    return res


class SyntheticSampler:
    """sequential sampler taking synthetic two-column data"""

    def __init__(self, dataset, coverage=0.9, growth_threshold=None, mode='SSJ'):
        self.dataset = dataset
        self.coverage = coverage
        self.growth_threshold = growth_threshold
        self.mode = mode

        self.N = dataset.get_N()
        self.M = 2
        self.omega = ['A', 'B']

        self.load_data()

    def load_data(self):
        # gen tids
        tuple2tid = {k: {} for k in self.omega}

        for x in self.dataset.df.itertuples():
            y = x._asdict()
            tid = y.pop('Index')  # current index

            for k, v in y.items():
                if v in tuple2tid[k]:
                    tuple2tid[k][v].append(tid)
                else:
                    tuple2tid[k][v] = [tid]

        # remove singletons
        # res_tids = {k: {} for k in self.omega}
        # for attribute in tuple2tid.keys():
        #     for instance, tid in tuple2tid[attribute].items():
        #         if len(tid) >= 1:
        #             res_tids[attribute][instance] = tid

        # self._tids = res_tids
        self._tids = tuple2tid

    def entropy(self, coverage=None):
        if self.mode == 'SSJ':
            return self.entropy_ssj(coverage)
        else:
            return self.entropy_is(coverage)

    def entropy_ssj(self, coverage=None):
        """approximate H(X) using SSJ"""
        start_time = time.perf_counter()
        res_data = self.ssj(coverage=coverage)
        end_time = time.perf_counter()

        # add time
        res_data['time'] = end_time - start_time

        # get output frequencies
        TX = res_data['frequencies']

        HUN = 0
        for x in TX.values():
            q2 = len(x) / self.N
            HUN -= q2 * np.log2(q2)

        TX_dist = self.get_X_dist(TX)
        HN = H_dict(TX_dist)
        res_data['H'] = HN  # entropy since defined over probability distribution
        res_data['HUN'] = HUN  # technically not entropy

        # add here: upper bounds for H

        return res_data

    def entropy_is(self, coverage):
        distributions = self.generate_distributions()
        # IS - accumulate H with each sample
        if not coverage:
            coverage = self.coverage

        H_baseline = self.entropy_baseline()
        target_H = H_baseline * coverage

        H_is = 0
        sampled = {}
        frequencies = {}
        num_samples = 0

        start_time = time.perf_counter()
        while H_is < target_H:
            num_samples += 1

            instance = self.sample_instance(distributions)
            x = tuple(instance.values())

            if x in sampled.keys():
                continue

            sampled[x] = 1

            lists = [self._tids[attribute][value] for attribute, value in instance.items()]
            sets = map(set, lists)
            intersection_indices = sorted(list(set.intersection(*sets)))
            L = len(intersection_indices)
            if L == 0:
                continue

            frequencies[x] = intersection_indices

            # compute weight function and add to total
            H_is += self.aggregate_IS(distributions, x, L)
        finish_time = time.perf_counter()

        frequencies = sort_by_key(frequencies)
        rho = H_is / H_baseline
        res_data = {
            'H': H_is,
            'frequencies': frequencies,
            'num_samples': num_samples,
            'sigma': coverage,
            'rho': rho,
            'time': finish_time - start_time
        }

        return res_data

    def get_X_dist(self, frequencies):
        lens = []
        for v in frequencies.values():
            lens.append(len(v))

        res = {k: v for k, v in zip(frequencies.keys(), lens)}
        return res

    def aggregate_IS(self, distributions, x, L):
        """
        W(X) = P(X)/Q(X)
        """
        P = L / self.N
        # a = x[0]
        # b = x[1]
        # Qa = distributions['A'][a] / self.N
        # Qb = distributions['B'][b] / self.N

        # W = P / (Qa * Qb)
        # return W * np.log2(1 / P)
        return P * np.log2(1 / P)

    def entropy_baseline(self):
        """compute H(X) by accessing data directly"""
        occurrences = self.dataset.df.value_counts()
        dist = {k: v for k, v in zip(occurrences.index.to_list(), occurrences.values.tolist())}

        return H_dict(dist)

    def ssj(self, coverage=None):
        """
        main workhorse.
        1. gen distribution per predicate
        2. init global N
        3. sample
        4. update distributions

        return:
        frequencies - output TID
        num_samples - number of samples to hit rho N entries
        sigma - target coverage
        rho - effective coverage > sigma
        nulls - (a,b) with empty frequency
        """
        distributions = self.generate_distributions()

        if not coverage:
            coverage = self.coverage
        target_N = int(self.N * coverage)
        total_sampled = 0

        growth_counter = 0

        frequencies = {}  # result TX
        sampled = {}  # for resamples
        nulls = []  # for x s.t. I(x)=\emptyset
        num_samples = 0
        while total_sampled < target_N:
            num_samples += 1
            instance = self.sample_instance(distributions)
            x = tuple(instance.values())

            if x in sampled.keys():
                L = 0
                growth_counter += 1  # track failed samples
            else:
                sampled[x] = 1
                # I(x) = I(a) \cap I(b)
                lists = [self._tids[attribute][value] for attribute, value in instance.items()]
                sets = map(set, lists)
                intersection_indices = sorted(list(set.intersection(*sets)))
                L = len(intersection_indices)

                if L == 0:
                    nulls.append(x)
                    growth_counter += 1  # track failed samples
                else:
                    frequencies[x] = intersection_indices
                    growth_counter = 0  # reset for successful samples

            # update distributions. if some dist is empty break
            distributions = self.update_distributions(distributions, instance, L)
            if not distributions:
                break
            total_sampled += L

            # track growth
            if self.growth_threshold:
                if growth_counter == self.growth_threshold:
                    print(f'-W- Growth too slow. Sampling terminated')
                    break

        frequencies = dict(sorted(frequencies.items(), key=lambda item: item[0]))
        rho = total_sampled / self.N  # effective coverage
        res_data = {
            'frequencies': frequencies,
            'num_samples': num_samples,
            'sigma': coverage,
            'rho': rho,
            'nulls': nulls
        }
        return res_data

    def generate_distributions(self):
        distributions = {}
        for attribute, instances in self._tids.items():
            d = {x: len(self._tids[attribute][x]) for x in self._tids[attribute].keys()}
            d = dict(sorted(d.items(), key=lambda item: item[1]))
            distributions[attribute] = d

        return distributions

    def sample_instance(self, distributions):
        res = {}
        for attribute in distributions.keys():
            domain = list(distributions[attribute].keys())
            weights = distributions[attribute].values()
            try:
                res[attribute] = choices(domain, weights=weights)[0]
            except:
                print('h')

        return res

    def update_distributions(self, distributions, instance, L):
        for attribute, value in instance.items():
            distributions[attribute][value] -= L
            if distributions[attribute][value] == 0:
                del distributions[attribute][value]

                # empty distributions
                if not distributions[attribute]:
                    return None

        return distributions

    def framework(self, X):
        """
        test sampling algorithm for varying coverage. compute lower and upper bounds on H(X)
        """
        coverages = [x / 100 for x in np.arange(55, 105, 5)]
        H_baseline = [self.entropy_baseline(None, X)] * len(coverages)

        Hs_ssj_normalized = []
        Hs_ssj_not_normalized = []
        U1s = []
        U2s = []
        U3s = []
        U4s = []
        Ns = []
        for c in coverages:
            res_data = self.entropy_ssj(coverage=c)
            U1, U2, U3, U4, U5, U6 = self.get_bounds(res_data)

            Hs_ssj_normalized.append(res_data['HN'])
            Hs_ssj_not_normalized.append(res_data['HUN'])
            Ns.append(res_data['num_samples'])
            U1s.append(U1)
            U2s.append(U2)
            U3s.append(U3)
            U4s.append(U4)

        plt.plot(coverages, H_baseline)
        plt.plot(coverages, Hs_ssj_normalized)
        plt.plot(coverages, Hs_ssj_not_normalized)
        plt.plot(coverages, U1s)
        plt.plot(coverages, U2s)
        plt.plot(coverages, U3s)
        plt.plot(coverages, U4s)
        plt.legend([
            r'$H$',
            r'$H^{sampled,normalized}$',
            r'$H^{sampled,unnormalized}$',
            r'$H_{Q}^{normalized}+H^{sampled,normalized}$',
            r'$H_{Q}^{normalized}+H^{sampled,unnormalized}$',
            r'$H_{Q}^{unnormalized}+H^{sampled,normalized}$',
            r'$H_{Q}^{unnormalized}+H^{sampled,unnormalized}$',
        ])
        print('hello')

    def get_bounds(self, res_data):
        mis_sampled = self.reduce_sampled(res_data['frequencies'])  # all pairs in product set not sampled
        occur_in_R, not_occur_in_R = self.split_mis_sampled(mis_sampled)

        # build distribution for pairs with I(x)>0
        HQ = H_dict(occur_in_R)
        HQ_UN = 0
        for x in occur_in_R.values():
            q = x / self.N
            HQ_UN -= q * np.log2(q)

        # Ps = [res_data['H'], res_data['HUN']]
        # Qs = [HQ, HQ_UN]
        # U1 = Ps[0] + Qs[0]  # HN + HQN (both normalized)
        # U2 = Ps[0] + Qs[1]  # HN + HQUN
        # U3 = Ps[1] + Qs[0]  # HUN + HQN # makes no sense
        # U4 = Ps[1] + Qs[1]  # HUN + HQUN (should equal H(X))
        #
        # # compute new bound
        # rho = res_data['rho']
        # rho_bar = 1 - rho
        # U5 = rho * Ps[0] + rho_bar * Qs[0] - binary_entropy(rho)
        # U6 = rho * Ps[0] + rho_bar * np.log2(len(occur_in_R)) - binary_entropy(rho)
        #
        # bounds = {
        #     'U1': U1,
        #     'U2': U2,
        #     'U3': U3,
        #     'U4': U4,
        #     'U5': U5,
        #     'U6': U6
        # }
        # return bounds
        res = {
            'HQ': HQ,
            'HQ_UN': HQ_UN,
            'MISS': len(occur_in_R.keys()),
            'EMPTY': len(not_occur_in_R.keys())
        }
        return res

    def reduce_sampled(self, frequencies):
        """pairs in product set not sampled by SSJ"""
        target_tids = self._tids
        As = target_tids['A'].keys()
        Bs = target_tids['B'].keys()
        product_set = {k: 1 for k in itertools.product(As, Bs)}
        sampled = {k: 1 for k in frequencies.keys()}

        mis_sampled = {}
        for k in product_set.keys():
            if k not in sampled:
                mis_sampled[k] = 1

        return mis_sampled

    def split_mis_sampled(self, mis_sampled):
        df = self.dataset.df
        vc = df.value_counts()

        instances = vc.index.to_list()
        frequencies = vc.values.tolist()
        data_pairs = {k: v for k, v in zip(instances, frequencies)}

        occur_in_R = {}
        not_occur_in_R = {}

        for k in mis_sampled.keys():
            if k in data_pairs:
                occur_in_R[k] = data_pairs[k]
            else:
                not_occur_in_R[k] = 1

        return occur_in_R, not_occur_in_R

    def get_H_mis_sampled(self, Q):
        NQ = sum([len(x) for x in Q.values()])
        res = 0
        for v in Q.values():
            q = v / NQ
            res -= q * np.log(q)

        return res


def edbt_main():
    print('hello')


if __name__ == '__main__':
    edbt_main()
