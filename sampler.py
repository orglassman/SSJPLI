import itertools
import time
from random import choices

import numpy as np

from common import sort_by_key, flatten, DKL
from dataset import RealDataSet


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


class Sampler:
    def __init__(self, dataset, partition_factor=1):
        self.dataset = dataset
        self.partition_factor = partition_factor

        self.N = dataset.get_N()
        self.M = dataset.get_M()
        self.omega = dataset.get_omega()

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

        self._tids = tuple2tid

    # handle only partition_factor=1 for now
    def fetch_tids(self, X):
        res = {}
        for A in X:
            res[A] = self._tids[A]

        return res

    def entropy(self, X, coverage=0.9, mode='ssj'):
        if len(X) == self.M:
            return {f'H_{mode}': np.log2(self.N)}


        if mode == 'ssj':
            return self.entropy_ssj(X, coverage)
        elif mode == 'is':
            return self.entropy_is(X, coverage)
        elif mode == 'explicit':
            return self.entropy_explicit(X)
        else:
            return self.entropy_pli(X)  # deterministic, coverage=1

    # main entropy
    def entropy_ssj(self, X, coverage):
        """approximate H(X) using SSJ"""
        tids = self.fetch_tids(X)
        product_set_size = self.get_product_set_size(tids)

        rho = 0
        total_time = 0
        total_samples = 0
        current = tids.pop(X[0])
        i = 1
        while i < len(X):
            next_tid = tids.pop(X[i])
            start = time.perf_counter()
            res_data = self.ssj(current, next_tid, coverage=coverage)
            finish = time.perf_counter()

            rho = res_data['rho']
            current = res_data['frequencies']
            total_samples += res_data['num_samples']
            total_time += finish - start
            i += 1

        res_data = {
            't_ssj': total_time,
            'frequencies': current,
            'HUN': self.get_unnormalized_entropy(current),
            'H_ssj': H_dict(self.get_dist_from_frequency_table(current)),
            'rho': rho,
            'sigma': coverage,
            'N': self.dataset.get_N(),
            'product_set_size': product_set_size,
            'samples': total_samples
        }

        return res_data

    def entropy_pli(self, X):
        tids = self.fetch_tids(X)
        current = tids.pop(X[0])
        i = 1

        start = time.perf_counter()
        while i < len(X):
            next_tid = tids.pop(X[i])

            frequencies = {}

            for a, ta in current.items():
                for b, tb in next_tid.items():
                    I = list(set(ta).intersection(set(tb)))
                    if bool(I):
                        frequencies[flatten((a, b))] = I

            current = frequencies
            i += 1
        finish = time.perf_counter()

        res_data = {
            't_pli': finish - start,
            'frequencies': current,
            'H_pli': H_dict(self.get_dist_from_frequency_table(current))
        }

        return res_data

    def entropy_is(self, X, coverage):
        return None

    #     distributions = self.generate_distributions_ssj()
    #     # IS - accumulate H with each sample
    #     if not coverage:
    #         coverage = self.coverage
    #
    #     H_baseline = self.entropy_baseline()
    #     target_H = H_baseline * coverage
    #
    #     H_is = 0
    #     sampled = {}
    #     frequencies = {}
    #     num_samples = 0
    #
    #     start_time = time.perf_counter()
    #     while H_is < target_H:
    #         num_samples += 1
    #
    #         instance = self.sample_instance_ssj(distributions)
    #         x = tuple(instance.values())
    #
    #         if x in sampled.keys():
    #             continue
    #
    #         sampled[x] = 1
    #
    #         lists = [self._tids[attribute][value] for attribute, value in instance.items()]
    #         sets = map(set, lists)
    #         intersection_indices = sorted(list(set.intersection(*sets)))
    #         L = len(intersection_indices)
    #         if L == 0:
    #             continue
    #
    #         frequencies[x] = intersection_indices
    #
    #         # compute weight function and add to total
    #         H_is += self.aggregate_IS(distributions, x, L)
    #     finish_time = time.perf_counter()
    #
    #     frequencies = sort_by_key(frequencies)
    #     rho = H_is / H_baseline
    #     res_data = {
    #         'H': H_is,
    #         'frequencies': frequencies,
    #         'num_samples': num_samples,
    #         'sigma': coverage,
    #         'rho': rho,
    #         'time': finish_time - start_time
    #     }
    #
    #     return res_data

    def entropy_explicit(self, X):
        df = self.dataset.df[X]
        dist = {}
        row_counter = 0

        start = time.perf_counter()
        for index, row in df.iterrows():
            row_counter += 1
            symbol = tuple(row.values)
            if symbol in dist.keys():
                dist[symbol] += 1
            else:
                dist[symbol] = 1
        finish = time.perf_counter()

        res_data = {
            't_explicit': finish - start,
            'H_explicit': H_dict(dist)
        }
        return res_data

    def get_product_set_size(self, tids):
        lens = [len(tids[t].keys()) for t in tids.keys()]
        res = 1
        for l in lens:
            res *= l
        return res

    def get_unnormalized_entropy(self, TX):
        HUN = 0
        for x in TX.values():
            q2 = len(x) / self.N
            HUN -= q2 * np.log2(q2)
        return HUN

    def get_dist_from_frequency_table(self, frequencies):
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

    def entropy_pandas(self):
        """compute H(X) by accessing data directly"""
        occurrences = self.dataset.df.value_counts()
        dist = {k: v for k, v in zip(occurrences.index.to_list(), occurrences.values.tolist())}

        return H_dict(dist)

    # ssj utilities
    def ssj(self, TA, TB, coverage):
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
        tids = {
            'A': TA,
            'B': TB
        }
        sampling_weights = self.gen_ssj_sampling_weights(tids)




        target_N = int(self.N * coverage) + 1
        total_sampled = 0

        frequencies = {}  # result TX
        sampled = {}  # for resamples
        nulls = []  # for x s.t. I(x)=\emptyset
        num_samples = 0
        KLs = []
        while total_sampled < target_N:


            # real_distributions = self.build_dist_real(X, frequencies) # real distribution of remaining entries
            # sample_distributions = self.build_dist_ssj(total_sampled, sampling_weights) # ssj distribution of remaining entries

            # KLs.append(DKL(real_distributions, sample_distributions))


            num_samples += 1
            instance = self.sample_instance_ssj(sampling_weights)
            x = flatten(tuple(instance.values()))

            if x in sampled.keys():
                L = 0
            else:
                sampled[x] = 1
                # I(x) = I(a) \cap I(b)
                lists = [tids[attribute][value] for attribute, value in instance.items()]
                sets = map(set, lists)
                intersection_indices = sorted(list(set.intersection(*sets)))
                L = len(intersection_indices)

                if L == 0:
                    nulls.append(x)
                else:
                    frequencies[x] = intersection_indices

            # update distributions. break if dist empty
            sampling_weights = self.update_ssj_sampling_weights(sampling_weights, instance, L)
            if not sampling_weights:
                break
            total_sampled += L

        res_data = {
            'frequencies': sort_by_key(frequencies),
            'num_samples': num_samples,
            'sigma': coverage,
            'rho': total_sampled / self.N,
            'nulls': nulls
        }

        return res_data

    def gen_ssj_sampling_weights(self, tids):
        weights = {
            'A': {},
            'B': {}
        }
        for attribute, freq_table in tids.items():
            for instance, tid in freq_table.items():
                weights[attribute][instance] = len(tid)

        return weights

    def sample_instance_ssj(self, sampling_weights):
        res = {}
        for attribute in sampling_weights.keys():
            domain = list(sampling_weights[attribute].keys())
            weights = sampling_weights[attribute].values()
            try:
                res[attribute] = choices(domain, weights=weights)[0]
            except:
                print('h')

        return res

    def update_ssj_sampling_weights(self, distributions, instance, L):
        for attribute, value in instance.items():
            distributions[attribute][value] -= L
            if distributions[attribute][value] == 0:
                del distributions[attribute][value]

                # empty distributions
                if not distributions[attribute]:
                    return None

        return distributions

    def build_dist_real(self, X, frequencies):
        sampled_indices = []
        if bool(frequencies):
            for v in frequencies.values():
                sampled_indices += v

        remaining = self.dataset.df[X].drop(sampled_indices)
        dist = remaining.value_counts(normalize=True).to_dict()
        return dist

    def build_dist_ssj(self, total_sampled, ssj_weights):
        weights_A = ssj_weights['A']
        weights_B = ssj_weights['B']

        remaining_N = self.N - total_sampled
        dist = {}
        for a in weights_A.keys():
            for b in weights_B.keys():
                x = (a, b)
                dist[x] = weights_A[a] * weights_B[b] / remaining_N**2

        return dist

    # for stochastic join, error analysis
    def get_bounds(self, res_data):
        mis_sampled = self.reduce_sampled(res_data['frequencies'])  # all pairs in product set not sampled
        occur_in_R, not_occur_in_R = self.split_mis_sampled(mis_sampled)

        # build distribution for pairs with I(x)>0
        HQ = H_dict(occur_in_R)
        HQ_UN = 0
        for x in occur_in_R.values():
            q = x / self.N
            HQ_UN -= q * np.log2(q)

        res = {
            'HQ': HQ,
            'HQUN': HQ_UN,
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


if __name__ == '__main__':
    in_file = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Adult\\adult_categorical.csv"
    ds = RealDataSet(in_file)
    sampler = Sampler(ds)

    X = list('ABC')
    res1 = sampler.entropy(X, mode='pli')
    res2 = sampler.entropy(X, mode='ssj')
    print('hello')
