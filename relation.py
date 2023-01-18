import itertools
import random
import sys
import time
from math import ceil
from typing import Union, Any

import more_itertools as mit
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy

from common import convert_to_numeric, min_set_cover, flatten


class Relation:
    def __init__(self, path, name='unnamed', l=1, efficient_partition=True, parallel=False, mode='product', coverage=1, hybrid=True, growth=0.1, pace=20):
        """
        path                    - path to input CSV
        l                       - partition M attributes into K = ceil(M/l) subsets
        efficient_partition     - crudely partition data (A,B)(C,D) or take domain size into effect
        parallel                - initialize DB through multiprocessing
        mode                    - how to determine join candidates;
                                    deterministic:
                                    product - take all possible candidates; no pairwise. e.g.
                                                join J tables at once O(2^J)
                                    pli     - set l=1 and do pairwise join (product candidates)

                                    stochastic (all pairwise):
                                    ssj     - sequential sample join; pairwise
                                    mssj    - max ssj - take argmax(P) at each
                                    cssj    - sample A, then sample B

        # sequential sampling control
        coverage                - stop sampling upon covering coverage * N

        growth                  - for predetermined pace (number of samples), determine target growth of output
        pace                    - how many steps to achieve predetermined growth rate
        hybrid                  - if True, revert to 'product' if growth too low for input pace
        """
        t1 = time.perf_counter()

        self._path = path
        self.name = name
        self.l = int(l)

        # flow control
        self._efficient_partition = efficient_partition
        self._parallel = parallel
        self.mode = mode

        # PS
        self._coverage = float(coverage)    # percentage of output to cover thru sampling
        self._hybrid = hybrid               # revert to product if sampling too slow
        self._growth = growth               # track sampling - if too slow revert to product
        self._pace = pace                   # track sampling - if too slow revert to product

        # shared init methods
        self.load_data()
        self.check()
        self.determine_subsets()

        # single TIDs and grouping to subsets
        if mode != 'project':
            self.gen_single_tids()
            self.group_tids()
            if self.l > 1:
                self.gen_multi_tids()

        t2 = time.perf_counter()
        print(f'-I- Relation {name}: init time {t2-t1}')

    # ------------------------------------------------------------------
    #   INIT    --------------------------------------------------------
    # ------------------------------------------------------------------
    def check(self):
        """
        check that:
        - l < M otherwise set l=1
        """
        if self.l > self.M:
            print(
                f'-W- Relation {self.name}: Partition parameter l={self.l} is larger than number of attributes M={self.M}'
            )
            print('-W- setting l=1')
            self.l = 1

        if self.mode not in ['product', 'pli', 'project', 'ssj', 'mssj', 'cssj']:
            print('-E- Join mode unrecognized. Setting to cartesian product')
            self.mode = 'product'

        if self.mode == 'pli':
            self.l = 1


    def load_data(self):
        """
        load data. convert to numeric
        """
        df = convert_to_numeric(pd.read_csv(self._path, index_col=False))
        self._df = df
        self._omega = list(df.columns)
        self.N = df.shape[0]
        self.M = df.shape[1]
        self._K = ceil(self.M / self.l)  # number of subsets

    def determine_subsets(self):
        """
        each attribute assigned its subset
        """
        H = {}

        if self._efficient_partition and self.l > 1:
            H = self.efficient_partitioning()
            self._H = H
            return

        curr_subset = 0
        for i in range(self.M):
            curr_attribute = self._omega[i]

            # start new subset
            if curr_subset not in H.keys():
                H[curr_subset] = [curr_attribute]

                # check increment subset
                if len(H[curr_subset]) == self.l:
                    curr_subset = curr_subset + 1
                continue

            # populate existing subset
            H[curr_subset].append(curr_attribute)

            # check increment subset
            if len(H[curr_subset]) == self.l:
                curr_subset = curr_subset + 1

        self._H = H

    def gen_single_tids(self):
        """ see if single attribute freq tables can be constructed faster"""
        tuple2tid = {k: {} for k in self._omega}

        for x in self._df.itertuples():
            y = x._asdict()
            tid = y.pop('Index')  # current index

            # run over M attributes
            for k, v in y.items():
                if v in tuple2tid[k]:
                    tuple2tid[k][v].append(tid)
                else:
                    tuple2tid[k][v] = [tid]

        # remove singletons
        self._tuple2tid = {k: {} for k in self._omega}
        for attribute in tuple2tid.keys():
            for instance, tids in tuple2tid[attribute].items():
                if len(tids) > 1:
                    self._tuple2tid[attribute][instance] = tids

        # clear memory
        del self._df

    def group_tids(self):
        """split single tids to groups by H"""
        tid_structure = {k: {} for k in self._H.keys()}
        for subset_number, subset_attributes in self._H.items():
            for single_attribute in subset_attributes:
                tid_structure[subset_number][single_attribute] = self._tuple2tid[single_attribute]

        self._TIDs = tid_structure

    def gen_multi_tids(self):
        """
        generate all tids of arity >1
        """

        # when join output is empty, save here
        self._empty_joins = {}

        if self._parallel:
            self.init_parallel()
        else:
            for subset, tids in self._TIDs.items():
                self.single_subset_tids(subset, tids)

    # ------------------------------------------------------------------
    #   MULTI TIDs  ----------------------------------------------------
    # ------------------------------------------------------------------
    def single_subset_tids(self, subset, tids):
        time1 = time.perf_counter()

        if len(tids) < 2:
            time2 = time.perf_counter()
            print(f'-I- Subset {subset}; time {time2 - time1}')
            return

        attributes = tids.keys()
        powerset = mit.powerset(attributes)
        for i in range(len(attributes) + 1):
            n = next(powerset)  # remove empty set + single attributes

        try:
            while (1):
                p = sorted(list(next(powerset)))
                cover, target_tids = self.get_target_tids(p, tids)

                # empty cover when join on p is empty
                if not cover:
                    continue


                #TODO: fix swap issues
                swap = False
                res_name = ','.join(sorted(cover))
                res_sorted_name = ','.join(p)
                if res_sorted_name != res_name:
                    swap = True


                res_data = self.PLI_join(target_tids)
                frequencies = res_data['frequencies']
                if frequencies:
                    tids[res_name] = frequencies
                else:
                    print(f'-W- Join {p}: empty output; {p} is primary key')

        except StopIteration:
            time2 = time.perf_counter()
            print(f'-I- Subset {subset}; time {time2 - time1}')

    def swap_res(self, p, res):
        new_keys = []
        for k in res.keys():
            x = k.split(',')
            tmp = x[-1]
            x[-1] = x[-2]
            x[-2] = tmp
            new_keys.append(','.join(x))

        new_res = dict(zip(new_keys, list(res.values())))

        a = ''.join([''.join(k.split(',')) for k in p])
        final_p = ','.join(sorted(a))

        return new_res, final_p

        #
        # unsorted_columns = []
        # for x in p:
        #     unsorted_columns += x.split(',')
        #
        #
        # inverse = {w:k for k,v in res.items() for w in v}
        # sorted_inverse = dict(sorted(inverse.items(), key=lambda item: item[0]))
        # df = pd.DataFrame([x.split(',') for x in sorted_inverse.values()])
        # df.columns = unsorted_columns
        # swapped = df[final_columns]

    def get_target_tids(self, p, tids):
        tids_keys2lists = [l.split(',') for l in tids.keys()]
        cover = min_set_cover(p, tids_keys2lists)
        # target_tids = {','.join(c):tids[','.join(c)] for c in cover}

        for i in cover:
            if i in self._empty_joins.keys():
                print(f'-W- Join {p}: {i} is empty, join aborted')
                self._empty_joins[p] = 1
                return None

        target_tids = {c: tids[c] for c in cover}
        return cover, target_tids

    def efficient_partitioning(self):
        # sort attributes by cardinality
        x2card = {x: len(self._df[x].unique()) for x in self._omega}
        x2card = dict(sorted(x2card.items(), key=lambda item: item[1]))
        omega_prime = list(x2card.keys())

        # populate subsets
        i = 0
        num_subsets = self._K
        max_subset_size = self.l
        subsets = {j: [] for j in range(num_subsets)}

        while len(omega_prime):
            max_card = omega_prime.pop(-1)
            subsets[i].append(max_card)

            if len(subsets[i]) == max_subset_size:
                i = (i + 1) % num_subsets

            if not len(omega_prime):
                break

            min_card = omega_prime.pop(0)
            subsets[i].append(min_card)

            i = (i + 1) % num_subsets

        for i in subsets.keys():
            tmp = subsets[i]
            subsets[i] = sorted(tmp)

        return subsets

    # ------------------------------------------------------------------
    # JOIN  ------------------------------------------------------------
    # ------------------------------------------------------------------
    def product_LFJ(self, p, target_tids, singletons=False):
        instances = self.cartesian_product(target_tids)

        res = {}
        for x in instances:
            lists = []
            try:
                lists = [target_tids[attribute][value] for attribute,value in zip(target_tids.keys(), x)]
            except:
                print(f'-E- {p}: failed to access TID for instance {x}')

            # perform the join
            sets = map(set, lists)
            intersection_indices = sorted(list(set.intersection(*sets)))

            if not intersection_indices:
                continue

            if (len(intersection_indices)>1) or (singletons==True):
                res[','.join(flatten(x))] = intersection_indices

        res_data = {
            'frequencies': res,
        }
        return res_data

    def PLI_join(self, target_tids):
        """
        when mode=PLI, join is pairwise:
        J(A,B,C) = J(A, J(B,C)) etc.
        """
        key1 = list(target_tids.keys())[0]
        first = target_tids.pop(key1)

        res_name = key1
        res = first
        while len(target_tids):
            # pop next element in target tids
            nkey = list(target_tids.keys())[0]
            next = target_tids.pop(nkey)

            next_targets = {
                res_name: res,
                nkey: next
            }
            res_name += ',' + nkey
            res = self.product_LFJ(res_name, next_targets)['frequencies']

        res_data = {
            'frequencies': res
        }
        return res_data

    def sequential_sampling(self, target_tids):
        """
        1. gen unnormalized distribution per attribute
        2. init global N
        3. sample from distribution
        4. update probability tables:
            target_attribtues -= size of output
            global N -= size of output
        """
        distributions = self.generate_distributions(target_tids)

        target_N = self.N * self._coverage
        total_sampled = 0

        # target_growth = self.N * self._growth
        # target_growth = target_N * self._growth
        # pace = self._pace
        # growth_tracker = 0
        # growth_counter = 0

        res = {}
        sampled = {}
        sampled_empty = {}
        samples = 0

        while (total_sampled < target_N):
            samples = samples + 1

            # track growth
            # growth_counter = growth_counter + 1
            # if growth_counter == pace:
            #     if growth_tracker > target_growth:
            #         growth_counter = 0
            #         growth_tracker = 0
            #     else:
            #         if self._hybrid:
            #             revert to product
                        # print(f'-W- Join {p}: growth too low for pace {pace} ({growth_tracker}/{target_growth} tuples); revert to product')
                        # return self.product_LFJ(p, target_tids)
                    # else:
                    #     return incomplete set of instances
                    #     # time2 = time.perf_counter()
                        # return res, p, time2-time1, samples


            # generate sample
            instance = self.sample_instance(distributions)
            x = ','.join(flatten(tuple(instance.values())))

            if x in sampled.keys():
                L = 0
            else:
                sampled[x] = 1
                # perform LFJ
                lists = [target_tids[attribute][value] for attribute, value in instance.items()]
                sets = map(set, lists)
                intersection_indices = sorted(list(set.intersection(*sets)))
                L = len(intersection_indices)

                if L < 2:
                    sampled_empty[x] = 1
                else:
                    res[x] = intersection_indices

            # update distributions. if some dist is empty break
            distributions = self.update_distributions(distributions, instance, total_sampled, L)
            if not distributions:
                break
            total_sampled += L

            # track growth
            # growth_tracker = growth_tracker + L
            # skip singletons

        res = dict(sorted(res.items(), key=lambda item: item[0]))
        res_data = {
            'frequencies': res,
            'num_samples': samples,
        }
        return res_data

    def sequential_sampling2(self, target_tids):
        """
        test1 - sample a, then sample b (conditional)
        O ( A x B log B )
        take B to be smaller TID
        """
        # assuming pairwise join
        A, B = target_tids.keys()
        TA = target_tids[A]
        TB = target_tids[B]

        #

        distributions = self.generate_distributions(target_tids)
        distA = distributions[A]
        distB = distributions[B]

        target_N = self.N * self._coverage
        total_sampled = 0

        res = {}
        sampled = {}
        sampled_empty = {}
        samples = 0

        # for instance_a in TA.keys():
        while total_sampled < target_N:
            # generate sample
            instance_a = self.sample_instance2(distA)
            tid_a = TA[instance_a]
            tot_a = 0
            while tot_a < len(tid_a):
                instance_b = self.sample_instance2(distB)
                samples = samples + 1

                tid_b = TB[instance_b]
                x = ','.join([instance_a, instance_b])

                if (x in sampled.keys()) or (x in sampled_empty.keys()):
                    continue
                else:
                    sampled[x] = 1
                    # perform LFJ
                    lists = [tid_a, tid_b]
                    sets = map(set, lists)
                    intersection_indices = sorted(list(set.intersection(*sets)))
                    L = len(intersection_indices)
                    if L:
                        tot_a += L
                        distB = self.update_distribution(distB, instance_b, L)

                        if L < 2:
                            sampled_empty[x] = 1
                        else:
                            res[x] = intersection_indices
            total_sampled += tot_a
              # update distributions. if some dist is empty break
            distA = self.update_distribution(distA, instance_a, tot_a)
            if not distA:
                break

        res = dict(sorted(res.items(), key=lambda item: item[0]))
        res_data = {
            'frequencies': res,
            'num_samples': samples,
        }
        return res_data

    def pairwise_sequential_sampling(self, target_tids):
        key1 = list(target_tids.keys())[0]
        first = target_tids.pop(key1)

        res_name = key1
        res = first
        while len(target_tids):
            # pop next element in target tids
            nkey = list(target_tids.keys())[0]
            next = target_tids.pop(nkey)

            next_targets = {
                res_name: res,
                nkey: next
            }
            res_name += ',' + nkey
            res = self.sequential_sampling(next_targets)['frequencies']

        res_data = {
            'frequencies': res
        }
        return res_data

    # ------------------------------------------------------------------
    # ENTROPY ----------------------------------------------------------
    # ------------------------------------------------------------------
    def load_active_domain(self, X):
        p = sorted(X)
        df = convert_to_numeric(pd.read_csv(self._path, usecols=p))
        domain = df.value_counts().index.to_flat_index()
        flattened = [','.join(x) for x in domain]
        return flattened

    def intersect(self, p):
        set_p = set(p)
        target_tids = {}
        for subset, tids in self._TIDs.items():
            xi = set.intersection(set_p, set(tids.keys()))
            if not len(xi):
                continue
            elif len(xi) == 1:
                item = list(xi)[0]
            else:
                item = self.tids_set_comparison(xi, tids)

            target_tids[item] = tids[item]

        return target_tids

    def tids_set_comparison(self, xset, tids):
        for k in tids.keys():
            if xset==set(k.split(',')):
                return k

    def get_product_set_size(self, target_tids):
        domain_sizes = []
        for x,tids in target_tids.items():
            domain_sizes.append(len(tids))

        res = 1
        for d in domain_sizes:
            res *= d

        return res

    def partial_entropy(self, data):
        """
        when sampling data, H_S represents partial entropy as S_AB \subseteq D_AB and H is over D_AB
        """
        sum = 0
        for x in data:
            sum -= x/self.N * np.log2(x/self.N)

        return sum

    def entropy_project(self, X):
        p = sorted(X)
        df = convert_to_numeric(pd.read_csv(self._path, usecols=p))
        instances = {}
        for t in range(self.N):

            # y = t._asdict()
            # tid = y.pop('Index')
            # instance = ','.join(y.values())
            # if instance in instances.keys():
            #     instances[instance] += 1
            # else:
            #     instances[instance] = 1
            instance = ','.join(df.iloc[t].values)
            if instance in instances.keys():
                instances[instance] += 1
            else:
                instances[instance] = 1

        dist = [x/self.N for x in instances.values()]
        res_data = {
            'H': entropy(dist, base=2),
            'joins': None
        }
        return res_data

    # def stratified_entropy(self, X, num_strata=10):
    #     """
    #     X - query
    #     pop_size - size of pupulation per stratum
    #     """
        # if len(X)==self._M:
        #     return np.log2(self.N)

        # p = sorted(X)
        # target_tids = self.intersect(p)

        # generate product set
        # product_set = self.cartesian_product(target_tids)
        # product_size = self.cp_size(target_tids)
        # stratify product set
        # num_populations = int(ps_size * population_ratio)
        # strata, num_strata = self.stratify(product_set, product_size, num_strata)

        # for every stratum, estimate entropy
        # Ls = []
        # accum = 0
        # for s in strata.keys():
        #     sampled_L = self.stratum_entropy(strata[s], target_tids, p) # sample single instance in stratum
        #     mus.append(mu)
        #     vars.append(variance)
        #     Ls.append(sampled_L)

            # take average length * set size
            # accum += sampled_L * len(strata[s]) * np.log2(sampled_L * len(strata[s]))
        #
        # return np.log2(self.N)-accum/(self.N * num_strata)

    # ------------------------------------------------------------------
    # STRATIFIED ENTROPY -----------------------------------------------
    # ------------------------------------------------------------------
    # def cp_size(self, target_tids):
    #     """
    #     calculate size of product set
    #     """
    #     lens = [len(target_tids[k]) for k in target_tids.keys()]
    #     size = 1
    #     for l in lens:
    #         size *= l
    #
    #     return size

    # def stratify(self, product_set, N, num_strata):
    #     """
    #     partition product set (of size N) to num_pops groups
    #     randomize to avoid bias
    #     """

        # strata = {k: [] for k in range(num_strata)}
        #
        # temp = []
        # fill in first sample size elements
        # try:
        #     for _ in range(N):
        #         temp.append(next(product_set))
        # except StopIteration:
        #     raise ValueError("-E- Sample larger than population")
        #
        # random.shuffle(temp)
        #
        # for i in range(N):
        #     strat_num = random.randint(0, num_strata-1)
        #     strata[strat_num].append(temp[i])  # at a decreasing rate, replace random items
        #
        # assert sum([len(strata[k]) for k in strata.keys()]) == N

        # remove empty populations
        # for k in list(strata.keys()):
        #     if not strata[k]:
        #         del strata[k]
        #
        # new_num_pops = len(strata)
        # return strata, new_num_pops

    # def stratum_entropy(self, stratum, target_tids, p):
    #     """
    #     stratum - population of instances
    #     """
    #     Ls = []
    #     mus = []
    #     vars = []

        # randomly select instance
        # instance = random.choice(stratum)

        # OBSOLETE: sample single instance to represent stratum rather than traversing it
        # for x in stratum:
        # lists = []
        # try:
        #     lists = [target_tids[attribute][value] for attribute, value in zip(target_tids.keys(), instance)]
        # except:
            # print(f'-E- {p}: failed to access TID for instance {x}')
            # pass
        #
        #     execute join
        # sets = map(set, lists)
        # intersection_indices = sorted(list(set.intersection(*sets)))
            #
        # if len(intersection_indices) > 1:
        #     return len(intersection_indices)
        #
        # S = sum(Ls)
        # normalized = [l/S for l in Ls]
        # stratum_H = entropy(normalized, base=2)
        #
        # mu = np.average(Ls)
        # variance = np.var(Ls)
        #
        # return mu, variance, stratum_H

    # ------------------------------------------------------------------
    #   CANDIDATES   ---------------------------------------------------
    # ------------------------------------------------------------------
    def cartesian_product(self, targets):
        """perform cartesian product on all target values"""
        to_multiply = [targets[x].keys() for x in targets.keys()]
        product = itertools.product(*to_multiply)
        return product

    # ------------------------------------------------------------------
    #   SEQUENTIAL SAMPLING   -------------------------------------------
    # ------------------------------------------------------------------
    def generate_distributions(self, target_tids):
        distributions = {}
        for attribute, instances in target_tids.items():
            d = {x: len(target_tids[attribute][x]) for x in target_tids[attribute].keys()}
            d = dict(sorted(d.items(), key=lambda item: item[1]))
            distributions[attribute] = d

        return distributions

    def sample_instance(self, distributions):
        res = {}
        if self.mode == 'mssj':
            for attribute in distributions.keys():
                res[attribute] = max(distributions[attribute], key=distributions[attribute].get)
        else:
            for attribute in distributions.keys():
                res[attribute] = random.choices(list(distributions[attribute].keys()), weights=distributions[attribute].values())[0]

        # sort output by keys to match p=sorted(X)
        res = dict(sorted(res.items(), key=lambda item: item[0]))
        return res


    def sample_instance2(self, distribution):
        return random.choices(list(distribution.keys()), weights=distribution.values())[0]

    def update_distributions(self, distributions, instance, total_sampled, L):
        for attribute, value in instance.items():
            distributions[attribute][value] -= L
            if distributions[attribute][value] == 0:
                del distributions[attribute][value]

                # empty distributions
                if not distributions[attribute]:
                    return None

        return distributions

    def update_distribution(self, distribution, instance, L):
        distribution[instance] -= L
        if distribution[instance] == 0:
            del distribution[instance]

        return distribution

    def compare_domains(self, original, sampled):
        shared = sorted(list(set(original).intersection(set(sampled))))
        not_shared = []
        for x in original:
            if x in shared:
                continue
            not_shared.append(x)

        return shared, not_shared

    def get_Q_dist(self, instances, X):
        # load data & flatten index
        p = sorted(X)
        df = convert_to_numeric(pd.read_csv(self._path, usecols=X))
        counts = df.value_counts()
        gesheft = [','.join(x) for x in counts.index.to_flat_index()]
        counts.index = gesheft
        Q = []
        for instance in instances:
            Q.append(counts[instance])

        QN = sum(Q)
        normalized = [x/QN for x in Q]
        return normalized

    def set_ps_params(self, coverage):
        self._coverage = coverage
        self._hybrid = False
        self._candidates = 'ps'

    # ------------------------------------------------------------------
    #   PARALLEL    ----------------------------------------------------
    # ------------------------------------------------------------------
    def init_parallel(self):

        # with ProcessPoolExecutor() as pool:
        #     pool.map(self.single_subset_tids, self._TIDs.keys(), self._TIDs.values())
        n_jobs = len(self._TIDs)
        Parallel(n_jobs=n_jobs)(delayed(self.single_subset_tids)(subset, tids) for subset, tids in self._TIDs.items())
        print('-I- Finished parallelized init')

    # ------------------------------------------------------------------
    #   API    ---------------------------------------------------------
    # ------------------------------------------------------------------
    def get_frequency(self, X):
        p = sorted(X)
        target_tids = self.intersect(p)
        res_data = {
            'joins': len(target_tids)
        }

        # empty cover when join on p is empty
        if not len(target_tids):
            print(f'-E- Join {p}: empty output')
            return None
        elif len(target_tids) == 1:
            key = list(target_tids.keys())[0]
            res_data['frequencies'] = target_tids[key]
            return res_data

        if self.mode == 'pli':
             res_data.update(self.PLI_join(target_tids)) # pairwise O(K*N^2)
        elif self.mode == 'product':
            res_data.update(self.product_LFJ(p, target_tids)) # not pairwise O(N^K)
        elif self.mode in ['ssj', 'mssj']:
            res_data.update(self.pairwise_sequential_sampling(target_tids))
        elif self.mode == 'cssj':
            res_data.update(self.sequential_sampling2(target_tids))
        else:
            raise('Code should not execute this')

        if res_data:
            return res_data
        else:
            print(f'-W- Join {p}: empty output; {p} is primary key')
            return None

    # def entropy_framework(self, X, coverage):
    #     # prior to sampling
    #     self.set_ps_params(coverage)
    #
    #     # sample
    #     frequencies, name, join_time, num_samples, empty_samples, product_set_size = self.get_frequency(X)
    #
    #
    #
    #
    #
    #     # analyze sample output
    #     lens = [len(l) for l in frequencies.values()]     # number of entries per sampled instance
    #     effective_N = sum(lens)                           # effective N covered
    #     effective_coverage = effective_N / self.N         # rho
    #     rho_bar = 1 - effective_coverage                  # rho bar
    #
    #     # Q_AB - distribution over non-sampled instances
    #     active_domain = self.load_active_domain(X)                 # D_AB
    #     sampled_domain = frequencies.keys()                        # S_AB
    #     shared, not_shared = self.compare_domains(active_domain, sampled_domain) # not_shared = Q_AB
    #     Q = self.get_Q_dist(not_shared, X)
    #     H_Q = entropy(Q, base=2)
    #
    #     # S_AB - set of sampled instances (S_AB \subseteq D_AB)
    #     S_dist = [l/effective_N for l in lens]
    #     H_S = self.partial_entropy(lens)
    #     H_S_normalized = entropy(S_dist, base=2)
    #
    #     # error terms
    #     E_exact =  rho_bar * (H_Q-np.log2(rho_bar))            # this should add up to H along with H_S
    #     E1 = rho_bar * (np.log2(len(not_shared)-np.log2(rho_bar)))
    #     E2 = rho_bar * np.log2(self.N)
    #     E3 = rho_bar * (product_set_size-len(sampled_domain)-len(empty_samples)-np.log2(rho_bar))
    #
    #     data = {
    #         'partial_H_sampled':H_S,
    #         'H_sampled_normalized':H_S_normalized,
    #         'E_exact':E_exact,
    #         'E1':E1,
    #         'E2':E2,
    #         'E3':E3
    #     }
    #     stats = {
    #         'times':join_time,
    #         'num_samples':num_samples,
    #     }
    #     return data, stats

    def get_attributes(self):
        return self._omega.copy()

    def entropy(self, X):
        # access disk
        if self.mode == 'project':
            return self.entropy_project(X)

        # assuming normalized input
        if len(X) == self.M:
            # O(M log M); only if previous condition met
            if sorted(X) == sorted(self._omega):
                return np.log2(self.N)

        # main workhorse
        res_data = self.get_frequency(X)
        frequencies: dict = res_data['frequencies']

        # evaluate distribution based on frequencies
        if not frequencies:
            return

        # generate distribution; evaluate H
        lens = [len(l) for l in frequencies.values()]
        effective_N = sum(lens) # dropping singletons causes effective_N < N
        dist = [l / effective_N for l in lens]
        H_S = entropy(dist, base=2)
        res_data['dist'] = dist

        res_data['H'] = H_S
        return res_data

def R_debug():
    csv1 = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\School_Results\\school_results.csv"
    name1 = 'school_results'

    R = Relation(path=csv1, name=name1, mode='cssj')
    R.entropy(['SCHOOL', 'MALEX'])

if __name__ == '__main__':
    R_debug()
