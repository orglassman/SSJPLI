# import itertools
# import pandas as pd
#
# from common import mergesort, convert_to_numeric
# from sparse_matrix import SparseMatrix
#
#
# class leapfrog_join():
#     def __init__(self, iterators, df, candidates='cartesian'):
#         """
#         iterators - list of iterators
#         df        - if 'project' is selected, pass df reference
#         candidates - how to determine candidates for join:
#             1. cartesian    - perform cartesian product on iterators (all possible instances)
#             2. project      - project onto original relation and fetch unique values
#             3. is           - employ importance sampling
#         """
#
#         attributes = [it.get_attributes() for it in iterators]
#         self._attributes = attributes
#         self._iterators = dict(zip(attributes, iterators))
#         self._df = df
#         self._candidates = candidates
#         self._number_joins = len(iterators)
#
#         self._num_atts = len(attributes)
#         self._num_entries = len(df)
#
#
#     def join(self, merge_attributes=False, return_keys=False, remove_singletons=True):
#         """
#         perform leapfrog join:
#         1. multiply iterators (get all possible instances)
#         2. do leapfrog for every possible instance (in separate method)
#
#         merge_attributes - if True, squeeze attributes into single column. e.g. ['A', 'B'] -> ['A,B']
#         return_keys - if True, append list of keys as well as frequency for every instance
#         i.e.
#         return not only CNT but also TID for joined table
#         """
#         init empty result df
        # attributes = self._attributes
        # attributes_sorted = sorted(list(itertools.chain.from_iterable([s.split(',') for s in attributes])))
        # if merge_attributes:
        #     merged = ','.join(attributes)
        #     columns = [merged, 'COUNT']
        # else:
        #     columns = attributes + ['COUNT']
        #
        # if return_keys:
        #     columns.append('TID')
        #
        # res_entries = []
        #
        # get all possible instances for given attributes (possible bottleneck)
        # all_instances = self.get_candidates()
        #
        # try:
        #     while True:
        #         current_instance = next(all_instances)
        #
                # at this point we know that:
                # instance is sorted according to self._attributes
                #
                # make sure each iterator points to correct instance
                # self.sort_iterators(current_instance)
                #
                # main work horse - perform leapfrog for every instance
                # res = self.join_single_instance(return_keys=return_keys)
                #
                # if merge_attributes:
                #     frequency = res[0]
                #     tids = res[1]
                #     entry = [','.join(list(current_instance)), frequency, tids]
                # else:
                #     frequency = res
                #     entry = list(current_instance) + [frequency]
                #
                # if remove_singletons and frequency <= 1:
                #     continue
                #
                # res_entries.append(entry)
        #
        # except StopIteration:
        #     print(f'-I- For attributes {attributes}, number of joins {self._number_joins}')
        #     pass
        # except KeyError:
        #     print('-E- Key error: lal')
        #
        # res = pd.DataFrame(res_entries, columns=columns)
        # return res
    #
    # def join_single_instance(self, return_keys):
    #     """
    #     row: dictionary where
    #     keys are ['Index', '_1', '_2',...]
    #     items are [index, instance for attribute 0, instance for attribute 1,...]
    #
    #     return_keys - if True, return not only CNT but also TID for instance
    #     """
    #
    #     simply count how many times current vector appears in joined relation
        # lists = [it._current_list for it in self._iterators.values()]
        # sets = [set(l) for l in lists]
        # keys = set.intersection(*sets)
        #
        # freq = len(keys)
        # if return_keys:
        #     return [freq, list(keys)]
        #
        # return freq
        #
        ##
          #   obsolete code
        ##
        #
        # freq = 0
        # attributes = [it.get_attributes() for it in self._iterators.values()]
        # keys = [it.get_key() for it in self._iterators.values()]
        #
        # min_key = keys[0]
        # max_key = keys[-1]
        #
        #
        # while True:
        #     if self.finished_any_list():
        #         print(f'-I- Finished single list, leapfrog finished for instance {instance}')
        #         break
        #     while min_key < max_key:
        #         for it in self._iterators.values():
        #             it.seek(max_key)
        #             current_key = it.get_key()
        #             if current_key > max_key:
        #                 max_key = current_key
        #
                # once seek operation complete, check if need to join
                # if self.equality():
                #     freq += 1
                #     l = list(self._iterators.values())
                #     first_iterator = l[0]
                #     last_iterator = l[-1]

                    # next(last_iterator)
                    # update min/max
                    # min_key = first_iterator.get_key()
                    # max_key = last_iterator.get_key()
                    # break
        #
        # return freq
    #
    # def get_all_attributes(self):
    #     return list(self._attributes)
    #
    # ------------------------------------------------------------------
    # CANDIDATES GENERATION --------------------------------------------
    # ------------------------------------------------------------------
    #
    # def get_candidates(self):
    #     if self._candidates == 'cartesian':
    #         return self.multiply_iterators()
    #     elif self._candidates == 'project':
    #         return self.project_iterators()
    #     elif self._candidates == 'is':
    #         return self.importance_sample()
    #     else:
    #         TODO: create exception class for this case and handle
            # raise
    #
    # def multiply_iterators(self):
    #     """
    #     multiply all iterators to get all possible instances
    #     this is time consuming and could potentially be the bottleneck of the entire operation
    #     """
    #     get list of pandas columns
        # instances = [it.get_all_instances() for it in self._iterators.values()]
        #
        # generate cartesian product table
        # product = itertools.product(*instances)
        # product = cartesian_product_multi(*instances)
        #
        # return product
    #
    # def project_iterators(self):
    #     """load original df and do projection to obtain only relevant tuples"""
    #     cols = list(itertools.chain.from_iterable([x.split(',') for x in self._attributes]))
    #     slice = self._df[cols]
    #
    #     uniqify
        # unique = slice.value_counts().reset_index()[cols]
        #
        # squeeze columns together
        # for att in self._attributes:
        #     if len(att) == 1:
        #         continue
        #
        #     split_atts = att.split(',')
        #     unique[att] = unique[split_atts].aggregate(','.join, axis=1)
        #     unique.drop(columns=split_atts, inplace=True)
        #
        # sort columns
        # unique = unique[sorted(unique.columns)]
        #
        # projection = unique.itertuples(index=False, name=None)
        # projection = slice.value_counts().reset_index()[cols].itertuples(index=False)
        # return projection
    #
    # def importance_sample(self):
    #     """
    #     convert TIDs to sparse matrices, multiply and employ IS
    #     """
    #
    #     dimensions of sparse matrices
        # matrices = self.gen_sparse_matrices()
        #
        #
        # IS = ImportanceSampler(matrices=matrices, delta=0.1, K=2)
        # print('hello world')
    #
    # def gen_sparse_matrices(self):
    #     res = []
    #     for it in self._iterators.values():
    #         res.append(SparseMatrix(it.get_all_tids()))
    #
    #     return res
    #
    # def sort_iterators(self, instance):
    #     """
    #     1. make sure each iterator points to correct instance
    #     2. perform merge sort on keys
    #     """
    #     i = 0
    #     attributes = self.get_all_attributes()
    #
    #     for att in attributes:
    #         att_instance = instance[i]
    #
    #         fetch correct list of keys - current instance
            # self._iterators[att].set_list(att_instance)
            # i += 1
        #
        #
        # self._iterators = mergesort(self._iterators)
    #
    # def finished_any_list(self):
    #     for it in self._iterators.values():
    #         if it.is_end_list():
    #             return True
    #
    #     return False
    #
    # def equality(self):
    #     keys = [it.get_key() for it in self._iterators.values()]
    #     return keys.count(keys[0]) == len(keys)
    #
    # def gen_output_vector(self, row_dict):
    #     items = [str(item) for idx, item in row_dict.items()]
    #     return ','.join(items)
    #
    # def get_keys(self):
    #     return [it.get_key() for it in self._iterators]
    #
    # def tune_iterators(self, row):
    #     """
    #     row - hash with
    #     key = table number
    #     value = instance
    #     """
    #     it_num = 0
    #     for table_num, instance in row.items():
    #
    #         for each iterator, make sure we're pointing to the same instance
            # current_iterator = self._iterators[it_num]
            # while instance != current_iterator.get_current_instance():
            #     current_iterator.next_instance()
            #
            # it_num += 1