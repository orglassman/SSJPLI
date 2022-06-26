import itertools

import pandas as pd
import heapq as heap
from common import cartesian_product_multi, convert_attribute_num_str, mergesort
from minheap import minheap


class leapfrog_join():
    def __init__(self, iterators):
        """
        take list of iterators
        """
        # self._iterators = iterators

        attributes = [it.get_attributes() for it in iterators]
        self._iterators = dict(zip(attributes, iterators))
        self._attributes = attributes

    def join(self):
        """
        perform leapfrog join:
        1. multiply iterators (get all possible instances)
        2. do leapfrog for every possible instance (in separate method)
        """
        # init empty result df
        attributes = self._attributes
        columns = attributes + ['frequency']
        res_entries = []

        # get all possible instances for given attributes (possible bottleneck)
        all_instances = self.multiply_iterators()

        try:
            while True:
                current_instance = next(all_instances)

                # at this point we know that:
                # instance is sorted according to self._attributes

                # make sure each iterator points to correct instance
                self.sort_iterators(current_instance)

                # main work horse - perform leapfrog for every instance
                freq = self.join_single_instance(current_instance)

                entry = list(current_instance) + [freq]
                res_entries.append(entry)

        except StopIteration:
            print(f'-I- Finished iterating over all instances for attributes {attributes}')

        res = pd.DataFrame(res_entries, columns=columns)
        return res

    def join_single_instance(self, instance):
        """
        row: dictionary where
        keys are ['Index', '_1', '_2',...]
        items are [index, instance for attribute 0, instance for attribute 1,...]
        """

        # simply count how many times current vector appears in joined relation
        lists = [it._current_list for it in self._iterators.values()]
        sets = [set(l) for l in lists]
        keys = set.intersection(*sets)

        freq = len(keys)
        return freq

        ###
        #   #   obsolete code
        ###

        # freq = 0
        # attributes = [it.get_attributes() for it in self._iterators.values()]
        # keys = [it.get_key() for it in self._iterators.values()]
        #
        # min_key = keys[0]
        # max_key = keys[-1]


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
        #         # once seek operation complete, check if need to join
        #         if self.equality():
        #             freq += 1
        #             l = list(self._iterators.values())
        #             first_iterator = l[0]
        #             last_iterator = l[-1]
        #
        #             next(last_iterator)
        #             # update min/max
        #             min_key = first_iterator.get_key()
        #             max_key = last_iterator.get_key()
        #             break

        return freq

    def get_all_attributes(self):
        return list(self._attributes)

    def multiply_iterators(self):
        """
        when performing join, multiply all iterators to get all possible instances
        this is time consuming and could potentially be the bottleneck of the entire operation
        """
        # get list of pandas columns
        instances = [it.get_all_instances() for it in self._iterators.values()]

        # generate cartesian product table
        product = itertools.product(*instances)
        # product = cartesian_product_multi(*instances)

        return product

    def sort_iterators(self, instance):
        """
        1. make sure each iterator points to correct instance
        2. perform merge sort on keys
        """
        i = 0
        attributes = self.get_all_attributes()
        for att in attributes:
            att_instance = instance[i]

            # fetch correct list of keys - current instance
            self._iterators[att].set_list(att_instance)
            i += 1


        self._iterators = mergesort(self._iterators)

    def finished_any_list(self):
        for it in self._iterators.values():
            if it.is_end_list():
                return True

        return False

    def equality(self):
        keys = [it.get_key() for it in self._iterators.values()]
        return keys.count(keys[0]) == len(keys)

    def gen_output_vector(self, row_dict):
        items = [str(item) for idx, item in row_dict.items()]
        return ','.join(items)

    def get_keys(self):
        return [it.get_key() for it in self._iterators]

    def tune_iterators(self, row):
        """
        row - hash with
        key = table number
        value = instance
        """
        it_num = 0
        for table_num, instance in row.items():

            # for each iterator, make sure we're pointing to the same instance
            current_iterator = self._iterators[it_num]
            while instance != current_iterator.get_current_instance():
                current_iterator.next_instance()

            it_num += 1