"""
1. given CSV, and partition parameter l, partition table into ceil(K/l) subtables (c'tor)
2. given X \subseteq \Omega, locate relevant tables (get_freqneucy)
3. perform leapfrog join on TID tables by iterating TID entries for every possible x \in X (get_frequency)
"""
from math import ceil

import time
import more_itertools as mit
import numpy as np
import pandas as pd

from common import convert_to_numeric, swap, intersection
from leapfrog_iterator import leapfrog_iterator
from leapfrog_join import leapfrog_join


class relation:
    def __init__(self, path, name, num_atts, l=2):
        """
        path  - input CSV
        name  - arbitrary
        l     - partition parameter. partition K attributes into ceil(K/l) groups
        """
        self._path = path
        self._name = name
        self._l = l
        self._num_atts = num_atts
        self._num_subtables = ceil(self._num_atts / self._l)

        self.check_num_atts()  # check that l < K. otherwise set l=1
        self.load_data()  # load df

        # for each attribute determine its subset (data structure H in self)
        self.determine_subsets()

        # partition the data into subsets
        self.partition_data()

        # for each subtable - build CNT and TID subtables
        self.build_freq_tables()
        pass
    @classmethod
    def from_df(cls, df):
        path = df
        name = 'FROM_DF'
        l = 1
        num_atts = len(df.columns)
        return cls(path, name, num_atts, l)

    # ------------------------------------------------------------------
    #   INIT    --------------------------------------------------------
    # ------------------------------------------------------------------

    def check_num_atts(self):
        """
        check that l < num_atts otherwise set it to 1
        """
        if self._l >= self._num_atts:
            print(
                f'-W- Relation {self._name}: Partition parameter l={self._l} is larger than number of attributes. Resetting to 1')
            self._l = 1

    def load_data(self):
        """
        1. load CSV
        2. convert all entries to numeric values
        3. rename header to "0,1,..."
        """

        if self._name == 'FROM_DF':
            # when generating relation from DF instance, _path now contains the df
            # perform swap - load df onto self._df and remove path
            self._df = self._path
            # self._df = convert_to_numeric(self._path)
            self._path = ''

        else:
            df = convert_to_numeric(pd.read_csv(self._path))
            self._df = df

        self._attributes = self._df.columns

    def determine_subsets(self):
        """
        for each attribute, determine which subset will contain it
        store in hash H
        """
        H = {}
        curr_subset = 0
        for i in range(self._num_atts):
            curr_attribute = self._df.columns[i]

            # start new subset
            if curr_subset not in H.keys():
                H[curr_subset] = [curr_attribute]
                continue

            # populate existing subset
            H[curr_subset].append(curr_attribute)

            if len(H[curr_subset]) == self._l:
                curr_subset = curr_subset + 1

        self._H = H

    def partition_data(self):
        """
        based on H, split data
        """
        subtables = {}
        for subtable_name in self._H.keys():
            attributes = self._H[subtable_name]
            subtable = self._df[attributes]
            subtables[subtable_name] = subtable

        self._subtables = subtables

        # remove from memory
        del self._df

    def build_freq_tables(self):
        """
        for each subtable (K/l attributes)
        build CNT and TID tables

        need to build powerset for each subtable (set of all subsets)
        e.g.
        [A,B,C] -> [A, B, C, AB, AC, BC]
        for each subset, calculate CNT and TID tables
        """
        cnt_structure = {}
        tid_structure = {}
        for subtable_name, subtable in self._subtables.items():
            attributes = subtable.columns
            K_l = len(attributes)  # number of attributes in subtable

            # generate powerset (lexicographically sorted elements)
            powerset = [sorted(x) for x in list(mit.powerset(attributes))]
            # remove the empty set & set itself
            powerset.pop(0)

            # prevent popping off the top for single attributes
            if K_l > 1:
                powerset.pop(-1)
            cnt_tables, tid_tables = self.gen_freq_tables(powerset, subtable_name)

            # save
            cnt_structure[subtable_name] = cnt_tables
            tid_structure[subtable_name] = tid_tables

        self._cnt_structure = cnt_structure
        self._tid_structure = tid_structure

    def gen_freq_tables(self, powerset, subtable_name):
        """
        for every element in powerset,
        generate CNT and TID frequency tables
        """
        current_table = self._subtables[subtable_name]

        CNTs = {}
        TIDs = {}
        for subset in powerset:
            # attributes = [int(x) for x in list(subset)]   # get INTEGER list of attributes
            attributes = subset
            name = ','.join(attributes)
            # take relevant data slice
            df_slice = current_table[attributes]

            CNT = self.gen_cnt_table(df_slice, attributes, name)
            TID = self.gen_tid_table(df_slice.copy(), attributes, CNT, name)

            # append
            key = tuple(subset)
            CNTs[key] = CNT
            TIDs[key] = TID

        return CNTs, TIDs

    def gen_cnt_table(self, df, attributes, name):
        """
        for each possible tuple containing attributes,
        count number of occurrences in data

        remove singletons
        """
        # generate CNT table
        CNT = df.value_counts().reset_index()

        # for subsets containing more than single attribute, merge columns upon resetting indices
        subset_size = len(attributes)

        # for more than single attribute, merge values to single tuple
        # e.g. 1, 2 merged to (1,2) (2 columns -> 1 column)
        CNT[name] = CNT[CNT.columns[0:subset_size]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1)

        # from here on in - data is STRING

        # when dealing with more than single attribute, remove old single columns
        if subset_size > 1:
            CNT.drop(attributes, axis=1, inplace=True)
            # need to swap two columns
            new_columns = swap(list(CNT.columns), 0, 1)
            CNT = CNT[new_columns]

        # rename columns
        CNT.columns = [name, 'COUNT']

        # drop singletons
        CNT = CNT[CNT['COUNT'] > 1]

        return CNT

    def gen_tid_table(self, df, attributes, CNT, name):
        """
        based on CNT table, construct TID table where each tuple has list of indices in original table
        """
        # initialize empty dataframe
        TID = pd.DataFrame()
        TID[name] = CNT[name]
        # create merged column for original data
        # df['merge'] = df[attributes].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
        merge = df.loc[:, attributes].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
        df['merge'] = merge
        # for each tuple in TID, find corresponding indices in data
        TID['IDX'] = TID[name].apply(lambda row: df[df['merge'] == row].index.values)

        return TID

    # ------------------------------------------------------------------
    #   END INIT    ----------------------------------------------------
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #   GET ------------------------------------------------------------
    # ------------------------------------------------------------------
    def get_attributes(self):
        return list(self._attributes)

    def get_frequency(self, X):
        """
        for given set of attributes X (list), calculate frequency of every tuple

        returns pd.DataFrame
        """

        # fetch CNT TID maps for relevant attributes
        cnt_tables, tid_tables, possible_values = self.intersect_X(X)

        # get list of all tid tables
        all_tables = [table[1] for table in list(tid_tables.items())]

        # init iterator for each table
        iterators = [leapfrog_iterator(table) for table in all_tables]

        LFJ = leapfrog_join(iterators)

        output = LFJ.join()
        return output

    def get_subset(self, indices):
        """
        return subset of relation - only given indices
        """
        pass

    def intersect_X(self, X):
        """
        for every attribute in X, intersect it with R's sub relations.
        this is to determine which tables to retrieve - CNT & TID

        e.g. Nursery and l=3, we have [A,B,C],[D,E,F],[G,H,I]
        for get_frequency([A,B,D]) we need CNT and TID for [A,B] and [D]
        """
        cnt_tables = {}
        tid_tables = {}
        possible_values = {}

        for subset_number in self._H:
            curr_intersection = tuple(sorted(intersection(X, self._H[subset_number])))

            # intersection not empty - fetch relevant CNT and TID
            if curr_intersection:
                cnt_table = self._cnt_structure[subset_number][curr_intersection]
                tid_table = self._tid_structure[subset_number][curr_intersection]

                cnt_tables[curr_intersection] = cnt_table
                tid_tables[curr_intersection] = tid_table
                possible_values[curr_intersection] = tid_table[tid_table.columns[0]].values

        return cnt_tables, tid_tables, possible_values

    # ------------------------------------------------------------------
    #   END GET ------------------------------------------------------------
    # ------------------------------------------------------------------


def frequency_debug():
    """
    create single relation and get frequency for every tuple
    X - subset of Omega

    determine all possible values for x \in X
    and calculate frequency for every value
    """
    nursery_path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    nursery_name = 'Nursery'
    nursery_num_atts = 9
    nursery_l = 3

    credit_path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Credit\\credit.csv"
    credit_name = 'Credit'
    credit_num_atts = 16
    credit_l = 8

    # nursery
    R = relation(path=nursery_path, name=nursery_name, num_atts=nursery_num_atts, l=nursery_l)
    X = ['A', 'B', 'D']
    Y = ['A', 'C']
    Z = ['C', 'E', 'H']  # corresponds to [A,B,D]
    freq_ABD = R.get_frequency(X)
    freq_AC = R.get_frequency(Y)
    freq_CEH = R.get_frequency(Z)

    # credit
    S = relation(path=credit_path, name=credit_name, num_atts=credit_num_atts, l=credit_l)
    X = ['A0', 'A3', 'A5', 'A10', 'A12']
    freq_0351012 = S.get_frequency(X)

    print('-I- Finished')


def RST_debug():
    csv1 = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\LeapfrogPYJoin\\test\\R.csv"  # A,B
    csv2 = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\LeapfrogPYJoin\\test\\S.csv"  # B,C
    csv3 = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\LeapfrogPYJoin\\test\\T.csv"  # A,C
    name1 = 'R'
    name2 = 'S'
    name3 = 'T'

    R = relation(path=csv1, name=name1, num_atts=2, l=1)
    S = relation(path=csv2, name=name2, num_atts=2, l=1)
    T = relation(path=csv3, name=name3, num_atts=2, l=1)

    joined = R.join(S)
    joined2 = R.join(S).join(T)

    print("hello world!")


if __name__ == '__main__':
    frequency_debug()
