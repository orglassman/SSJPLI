import itertools
import string

import numpy as np
import pandas as pd
from scipy.stats import norm


class DataSet:
    def __init__(self, alpha=10, beta=5):
        """create product set"""
        self.alpha = alpha
        self.beta = beta

        product = itertools.product(range(alpha), range(beta))
        self.df = pd.DataFrame(product)

        self.df.columns = ['A', 'B']
        self.dropped = False

    def get_N(self):
        return self.df.shape[0]

    def get_M(self):
        return self.df.shape[1]

    def get_omega(self):
        return self.df.columns.tolist()

    def drop(self, k):
        """randomly discard k entries"""

        if k <= 0 or k > self.get_N() or type(k) != int:
            raise Exception(f'The number of entries to remove must be a positive integer between [1, {self.get_N()}]')
        if self.dropped:
            raise Exception(f'Current data set already reduced. Cannot drop entries')
        p = self.gen_random_weights()
        indices = None
        try:
            indices = np.random.choice(self.df.index, k, replace=False, p=p)
        except:
            print('hello')
        new_df = self.df.drop(indices, axis=0).reset_index(drop=True)
        self.df = new_df
        self.dropped = True  # can only drop once

        # compute I(A;B)
        self.I = self.get_I()

    def reset(self):
        self.__init__(self.alpha, self.beta)

    def gen_random_weights(self):
        support = self.df.index
        mean = int(len(support) / 2)
        pdfs = [norm.pdf(x, loc=mean) for x in support]
        Z = sum(pdfs)
        prob = [x / Z for x in pdfs]
        return prob

    def random_query(self, size):
        omega = self.get_omega()
        return np.random.choice(omega, size=size, replace=False)


    def H(self, X=None):
        if X is None:
            X = self.get_omega()
        try:
            P = self.df[X].value_counts(normalize=True).to_dict()
        except:
            print('hello')
        res = 0
        for v in P.values():
            res -= v * np.log2(v)

        return res

    def get_I(self, A, B):
        return self.H(A) + self.H(B) - self.H()

    def clone(self):
        new = DataSet(self.alpha, self.beta)
        new.df = self.df.copy(deep=True)
        new.dropped = self.dropped
        new.I = self.I
        return new

    def __eq__(self, other):
        return self.I == other.I

    def __lt__(self, other):
        return self.I < other.I

    def __gt__(self, other):
        return self.I > other.I

    def __le__(self, other):
        return self.I <= other.I

    def __ge__(self, other):
        return self.I >= other.I


class RealDataSet(DataSet):
    def __init__(self, path, columns=None):
        super().__init__()
        if not bool(columns):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.read_csv(path, usecols=columns)

        # column mappping
        self.rename_cols()
        self.print_init()

    def rename_cols(self):
        orig_cols = self.df.columns
        num_cols = len(orig_cols)
        new_cols = [f'A{i}' for i in range(num_cols)]
        self._orig_cols = {original: new for original, new in zip(orig_cols, new_cols)}
        self.df.columns = new_cols

    def print_init(self):
        print('-I- New real data instance initialized')
        print(f'-I- Total entries: {self.df.shape[0]}')
        print(f'-I- Column mapping: {self._orig_cols}')
        print('-I- Cardinalities:')
        for col in self.df.columns:
            print(f'-I- Column {col}: cardinality {self.df[col].value_counts().shape[0]}')

if __name__ == '__main__':
    dataset = DataSet(alpha=5, beta=10)
    dataset.drop(k=15)
    dataset.get_I()
    print('hello')
