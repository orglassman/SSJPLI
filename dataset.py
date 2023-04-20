import itertools

import numpy as np
import pandas as pd


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

    def effective_alpha(self):
        return len(self.df['A'].value_counts())

    def effective_beta(self):
        return len(self.df['B'].value_counts())

    def drop(self, k):
        """randomly discard k entries"""

        if k <= 0 or k > self.get_N() or type(k) != int:
            raise Exception(f'The number of entries to remove must be a positive integer between [1, {self.get_N()}]')
        if self.dropped:
            raise Exception(f'Current data set already reduced. Cannot drop entries')

        indices = np.random.choice(self.df.index, k, replace=False)
        new_df = self.df.drop(indices, axis=0).reset_index(drop=True)
        self.df = new_df
        self.dropped = True     # can only drop once

        # compute I(A;B)
        self.I = self.get_I()

    def reset(self):
        self.__init__(self.alpha, self.beta)

    def HAB(self):
        PAB = self.df.value_counts(normalize=True).to_dict()
        res = 0
        for v in PAB.values():
            res -= v * np.log2(v)

        return res

    def HA(self):
        PA = self.df['A'].value_counts(normalize=True).to_dict()
        res = 0
        for v in PA.values():
            res -= v * np.log2(v)

        return res

    def HB(self):
        PB = self.df['B'].value_counts(normalize=True).to_dict()
        res = 0
        for v in PB.values():
            res -= v * np.log2(v)

        return res

    def get_I(self):
        return self.HA() + self.HB() - self.HAB()

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

def data_generator_main():
    dataset = DataSet(alpha=5, beta=10)
    dataset.drop(k=15)
    dataset.get_I()
    print('hello')

if __name__ == '__main__':
    data_generator_main()