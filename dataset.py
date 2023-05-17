import itertools

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

    def drop(self, k):
        """randomly discard k entries"""

        if k <= 0 or k > self.get_N() or type(k) != int:
            raise Exception(f'The number of entries to remove must be a positive integer between [1, {self.get_N()}]')
        if self.dropped:
            raise Exception(f'Current data set already reduced. Cannot drop entries')
        p = self.gen_random_weights()
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


class RealDataSet(DataSet):
    def __init__(self, path, columns):
        super().__init__()
        self.df = pd.read_csv(path, usecols=columns)

        # column mappping
        self._orig_cols = {original: new for original, new in zip(columns, ['A', 'B'])}

        self.manipulate_data()
        self.print_init()

    def manipulate_data(self):
        self.df.columns = ['A', 'B']
        self.alpha = self.df['A'].value_counts().shape[0]
        self.beta = self.df['B'].value_counts().shape[0]

    def print_init(self):
        print('-I- New real data instance initialized')
        print(f'-I- Total entries: {self.df.shape[0]}')
        print(f'-I- Column mapping: {self._orig_cols}')
        print(f'-I- Cardinalities: A:{self.alpha}, B:{self.beta}')


if __name__ == '__main__':
    dataset = DataSet(alpha=5, beta=10)
    dataset.drop(k=15)
    dataset.get_I()
    print('hello')
