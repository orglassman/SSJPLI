import numpy as np
import scipy.sparse as ss

from sparse_matrix import SparseMatrix


class CohenLewisSampler:
    def __init__(self, matrices, delta=0.05, K=1):
        """
        A,B - two sparse matrices in CSR
        delta - probability parameter
        K - importance factor
        """
        self._delta = delta
        self._K = K

        A = matrices[0]
        AN, AM = A.get_shape()
        # BN, BM = B.get_shape()
        # assert(AN == BN)
        self._N = AN
        # self._AM = AM
        # self._BM = BM
        #
        # self._A = A
        # self._B = B
        self._Ms = matrices

    def single_sample(self):
        # randomly select row
        row = np.random.randint(0, self._N)

        # once we select row, we can sample from A,B
        instances = []
        for M in self._Ms:
            instance = M._matrix.getrow(row).nonzero()[1][0]
            instances.append(instance)

        # A_instance = self._A._matrix.getrow(row).nonzero()[1][0]
        # B_instance = self._B._matrix.getrow(row).nonzero()[1][0]

        # return (A_instance, B_instance)
        return tuple(instances)

    def sample(self):

        # determine number of samples
        gamma = self._N
        X = gamma/self._K
        Y = np.log(gamma/(self._K * self._delta))
        num_samples = int(np.ceil(X * Y))

        instances = {}

        # num_samples = int(self._N * np.log(self._N))

        for i in range(num_samples):
            instances[self.single_sample()] = 1

        return sorted(list(instances.keys()))