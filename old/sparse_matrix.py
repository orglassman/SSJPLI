import numpy as np
import scipy.sparse as ss


class SparseMatrix():
    def __init__(self, attributes, tids):
        """
        tids - dict where
            key = instance
            val = TIDs in R
        """
        self._attributes = attributes
        instances = list(range(len(tids)))

        # for fast referencing row -> instance and vice versa
        # self._row2instance = dict(zip(rows, tids.keys()))
        # self._instance2row = dict(zip(tids.keys(), rows))


        instance_tids = tids.values()
        max_values = [x[-1] for x in instance_tids]

        N = max(max_values) + 1
        M = len(instances)

        A = np.zeros([N, M])

        for instance, tid in enumerate(instance_tids):
            for x in tid:
                num_col = instance
                num_row = x
                A[num_row][num_col] = 1

        # convert to sparse matrix representation
        self._N = N
        self._M = M
        # self._matrix = ss.csr_matrix(A)
        # self._matrix = ss.csc_matrix(A)
        self._matrix = ss.coo_matrix(A)
    def get_shape(self):
        return [self._N, self._M]