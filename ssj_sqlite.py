import csv
import sqlite3

import numpy as np

import sql_utils as squ


class SSJSQL:
    def __init__(self, path, name):
        self.name = name
        print(f'-I- Initializing SSJSQL instance {self.name}')

        self.connection = sqlite3.connect(':memory:')
        self.cursor = self.connection.cursor()
        with open(path, 'r') as file:
            self.read_metadata(file)
            self.read_PLIs(file)

        self.init_db()
        print('-I- Init successful')

    def read_metadata(self, file):
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # read header

        self.N = sum(1 for row in csv_reader)
        self.M = len(header)
        self.omega = header.copy()

        print(f'-I- Number of records: {self.N}')
        print(f'-I- Number of attributes: {self.M}')

        # reset seek
        file.seek(0)

    def read_PLIs(self, file):
        print('-I- Reading single attribute PLIs')
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # read header

        PLIs = {x: {} for x in header}
        # create single attribute tids
        for idx, row in enumerate(csv_reader):
            for att_num, token in enumerate(row):
                if token in PLIs[header[att_num]].keys():
                    PLIs[header[att_num]][token].append(idx + 1)
                else:
                    PLIs[header[att_num]][token] = [idx + 1]

        self.weights = {k: {j: len(j) for j in PLIs[k].keys()} for k in PLIs.keys()}
        self.cardinalities = {k: len(v) for k, v in PLIs.items()}
        self.PLIs = PLIs
        file.seek(0)

    def init_db(self):
        print('-I- Initializing in-memory DB')
        for attribute, PLI in self.PLIs.items():
            header = [attribute, 'TID']
            name = 'T_' + attribute

            # unravel TID
            data = []
            for instance, tid in PLI.items():
                [data.append([instance, x]) for x in tid]

            squ.create_table(self.cursor, table_name=name, columns=header)
            squ.insert_data(self.cursor, table_name=name, columns=header, data=data)

        # commit changes
        self.connection.commit()

        # free memory
        del self.PLIs

    def entropy_join(self, X):
        P = squ.get_distribution_from_join(self.cursor, X)

        res = 0
        for p in P.values():
            res -= p * np.log2(p)

        return res

    def get_ps_size(self, X):
        attributes = list(X)
        res = 1
        for a in attributes:
            res *= self.cardinalities[a]

        return res

if __name__ == '__main__':
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"

    ssj_sampler = SSJSQL(path=path, name='NURSERY')


    ssj_sampler.entropy_join('ABC')


    print('hello')