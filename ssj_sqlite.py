import csv
import sqlite3
import sys
import traceback

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

        # remove singletons
        for attribute in PLIs.keys():
            for letter in PLIs[attribute].keys():
                if len(PLIs[attribute][letter]) == 1:
                    del PLIs[attribute][letter]

        self.weights = {k: {j: len(j) for j in PLIs[k].keys()} for k in PLIs.keys()}
        self.cardinalities = {k: len(v) for k, v in PLIs.items()}
        self.PLIs = PLIs
        file.seek(0)

    def init_db(self):
        print('-I- Initializing in-memory DB')

        for attribute, PLI in self.PLIs.items():
            tid_name = 'TID_' + attribute
            cnt_name = 'CNT_' + attribute

            tid_header = [attribute.lower(), 'tid']
            cnt_header = [attribute.lower(), 'cnt']

            # unravel TID
            tid_data = []
            cnt_data = []
            for instance, tid in PLI.items():

                [tid_data.append([instance, x]) for x in tid]
                cnt_data.append([instance, len(tid)])

            squ.create_table(self.cursor, table_name=tid_name, columns=tid_header)
            squ.insert_data(self.cursor, table_name=tid_name, columns=tid_header, data=tid_data)

            squ.create_table(self.cursor, table_name=cnt_name, columns=cnt_header)
            squ.insert_data(self.cursor, table_name=cnt_name, columns=cnt_header, data=cnt_data)

        # commit changes
        self.connection.commit()

        # free memory
        del self.PLIs

    def get_distribution_from_join(self, X):
        # prepare query
        table_names = [f'TID_{x}' for x in X]
        column_names = [x.lower() for x in X]

        rows = squ.equijoin(self.cursor, table_names, column_names, on='tid')

        # build distribution
        tmp = {}
        for row in rows:
            x = ','.join(row[:-1])
            index = row[-1]
            if x in tmp.keys():
                tmp[x].append(index)
            else:
                tmp[x] = [index]

        N = len(rows)
        P = {k: len(v) / N for k, v in tmp.items()}
        return P

    def get_distribution_from_PLI(self, X):
        tid_table_names = [f'TID_{x}' for x in X]
        rows = squ.PLI_join_cnts(self.cursor, tid_table_names, X)
        # squ.PLI_join_tids(self.cursor, tid_table_names, X)

        # build distribution
        N = sum([x[-1] for x in rows])
        P = {x[:-1]: x[-1]/N for x in rows}
        return P

    def get_distribution_from_SSJ(self, X):
        tid_table_names = [f'TID_{x}' for x in X]
        cnt_table_names = [f'CNT_{x}' for x in X]

        rho_N = 0
        lhs_weights = squ.SSJ_get_weights(self.cursor, cnt_table_names[0])
        lhs_tid_name = tid_table_names[0]
        for idx, next_att in enumerate(X[1:]):
            rhs_weights = squ.SSJ_get_weights(self.cursor, cnt_table_names[idx+1])
            lhs_weights, lhs_tid_name, rho_N = squ.SSJ_build_pairwise_tables(self.cursor, lhs_weights, rhs_weights, lhs_tid_name, tid_table_names[idx+1])
            squ.set_params(N=rho_N)

        return {k: v / rho_N for k, v in lhs_weights.items()}

    def get_distribution_from_SSJ2(self, X):
        tid_table_names = [f'TID_{x}' for x in X]
        cnt_table_names = [f'CNT_{x}' for x in X]

        rho_N = 0
        lhs_weights = squ.SSJ_get_weights2(self.cursor, cnt_table_names[0])
        lhs_tid_name = tid_table_names[0]
        for idx, next_att in enumerate(X[1:]):
            rhs_weights = squ.SSJ_get_weights2(self.cursor, cnt_table_names[idx+1])
            lhs_weights, lhs_tid_name, rho_N = squ.SSJ_build_pairwise_tables2(self.cursor, lhs_weights, rhs_weights, lhs_tid_name, tid_table_names[idx+1])
            squ.set_params(N=rho_N)

        return {k: v / rho_N for k, v in lhs_weights.items()}

    def entropy(self, X, mode='PLI'):
        P = None
        if mode == 'EXP':
            P = self.get_distribution_from_join(X)
        elif mode == 'PLI':
            P = self.get_distribution_from_PLI(X)
        elif mode == 'SSJ':
            P = self.get_distribution_from_SSJ(X)

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

    def clean(self, X, mode='SSJ'):
        singles = list(X)
        if mode == 'SSJ':
            for i in range(2, len(singles)+1):
                target = ",".join(singles[:i])
                self.cursor.execute(f'DROP TABLE "CNT_{target}"')
                self.cursor.execute(f'DROP TABLE "TID_{target}"')
        else:
            target = ','.join(X)
            self.cursor.execute(f'DROP TABLE "CNT_{target}"')

if __name__ == '__main__':
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"

    ssj_sampler = SSJSQL(path=path, name='NURSERY')


    ssj_sampler.entropy('ABC')


    print('hello')