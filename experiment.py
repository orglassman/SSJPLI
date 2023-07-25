#!/usr/bin/env python
import time
from argparse import ArgumentParser

import pandas as pd
import sql_utils as squ

from analysis_main import parse_q_sizes, dump_df, parse_coverages
from common import randomize_query
from ssj_sqlite import SSJSQL


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-in_file',                                         help='input CSV')
    parser.add_argument('-name',                                            help='name for DB')
    parser.add_argument('-out_dir',                                         help='output directory')

    parser.add_argument('-r',           default=100,                        help='sample size')
    parser.add_argument('-qs',          default="2,4,6",                    help='target query sizes, separated by commas, e.g., 2,3,4,5')
    parser.add_argument('-coverages',   default="0.25,0.5,0.75,0.9,0.99",   help='target coverages, separated by commas, e.g., 0.25,0.75,0.9')
    parser.add_argument('-machine',     default='local',                    help='local/dragon')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sampler = SSJSQL(path=args.in_file, name=args.name)
    squ.set_params(N=sampler.N, M=sampler.M)

    for q in parse_q_sizes(args.qs):
        print(f'-I- Query size {q}')
        for coverage in parse_coverages(args.coverages):
            print(f'-I- Query size {q}, coverage {coverage}')
            squ.set_params(coverage=coverage)

            H_ssjs = []
            H_exps = []
            H_plis = []
            t_ssjs = []
            t_exps = []
            t_plis = []
            ps = []

            for i in range(args.r):
                print(f'-I- Iteration {i}')
                # generate r random queries
                X = randomize_query(sampler.omega, q)

                start_ssj = time.perf_counter()
                H_ssj = sampler.entropy(X, mode='SSJ')
                finish_ssj = time.perf_counter()
                # clear out previously generated tables
                sampler.clean(X)

                # baselines
                start_exp = time.perf_counter()
                H_exp = sampler.entropy(X, mode='EXP')
                finish_exp = time.perf_counter()

                start_pli = time.perf_counter()
                H_pli = sampler.entropy(X, mode='PLI')
                finish_pli = time.perf_counter()
                sampler.clean(X, mode='PLI')

                H_ssjs.append(H_ssj)
                t_ssjs.append(finish_ssj - start_ssj)
                ps.append(sampler.get_ps_size(X))
                H_exps.append(H_exp)
                t_exps.append(finish_exp - start_exp)
                H_plis.append(H_pli)
                t_plis.append(finish_pli - start_pli)

            measurements = {
                'H_exp': H_exps,
                'H_pli': H_plis,
                'H_ssj': H_ssjs,
                't_exp': t_exps,
                't_pli': t_plis,
                't_ssj': t_ssjs,
                'product_set': ps
            }

            df = pd.DataFrame(measurements)
            dump_df(args.out_dir, df, **{'machine': args.machine, 'version': 'sql', 'coverage': coverage, 'query_size': q})

    print(f'-I- Check {args.out_dir}')
