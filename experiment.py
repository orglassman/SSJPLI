#!/usr/bin/env python
import time
from argparse import ArgumentParser

import pandas as pd

from analysis_main import parse_q_sizes, dump_df
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
    parser.add_argument('-mode',        default='ssj',                      help='pli/ssj/isj/explicit')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sampler = SSJSQL(path=args.in_file, name=args.name)


    for q in parse_q_sizes(args.qs):
        print(f'-I- Query size {q}')

        Hs = []
        ts = []
        ps = []

        # generate r random queries
        for i in range(args.r):
            X = randomize_query(sampler.omega, q)

            start = time.perf_counter()
            H = sampler.entropy_join(X)
            finish = time.perf_counter()

            Hs.append(H)
            ts.append(finish - start)
            ps.append(sampler.get_ps_size(X))

        measurements = {
            'H': Hs,
            't': ts,
            'product_set': ps
        }
        df = pd.DataFrame(measurements)
        dump_df(args.out_dir, df, **{'mode': 'sql', 'query_size': q})

    print(f'-I- Check {args.out_dir}')
