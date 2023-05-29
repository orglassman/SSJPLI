#!/usr/bin/env python
import logging
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from dataset import RealDataSet
from sampler import Sampler


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-in_file', help='path to input CSV')
    parser.add_argument('-columns', required=False, help='target columns, separated by commas, e.g., SCHOOL,YEAR')
    parser.add_argument('-R', default=100, help='Repetition factor for sampling analysis')
    parser.add_argument('-out_dir', help='directory to dump output')
    parser.add_argument('-max_q_size', default=4, help='maximum query size')
    args = parser.parse_args()
    return args


def dump_df(out_dir, df, **kwargs):
    out_file = f'{out_dir}{os.sep}'
    for k, v in kwargs.items():
        out_file += f'{k}_{v}_'
    out_file = out_file.rstrip('_')
    out_file += '.pkl'
    with open(out_file, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'-I- Check {out_file}')


def single_query(ds, sampler, R, max_q_size, out_dir):
    coverages = [0.25, 0.5, 0.75, 0.9, 1]

    for qs in range(2, max_q_size + 1):
        H_true = []
        H_ssj = []
        t_ssj = []
        N_sample_ssj = []
        rho_ssj = []
        sigma_ssj = []
        H_pli = []
        t_pli = []
        records = []
        product_set_size = []
        # H_explicit = np.zeros(R)
        # t_explicit = np.zeros(R)
        # absolute measurements
        H_ssj_ratio = []
        t_ssj_ratio = []
        MSEs = []
        # t_explicit_ratio = np.zeros(R)


        dist_ssj = {}
        dist_orig = {}


        # generate R random queries
        for i in range(R):
            X = ds.random_query(qs)
            HX = ds.H(X)
            # vary coverage for each query
            for coverage in coverages:
                ssj_res_data = sampler.entropy(list(X), coverage=coverage)
                pli_res_data = sampler.entropy(list(X), mode='pli')
                # explicit_res_data = sampler.entropy(list(X), mode='explicit')
                # bounds = sampler.get_bounds(ssj_res_data)

                H_true.append(HX)
                H_ssj.append(ssj_res_data['H_ssj'])
                t_ssj.append(ssj_res_data['t_ssj'])
                N_sample_ssj.append(ssj_res_data['samples'])
                rho_ssj.append(ssj_res_data['rho'])
                sigma_ssj.append(ssj_res_data['sigma'])
                H_pli.append(pli_res_data['H_pli'])
                t_pli.append(pli_res_data['t_pli'])
                records.append(ssj_res_data['N'])
                product_set_size.append(ssj_res_data['product_set_size'])
                H_ssj_ratio.append(ssj_res_data['H_ssj'] / HX)
                t_ssj_ratio.append(ssj_res_data['t_ssj'] / pli_res_data['t_pli'])
                MSEs.append((HX-ssj_res_data['H_ssj'])**2)

                # H_explicit[i] = explicit_res_data['H_explicit']
                # t_explicit[i] = explicit_res_data['t_explicit']
                # t_explicit_ratio[i] = t_explicit[i] / t_pli[i]


        measurements = {
            'H_ssj': H_ssj,
            'H_true': H_true,
            # 'HQ': HQs,
            # 'HQUN': HQUNs,
            # 'MISS': MISSs,
            # 'EMPTY': EMPTYs,
            # 'I': Is,
            'samples_ssj': N_sample_ssj,
            't_ssj': t_ssj,
            'rho': rho_ssj,
            'sigma': sigma_ssj,
            'records': records,
            'product_set': product_set_size,
            # 't_explicit': t_explicit,
            't_pli': t_pli,
            't_ssj_ratio': t_ssj_ratio,
            # 't_explicit_ratio': t_explicit_ratio,
            'h_ratio': H_ssj_ratio,
            'MSE': MSEs
        }

        df = pd.DataFrame(measurements)
        groups = df.groupby('sigma')
        for name, group in groups:
            dump_df(out_dir, group, **{'coverage': name, 'query_size': qs})

    print(f'-I- Check {out_dir}')


def real_data_main():
    args = parse_args()
    in_file = args.in_file
    R = int(args.R)
    max_q = int(args.max_q_size)
    out_dir = args.out_dir

    ds = RealDataSet(path=in_file)
    sampler = Sampler(ds)

    single_query(ds, sampler, R, max_q, out_dir)


if __name__ == '__main__':
    real_data_main()
