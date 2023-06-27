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
    parser.add_argument('-columns', required=False, help='target columns separated by commas, e.g., SCHOOL,YEAR')
    parser.add_argument('-R', default=100, help='sample size')
    parser.add_argument('-out_dir', help='directory to dump output')
    parser.add_argument('-q_sizes', default="2,4,6", help='target query sizes separated by commas, e.g., 2,3,4,5')
    parser.add_argument('-coverages', default="0.25,0.5,0.75,0.9,0.99", help='target coverages separated by commas, e.g., 0.25,0.75,0.9')
    parser.add_argument('-mode', default='ssj', help='pli/ssj/isj/msj/explicit')
    parser.add_argument('-K', default=1000, help='number of samples for ISJ')
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


def single_query_ssj(ds, sampler, R, q_sizes, coverages, out_dir):
    for qs in q_sizes:
        print(f'-I- Query size {qs}')
        H_true = []

        H_target = []
        t_target = []
        N_sample = []

        rho_ssj = []
        sigma_ssj = []

        H_pli = []
        t_pli = []

        records = []
        product_set_size = []

        # absolute measurements
        H_s_ratio = []
        t_s_ratio = []
        MSEs = []

        # generate R random queries
        for i in range(R):
            X = ds.random_query(qs)
            HX = ds.H(X)
            pli_res_data = sampler.entropy(list(X), mode='pli')

            # vary coverage for each query
            for coverage in coverages:
                target_res_data = sampler.entropy(X, coverage=coverage)

                H_true.append(HX)
                H_target.append(target_res_data['H_s'])
                t_target.append(target_res_data['t_s'])
                N_sample.append(target_res_data['num_samples'])

                rho_ssj.append(target_res_data['rho'])
                sigma_ssj.append(target_res_data['sigma'])

                # PLI
                H_pli.append(pli_res_data['H_pli'])
                t_pli.append(pli_res_data['t_pli'])

                # general
                records.append(target_res_data['N'])
                product_set_size.append(target_res_data['product_set_size'])

                # absolute measurements
                H_s_ratio.append(target_res_data['H_s'] / HX)
                t_s_ratio.append(target_res_data['t_s'] / pli_res_data['t_pli'])
                MSEs.append((HX-target_res_data['H_s'])**2)

        measurements = {
            'H_s': H_target,
            't_s': t_target,
            'H_true': H_true,
            'num_samples': N_sample,
            'records': records,
            'product_set': product_set_size,
            't_pli': t_pli,
            't_s_ratio': t_s_ratio,
            'h_s_ratio': H_s_ratio,

            'rho':  rho_ssj,
            'sigma':  sigma_ssj,

            'MSE': MSEs
        }
        df = pd.DataFrame(measurements)
        groups = df.groupby('sigma')
        for name, group in groups:
            dump_df(out_dir, group, **{'mode': 'ssj', 'coverage': name, 'query_size': qs})

    print(f'-I- Check {out_dir}')

def single_query_isj(ds, sampler, R, q_sizes, K, out_dir):
    for qs in q_sizes:
        print(f'-I- Query size {qs}')
        H_true = []

        # target
        H_target = []
        t_target = []
        N_sample = []

        # PLI
        H_pli = []
        t_pli = []

        # meta
        records = []
        product_set_size = []

        # absolute measurements
        H_s_ratio = []
        t_s_ratio = []
        MSEs = []

        # generate R random queries
        for i in range(R):
            X = ds.random_query(qs)
            HX = ds.H(X)
            pli_res_data = sampler.entropy(list(X), mode='pli')

            # vary coverage for each query
            target_res_data = sampler.entropy(X, mode='isj', K=K)

            H_true.append(HX)
            H_target.append(target_res_data['H_s'])
            t_target.append(target_res_data['t_s'])
            N_sample.append(target_res_data['num_samples'])


            # PLI
            H_pli.append(pli_res_data['H_pli'])
            t_pli.append(pli_res_data['t_pli'])

            # general
            records.append(target_res_data['N'])
            product_set_size.append(target_res_data['product_set_size'])

            # absolute measurements
            H_s_ratio.append(target_res_data['H_s'] / HX)
            t_s_ratio.append(target_res_data['t_s'] / pli_res_data['t_pli'])
            MSEs.append((HX-target_res_data['H_s'])**2)

        measurements = {
            'H_s': H_target,
            't_s': t_target,
            'H_true': H_true,
            'num_samples': N_sample,
            'records': records,
            'product_set': product_set_size,
            't_pli': t_pli,
            't_s_ratio': t_s_ratio,
            'h_s_ratio': H_s_ratio,
            'MSE': MSEs
        }
        df = pd.DataFrame(measurements)
        dump_df(out_dir, df, **{'mode': 'isj', 'query_size': qs})

    print(f'-I- Check {out_dir}')
    pass


def single_query(ds, sampler, R, q_sizes, coverages, mode, K, out_dir):
    if mode == 'ssj':
        single_query_ssj(ds, sampler, R, q_sizes, coverages, out_dir)
    elif mode == 'isj':
        single_query_isj(ds, sampler, R, q_sizes, K, out_dir)


    for qs in q_sizes:
        print(f'-I- Query size {qs}')
        H_true = []

        H_target = []
        t_target = []
        N_sample = []

        rho_ssj = []
        sigma_ssj = []

        H_pli = []
        t_pli = []

        records = []
        product_set_size = []

        # absolute measurements
        H_s_ratio = []
        t_s_ratio = []
        MSEs = []

        # dist_ssj = {}
        # dist_orig = {}

        # generate R random queries
        for i in range(R):
            X = ds.random_query(qs)
            HX = ds.H(X)
            pli_res_data = sampler.entropy(list(X), mode='pli')

            # vary coverage for each query
            for coverage in coverages:
                target_res_data = sampler.entropy(X, coverage=coverage, mode=mode, K=K)

                # explicit_res_data = sampler.entropy(list(X), mode='explicit')
                # bounds = sampler.get_bounds(ssj_res_data)

                H_true.append(HX)
                H_target.append(target_res_data['H_s'])
                t_target.append(target_res_data['t_s'])
                N_sample.append(target_res_data['num_samples'])

                # SSJ
                if mode == 'ssj':
                    rho_ssj.append(target_res_data['rho'])
                    sigma_ssj.append(target_res_data['sigma'])

                # PLI
                H_pli.append(pli_res_data['H_pli'])
                t_pli.append(pli_res_data['t_pli'])

                # general
                records.append(target_res_data['N'])
                product_set_size.append(target_res_data['product_set_size'])

                # absolute measurements
                H_s_ratio.append(target_res_data['H_s'] / HX)
                t_s_ratio.append(target_res_data['t_s'] / pli_res_data['t_pli'])
                MSEs.append((HX-target_res_data['H_s'])**2)


        measurements = {
            'H_s': H_target,
            't_s': t_target,
            'H_true': H_true,
            'num_samples': N_sample,
            'records': records,
            'product_set': product_set_size,
            't_pli': t_pli,
            't_s_ratio': t_s_ratio,
            'h_s_ratio': H_s_ratio,

            'MSE': MSEs
        }

        if mode == 'ssj':
            measurements['rho'] = rho_ssj
            measurements['sigma'] = sigma_ssj

        df = pd.DataFrame(measurements)
        if mode == 'ssj':
            groups = df.groupby('sigma')
            for name, group in groups:
                dump_df(out_dir, group, **{'mode': sampler.mode, 'coverage': name, 'query_size': qs})

    print(f'-I- Check {out_dir}')

def parse_q_sizes(q_sizes):
    return [int(x) for x in q_sizes.split(',')]

def parse_coverages(coverages):
    return [float(x) for x in coverages.split(',')]

def real_data_main():
    args = parse_args()

    in_file = args.in_file
    mode = args.mode
    R = int(args.R)
    q_sizes = parse_q_sizes(args.q_sizes)
    coverages = parse_coverages(args.coverages)
    K = int(args.K)
    out_dir = args.out_dir

    ds = RealDataSet(path=in_file)
    sampler = Sampler(ds)

    single_query(ds, sampler, R, q_sizes, coverages, mode, K, out_dir)


if __name__ == '__main__':
    real_data_main()
