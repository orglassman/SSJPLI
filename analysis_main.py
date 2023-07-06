#!/usr/bin/env python
import os
import pickle
from argparse import ArgumentParser

import pandas as pd

from dataset import RealDataSet
from ssjsampler import SSJSampler


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-in_file', help='path to input CSV')
    parser.add_argument('-columns', required=False, help='target columns separated by commas, e.g., SCHOOL,YEAR')
    parser.add_argument('-R', default=100, help='sample size')
    parser.add_argument('-out_dir', help='directory to dump output')
    parser.add_argument('-q_sizes', default="2,4,6", help='target query sizes separated by commas, e.g., 2,3,4,5')
    parser.add_argument('-coverages', default="0.25,0.5,0.75,0.9,0.99",
                        help='target coverages separated by commas, e.g., 0.25,0.75,0.9')
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


def single_query_ssj(args):
    ssj_args = get_ssj_args(args)

    ds = ssj_args['ds']
    sampler = ssj_args['sampler']
    out_dir = ssj_args['out_dir']

    for qs in ssj_args['q_sizes']:
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
        for i in range(ssj_args['R']):
            X = ds.random_query(qs)
            HX = ds.H(X)
            pli_res_data = sampler.entropy(list(X), mode='pli')

            # vary coverage for each query
            for coverage in ssj_args['coverages']:
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
                MSEs.append((HX - target_res_data['H_s']) ** 2)

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

            'rho': rho_ssj,
            'sigma': sigma_ssj,

            'MSE': MSEs
        }
        df = pd.DataFrame(measurements)
        groups = df.groupby('sigma')
        for name, group in groups:
            dump_df(out_dir, group, **{'mode': 'ssj', 'coverage': name, 'query_size': qs})

    print(f'-I- Check {out_dir}')


def single_query_isj(args):
    isj_args = get_isj_args(args)

    ds = isj_args['ds']
    sampler = isj_args['sampler']
    out_dir = isj_args['out_dir']

    for qs in isj_args['q_sizes']:
        print(f'-I- Query size {qs}')
        H_true = []

        # target
        H_target = []
        t_target = []
        N_sample = []
        target_precisions = []
        effective_precisions = []

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
        for i in range(isj_args['R']):
            X = ds.random_query(qs)
            HX = ds.H(X)
            pli_res_data = sampler.entropy(list(X), mode='pli')
            for precision in isj_args['precisions']:
                # vary coverage for each query
                target_res_data = sampler.entropy(X, mode='isj', isj_precision=precision)

                H_true.append(HX)
                H_target.append(target_res_data['H_s'])
                t_target.append(target_res_data['t_s'])
                N_sample.append(target_res_data['num_samples'])
                target_precisions.append(target_res_data['precision'])
                effective_precisions.append(target_res_data['precision_eff'])
                # PLI
                H_pli.append(pli_res_data['H_pli'])
                t_pli.append(pli_res_data['t_pli'])

                # general
                records.append(target_res_data['N'])
                product_set_size.append(target_res_data['product_set_size'])

                # absolute measurements
                H_s_ratio.append(target_res_data['H_s'] / HX)
                t_s_ratio.append(target_res_data['t_s'] / pli_res_data['t_pli'])
                MSEs.append((HX - target_res_data['H_s']) ** 2)

        measurements = {
            'H_s': H_target,
            't_s': t_target,
            'H_true': H_true,
            'num_samples': N_sample,
            'precision': target_precisions,
            'precision_eff': effective_precisions,
            'records': records,
            'product_set': product_set_size,
            't_pli': t_pli,
            't_s_ratio': t_s_ratio,
            'h_s_ratio': H_s_ratio,
            'MSE': MSEs
        }

        df = pd.DataFrame(measurements)
        groups = df.groupby('precision')
        for name, group in groups:
            dump_df(out_dir, group, **{'mode': 'isj', 'precision': name, 'query_size': qs})

    print(f'-I- Check {out_dir}')


def parse_q_sizes(q_sizes):
    return [int(x) for x in q_sizes.split(',')]


def parse_coverages(coverages):
    return [float(x) for x in coverages.split(',')]


def get_ssj_args(args):
    in_file = args.in_file
    ds = RealDataSet(path=in_file)
    sampler = SSJSampler(ds)

    ssj_args = {
        'ds': ds,
        'sampler': sampler,
        'R': int(args.R),
        'q_sizes': parse_q_sizes(args.q_sizes),
        'coverages': parse_coverages(args.coverages),
        'out_dir': args.out_dir

    }
    return ssj_args


def get_isj_args(args):
    in_file = args.in_file
    ds = RealDataSet(path=in_file)
    sampler = SSJSampler(ds)

    isj_args = {
        'ds': ds,
        'sampler': sampler,
        'R': int(args.R),
        'q_sizes': parse_q_sizes(args.q_sizes),
        'precisions': parse_coverages(args.coverages),      # COVERAGE ON H INSTEAD ON ENTRIES!!!
        'out_dir': args.out_dir
    }
    return isj_args


def real_data_main():
    args = parse_args()
    mode = args.mode

    if mode == 'ssj':
        single_query_ssj(args)
    elif mode == 'isj':
        single_query_isj(args)


if __name__ == '__main__':
    real_data_main()
