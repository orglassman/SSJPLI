#!/usr/bin/env python

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

def single_query(X, ds, sampler, R, out_dir):
    coverages = [0.25, 0.5, 0.75, 0.9, 0.99]
    output_data = {c: {} for c in coverages}
    for coverage in coverages:
        H_true = np.zeros(R)

        H_ssj = np.zeros(R)
        t_ssj = np.zeros(R)
        N_sample_ssj = np.zeros(R)
        rho_ssj = np.zeros(R)

        # HQs = np.zeros(R)
        # HQUNs = np.zeros(R)
        # MISSs = np.zeros(R)
        # EMPTYs = np.zeros(R)

        H_pli = np.zeros(R)
        t_pli = np.zeros(R)

        H_explicit = np.zeros(R)
        t_explicit = np.zeros(R)

        # absolute measurements
        H_ssj_ratio = np.zeros(R)
        t_ssj_ratio = np.zeros(R)
        t_explicit_ratio = np.zeros(R)

        # average over R repetitions
        for i in range(R):
            # generate results
            ssj_res_data = sampler.entropy(list(X), coverage=coverage)
            pli_res_data = sampler.entropy(list(X), mode='pli')
            explicit_res_data = sampler.entropy(list(X), mode='explicit')

            # bounds = sampler.get_bounds(ssj_res_data)

            H_true[i] = ds.H(list(X))

            H_ssj[i] = ssj_res_data['H_ssj']
            t_ssj[i] = ssj_res_data['t_ssj']
            N_sample_ssj[i] = ssj_res_data['samples']
            rho_ssj[i] = ssj_res_data['rho']

            #
            # HQs[i] = bounds['HQ']
            # HQUNs[i] = bounds['HQUN']
            # MISSs[i] = bounds['MISS']
            # EMPTYs[i] = bounds['EMPTY']
            # Is[i] = ds.get_I()
            # records[i] = ds.get_N()
            #

            H_pli[i] = pli_res_data['H_pli']
            t_pli[i] = pli_res_data['t_pli']

            H_explicit[i] = explicit_res_data['H_explicit']
            t_explicit[i] = explicit_res_data['t_explicit']

            H_ssj_ratio[i] = H_ssj[i] / H_true[i]
            t_ssj_ratio[i] = t_ssj[i] / t_pli[i]
            t_explicit_ratio[i] = t_explicit[i] / t_pli[i]
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
            't_explicit': t_explicit,
            't_pli': t_pli,
            't_ssj_ratio': t_ssj_ratio,
            't_explicit_ratio': t_explicit_ratio,
            'h_ratio': H_ssj_ratio
        }

        for k, measurement in measurements.items():
            avg = np.average(measurement)
            std = np.std(measurement)
            output_data[coverage][k] = [avg, std]

    dfs = {c: pd.DataFrame(output_data[c]) for c in coverages}
    for c, df in dfs.items():
        dump_df(out_dir, df, **{'coverage': c, 'query_size':len(X)})

    print(f'-I- Check {out_dir}')

def real_data_main():
    args = parse_args()
    in_file = args.in_file
    R = int(args.R)
    out_dir = args.out_dir

    ds = RealDataSet(path=in_file)
    sampler = Sampler(ds)

    X_strs = ['ABC', 'ABCD', 'ABCDE', 'ABCDEF', 'ABCDEFG', 'ABCDEFGH']

    for x in X_strs:
        single_query(x, ds, sampler, R, out_dir)


if __name__ == '__main__':
    real_data_main()
