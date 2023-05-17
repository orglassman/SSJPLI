#!/usr/bin/env python

import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from dataset import RealDataSet
from sampler import SyntheticSampler


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-in_file', help='path to input CSV')
    parser.add_argument('-columns', help='target columns, separated by commas, e.g., SCHOOL,YEAR')
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


def real_data_main():
    args = parse_args()
    in_file = args.in_file
    columns = args.columns.split(',')
    R = int(args.R)
    out_dir = args.out_dir

    ds = RealDataSet(path=in_file, columns=columns)
    sampler = SyntheticSampler(ds)

    coverages = [0.25, 0.5, 0.75, 0.9, 0.99]
    res_data = {c: {} for c in coverages}
    for coverage in coverages:
        Hs = np.zeros(R)
        H_trues = np.zeros(R)
        HQs = np.zeros(R)
        HQUNs = np.zeros(R)
        MISSs = np.zeros(R)
        EMPTYs = np.zeros(R)
        NSs = np.zeros(R)
        rhos = np.zeros(R)
        t_ssjs = np.zeros(R)
        Is = np.zeros(R)
        records = np.zeros(R)
        t_explicits = np.zeros(R)
        t_traverses = np.zeros(R)

        # absolute measurements
        t_ssj_ratios = np.zeros(R)
        t_explicit_ratios = np.zeros(R)
        h_ratios = np.zeros(R)

        # average over R repetitions
        for i in range(R):
            ssj_res_data = sampler.entropy(coverage=coverage, mode='ssj')
            bounds = sampler.get_bounds(ssj_res_data)
            t_explicit = sampler.explicit_entropy()
            t_traverse = sampler.traverse_entropy()

            Hs[i] = ssj_res_data['H']
            H_trues[i] = ds.HAB()
            HQs[i] = bounds['HQ']
            HQUNs[i] = bounds['HQUN']
            MISSs[i] = bounds['MISS']
            EMPTYs[i] = bounds['EMPTY']
            Is[i] = ds.get_I()
            records[i] = ds.get_N()
            NSs[i] = ssj_res_data['num_samples']
            t_ssjs[i] = ssj_res_data['time']
            rhos[i] = ssj_res_data['rho']
            t_explicits[i] = t_explicit
            t_traverses[i] = t_traverse

            t_ssj_ratios[i] = t_ssjs[i] / t_traverses[i]
            t_explicit_ratios[i] = t_explicits[i] / t_traverses[i]
            h_ratios[i] = Hs[i] / H_trues[i]

        measurements = {'H': Hs,
                        'H_true': H_trues,
                        'HQ': HQs,
                        'HQUN': HQUNs,
                        'MISS': MISSs,
                        'EMPTY': EMPTYs,
                        'I': Is,
                        'N': NSs,
                        't_ssj': t_ssjs,
                        'rho': rhos,
                        't_explicit': t_explicits,
                        't_traverse': t_traverses,
                        't_ssj_ratio': t_ssj_ratios,
                        't_explicit_ratio': t_explicit_ratios,
                        'h_ratio': h_ratios
                        }

        for k, measurement in measurements.items():
            avg = np.average(measurement)
            std = np.std(measurement)
            res_data[coverage][k] = [avg, std]

    dfs = {c: pd.DataFrame(res_data[c]) for c in coverages}
    for c, df in dfs.items():
        dump_df(out_dir, df, **{'coverage': c})

    print(f'-I- Check {out_dir}')


if __name__ == '__main__':
    real_data_main()
