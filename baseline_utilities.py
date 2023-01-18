import os
import pickle
import shutil
import time
from configparser import SafeConfigParser

import pandas as pd

from common import randomize_queries, calculate_entropy
from relation import Relation


def read_cfg(args):
    cfg_path = args.cfg

    config_parser = SafeConfigParser(os.environ)
    config_parser.read(cfg_path)

    as_dict = config_parser.__dict__
    data = as_dict['_sections']
    return data

def create_dir(name, args):
    out_dir = args.out_dir
    dataset_dir = f'{out_dir}\\{name}'
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.mkdir(dataset_dir)

    baselines = ['PLI', 'Kenig', 'project', 'SSJ', 'PSSJ', 'MSSJ']
    dirs = {}
    for baseline in baselines:
        baseline_dir = f'{dataset_dir}\\{baseline}'
        os.mkdir(baseline_dir)
        dirs[baseline] = baseline_dir

    return dirs

def gen_queries(path):
    with open(path, 'r') as F:
        header = F.readline().rstrip()

    attributes = header.split(',')
    return randomize_queries(attributes)

def run_relation(R, path, queries, out_dir):
    qs = []
    times1 = []
    times2 = []
    Hs = []
    HSs = []
    joins = []
    for q in queries:
        tic = time.perf_counter()
        H = calculate_entropy(path, q)
        toc = time.perf_counter()
        times1.append(toc - tic)

        q_str = ','.join(q)
        tic = time.perf_counter()
        res_data = R.entropy(q)
        toc = time.perf_counter()

        qs.append(q_str)
        times2.append(toc - tic)
        Hs.append(H)
        HSs.append(res_data['H'])
        joins.append(res_data['joins'])

    data = {
        'query': qs,
        'time_IDE': times1,
        'time': times2,
        'true H': Hs,
        'eval H': HSs,
        'joins': joins
    }

    df = pd.DataFrame(data)
    df.sort_values(by=['query', 'joins', 'time'])
    df.to_csv(f'{out_dir}\\out.csv', index=False)

    out_file = f'{out_dir}\\out.pkl'
    with open(out_file, 'wb') as F:
        pickle.dump(data, F)

# BASELINES MAIN
# def run_PLI(name, path, cfg, queries, out_dir):
def run_PLI(out_dir, **kwargs):
    path = kwargs['path']
    name = kwargs['name']
    queries = kwargs['queries']
    R = Relation(path=path, name=name, mode='pli')
    run_relation(R, path, queries, out_dir)
    del R

def run_Kenig(out_dir, **kwargs):
    path = kwargs['path']
    name = kwargs['name']
    queries = kwargs['queries']
    cfg = kwargs['cfg']
    l = cfg['PARAMS']['l']
    R = Relation(path=path, name=name, l=l)
    run_relation(R, path, queries, out_dir)
    del R

def run_project(out_dir, **kwargs):
    path = kwargs['path']
    name = kwargs['name']
    queries = kwargs['queries']
    R = Relation(path=path, name=name, mode='project')
    run_relation(R, path, queries, out_dir)
    del R

def run_SSJ(out_dir, **kwargs):
    path = kwargs['path']
    name = kwargs['name']
    queries = kwargs['queries']
    cfg = kwargs['cfg']
    l = cfg['PARAMS']['l']
    R = Relation(path=path, name=name, l=l, mode='ssj')
    run_relation(R, path, queries, out_dir)
    del R

def run_PSSJ(out_dir, **kwargs):
    path = kwargs['path']
    name = kwargs['name']
    queries = kwargs['queries']
    cfg = kwargs['cfg']
    l = cfg['PARAMS']['l']
    coverage = cfg['PARAMS']['coverage']
    R = Relation(path=path, name=name, l=l, mode='pssj', coverage=coverage)
    run_relation(R, path, queries, out_dir)
    del R

def run_MSSJ(out_dir, **kwargs):
    path = kwargs['path']
    name = kwargs['name']
    queries = kwargs['queries']
    cfg = kwargs['cfg']
    l = cfg['PARAMS']['l']
    coverage = cfg['PARAMS']['coverage']
    R = Relation(path=path, name=name, l=l, mode='mssj', coverage=coverage)
    run_relation(R, path, queries, out_dir)
    del R

def run_CSSJ(out_dir, **kwargs):
    path = kwargs['path']
    name = kwargs['name']
    queries = kwargs['queries']
    cfg = kwargs['cfg']
    l = cfg['PARAMS']['l']
    coverage = cfg['PARAMS']['coverage']
    R = Relation(path=path, name=name, l=l, mode='cssj', coverage=coverage)
    run_relation(R, path, queries, out_dir)
    del R


# MAIN FLOW
def run_baselines_dataset(name, path, cfg, args):
    dirs = create_dir(name, args)
    queries = gen_queries(path)

    baseline_funcs = [
        run_PLI,
        run_Kenig,
        run_project,
        run_SSJ,
        run_PSSJ,
        run_MSSJ,
        run_CSSJ
    ]

    kwargs = {
        'name': name,
        'path': path,
        'cfg': cfg,
        'queries': queries,
    }

    for func, dir in zip(baseline_funcs, dirs.values()):
        func(dir, **kwargs)

def run_baselines(args):
    cfg = read_cfg(args)
    datasets = cfg['DATASETS']

    for name, path in datasets.items():
        run_baselines_dataset(name, path, cfg, args)