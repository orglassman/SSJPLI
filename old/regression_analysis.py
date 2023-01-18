"""
designated set of tests to help maintainability
"""
import time
from math import floor
import random


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from relation import Relation

def test1():
    """
    simple run
    """
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    name = 'Nursery'
    num_atts = 9
    l = 3

    print(f'-I- For l={l}, process single query')


    tic = time.perf_counter()
    R = Relation(path=path, name=name, num_atts=num_atts, l=l)
    toc = time.perf_counter()
    print(f'-I- Time creating relation {toc - tic} seconds')

    X = ['A', 'B', 'D']

    tic = time.perf_counter()
    freq_X = R.get_frequency_LFJ(X)
    toc = time.perf_counter()
    print(f'-I- Time calculating frequencies {toc - tic} seconds')

    return True

def test2():
    """
    test for 2 different values of l
    """
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    name = 'Nursery'
    num_atts = 9
    l1 = 3          # expecting shorter init time, longer query processing time
    l2 = 5          # opposite

    print(f'-I- For l={l1},{l2} test performance')

    tic1 = time.perf_counter()
    R1 = Relation(path=path, name=name, num_atts=num_atts, l=l1)
    toc1 = time.perf_counter()

    tic2 = time.perf_counter()
    R2 = Relation(path=path, name=name, num_atts=num_atts, l=l2)
    toc2 = time.perf_counter()

    print(f'-I- Time creating relation with l={l1} {toc1 - tic1} seconds')
    print(f'-I- Time creating relation with l={l2} {toc2 - tic2} seconds')

    X = ['A', 'B', 'D']

    tic1 = time.perf_counter()
    freq_X1 = R1.get_frequency_LFJ(X)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    freq_X2 = R2.get_frequency_LFJ(X)
    toc2 = time.perf_counter()
    print(f'-I- Time calculating frequencies l={l1}, {toc1 - tic1} seconds')
    print(f'-I- Time calculating frequencies l={l2}, {toc2 - tic2} seconds')

    return True

def test3():
    """
    for given l, test various queries
    """
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    name = 'Nursery'
    num_atts = 9
    l = 3

    print(f'-I- For l={l} test several queries')

    tic = time.perf_counter()
    R = Relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True)
    toc = time.perf_counter()
    print(f'-I- Time creating relation {toc - tic} seconds')

    X = ['A', 'B', 'D']
    Y = ['A', 'B']        # shortest time
    Z = ['A', 'D', 'H']   # longest time

    tic = time.perf_counter()
    freq_X = R.get_frequency_LFJ(X)
    toc = time.perf_counter()
    print(f'-I- Time calculating frequencies {toc - tic} seconds')
    tic = time.perf_counter()
    freq_Y = R.get_frequency_LFJ(Y)
    toc = time.perf_counter()
    print(f'-I- Time calculating frequencies {toc - tic} seconds')
    tic = time.perf_counter()
    freq_Z = R.get_frequency_LFJ(Z)
    toc = time.perf_counter()
    print(f'-I- Time calculating frequencies {toc - tic} seconds')

    return True

def test4():
    """
    test leapfrog init in relation c'tor
    (expecting leapfrog init to be faster than traditional)
    """
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    name = 'Nursery'
    num_atts = 9
    l = 3

    print(f'-I- For l={l} test leapfrog init performance')

    tic1 = time.perf_counter()
    R = Relation(path=path, name=name, num_atts=num_atts, l=l)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    S = Relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True)
    toc2 = time.perf_counter()

    print(f'-I- Time creating relation without leapfrog init {toc1 - tic1} seconds')
    print(f'-I- Time creating relation with leapfrog init {toc2 - tic2} seconds')

    X = ['A', 'B', 'D']
    tic1 = time.perf_counter()
    freq_X1 = R.get_frequency_LFJ(X)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    freq_X2 = S.get_frequency_LFJ(X)
    toc2 = time.perf_counter()

    print(f'-I- Time processing query [\'A\', \'B\', \'D\'] without leapfrog init {toc1 - tic1} seconds')
    print(f'-I- Time processing query [\'A\', \'B\', \'D\'] with leapfrog init {toc2 - tic2} seconds')

    return True

def test5():
    """candidate generation:
        check cartesian product vs. projection"""
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    name = 'Nursery'
    num_atts = 9
    l = 3

    print(f'-I- For l={l} test candidate baselines: cartesian product vs. projection')

    tic1 = time.perf_counter()
    R = Relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True, candidates='cartesian')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    S = Relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True, candidates='project')
    toc2 = time.perf_counter()

    print(f'-I- Time creating relation with cartesian product {toc1 - tic1} seconds')
    print(f'-I- Time creating relation with projection {toc2 - tic2} seconds')

    X = ['A', 'B', 'D']
    tic1 = time.perf_counter()
    freq_X1 = R.get_frequency_LFJ(X)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    freq_X2 = S.get_frequency_LFJ(X)
    toc2 = time.perf_counter()

    print(f'-I- Time processing query [\'A\', \'B\', \'D\'] for cartesian product {toc1 - tic1} seconds')
    print(f'-I- Time processing query [\'A\', \'B\', \'D\'] for projection {toc2 - tic2} seconds')

    return True



def run_routine_tests():
    tests = [test1, test2, test3, test4, test5]
    for i, test in enumerate(tqdm(tests)):
        try:
            print('')
            print(f'-I- Running test {i}')
            status = test()
            print(f'-I- Pass test {i}: {status}')
            print('')
        except:
            print(f'-F- Failed test {i}')

    print('-I- Finished running all tests')
    print('-I- Have a wonderful day sir')

def test_leapfrog_init_time():
    """
    test the init time for leapfrog init compared to traditional init
    """
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Credit\\credit.csv"
    name = 'Credit'
    num_atts = 16
    l = 4

    times1 = []
    times2 = []
    N = 200
    print(f'-I- Running relation init {N} times')
    for i in tqdm(range(N)):
        tic1 = time.perf_counter()
        R = Relation(path=path, name=name, num_atts=num_atts, l=l)
        toc1 = time.perf_counter()
        times1.append(toc1 - tic1)
        del R
        tic2 = time.perf_counter()
        S = Relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True)
        toc2 = time.perf_counter()
        times2.append(toc2 - tic2)
        del S

    plt.figure()
    plt.scatter(range(N), times1)
    plt.scatter(range(N), times2)
    plt.xlabel('# iteration')
    plt.ylabel('time [seconds]')
    plt.title('Relation Init - Leapfrog vs. Traditional')
    plt.legend(['no leapfrog', 'with leapfrog'])
    data = pd.DataFrame({'times_traditional':times1, 'times_leapfrog':times2})
    pass

def test_leapfrog_init_various_l():
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    name = 'Nursery'
    num_atts = 9

    min_l = 1
    max_l = num_atts
    N = 10
    init_times = {k: [] for k in range(min_l, max_l+1)}
    query_times = {k: [] for k in range(min_l, max_l+1)}
    print(f'-I- Running relation init with l=1 {N} times')
    for l in range(min_l, max_l + 1):
        # init_times[l] = test_leapfrog_init_single_l(path=path, name=name, num_atts=num_atts, l=l, N=N)
        query_times[l] = test_leapfrog_single_query_single_l(path=path, name=name, num_atts=num_atts, l=l, N=N)
    print('-I- Finished')

def test_leapfrog_init_single_l(path, name, num_atts, l, N):
    res = []
    for i in tqdm(range(N)):
        tic = time.perf_counter()
        R = Relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True)
        toc = time.perf_counter()
        del R
        res.append(toc-tic)

    return res

def test_leapfrog_single_query_single_l(path, name, num_atts, l, N):
    """test single query involving all attribtues of nursery"""
    R = Relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True)
    X = list('ABCDEFGHI')
    res = []
    for i in tqdm(range(N)):
        tic = time.perf_counter()
        freq_X = R.get_frequency(X)
        toc = time.perf_counter()

        res.append(toc - tic)
        del freq_X

    return res


def test_queries_various_l():
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    name = 'Nursery'
    num_atts = 9
    N = 100
    # min_l = 2
    # max_l = 9

    attributes = list('ABCDEFGHI')
    l = 3
    query_sizes = [i for i in range(1,9+1)]

    res = {k:[] for k in query_sizes}

    for size in query_sizes:
        # randomly select queries and measure time
        queries = [random.sample(attributes, size) for i in range(N)]
        res[size] = test_queries_single_l(path, name, num_atts, l, N, queries)

    df = pd.DataFrame(res)
    print('-I- Finished ')


def test_queries_single_l(path, name, num_atts, l, N, queries):
    res = []
    R = Relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True)
    for idx, X in tqdm(enumerate(queries)):
        tic = time.perf_counter()
        Q = R.get_frequency(X)
        toc = time.perf_counter()
        res.append(toc-tic)

    del R
    return res

def test_candidates():
    path1 = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\Nursery.csv"
    name1 = 'Nursery'
    num_atts1 = 9
    l1 = 3

    path2 = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\School_Results\\school_results.csv"
    name2 = 'SR'
    num_atts2 = 27
    l2 = 3

    N = 100

    delta1 = []
    delta2 = []
    delta3 = []
    delta4 = []

    for i in range(N):

        tic1 = time.perf_counter()
        # R1 = relation(path=path1, name=name1, num_atts=num_atts1, l=l1, leapfrog_init=True, candidates='cartesian')
        toc1 = time.perf_counter()
        delta1.append(toc1 - tic1)
        tic2 = time.perf_counter()
        # R2 = relation(path=path1, name=name1, num_atts=num_atts1, l=l1, leapfrog_init=True, candidates='project')
        toc2 = time.perf_counter()
        delta2.append(toc2 - tic2)

        tic3 = time.perf_counter()
        S1 = Relation(path=path2, name=name2, num_atts=num_atts2, l=l2, leapfrog_init=False, candidates='cartesian')
        toc3 = time.perf_counter()
        delta3.append(toc3 - tic3)
        tic4 = time.perf_counter()
        S2 = Relation(path=path2, name=name2, num_atts=num_atts2, l=l2, leapfrog_init=True, candidates='project')
        toc4 = time.perf_counter()
        delta4.append(toc4 - tic4)

    data = {'nursery_cartesian_l_3': delta1,
            'nursery_projection_l_3': delta2,
            'credit_cartesian_l_3': delta3,
            'credit_projection_l_4': delta4
            }
    df = pd.DataFrame(data)
    print('lalala')




if __name__ == '__main__':
    # run_routine_tests()
    test_candidates()