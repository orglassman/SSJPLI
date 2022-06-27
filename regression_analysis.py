"""
designated set of tests to help maintainability
"""
import time

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from relation import relation


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
    R = relation(path=path, name=name, num_atts=num_atts, l=l)
    toc = time.perf_counter()
    print(f'-I- Time creating relation {toc - tic} seconds')

    X = ['A', 'B', 'D']

    tic = time.perf_counter()
    freq_X = R.get_frequency(X)
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
    R1 = relation(path=path, name=name, num_atts=num_atts, l=l1)
    toc1 = time.perf_counter()

    tic2 = time.perf_counter()
    R2 = relation(path=path, name=name, num_atts=num_atts, l=l2)
    toc2 = time.perf_counter()

    print(f'-I- Time creating relation with l={l1} {toc1 - tic1} seconds')
    print(f'-I- Time creating relation with l={l2} {toc2 - tic2} seconds')

    X = ['A', 'B', 'D']

    tic1 = time.perf_counter()
    freq_X1 = R1.get_frequency(X)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    freq_X2 = R2.get_frequency(X)
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
    R = relation(path=path, name=name, num_atts=num_atts, l=l)
    toc = time.perf_counter()
    print(f'-I- Time creating relation {toc - tic} seconds')

    X = ['A', 'B', 'D']
    Y = ['A', 'B']        # shortest time
    Z = ['A', 'D', 'H']   # longest time

    tic = time.perf_counter()
    freq_X = R.get_frequency(X)
    toc = time.perf_counter()
    print(f'-I- Time calculating frequencies {toc - tic} seconds')
    tic = time.perf_counter()
    freq_Y = R.get_frequency(Y)
    toc = time.perf_counter()
    print(f'-I- Time calculating frequencies {toc - tic} seconds')
    tic = time.perf_counter()
    freq_Z = R.get_frequency(Z)
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
    R = relation(path=path, name=name, num_atts=num_atts, l=l)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    S = relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True)
    toc2 = time.perf_counter()

    print(f'-I- Time creating relation without leapfrog init {toc1 - tic1} seconds')
    print(f'-I- Time creating relation with leapfrog init {toc2 - tic2} seconds')

    X = ['A', 'B', 'D']
    tic1 = time.perf_counter()
    freq_X1 = R.get_frequency(X)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    freq_X2 = S.get_frequency(X)
    toc2 = time.perf_counter()

    print(f'-I- Time processing query [\'A\', \'B\', \'D\'] without leapfrog init {toc1 - tic1} seconds')
    print(f'-I- Time processing query [\'A\', \'B\', \'D\'] with leapfrog init {toc2 - tic2} seconds')

    return True

def run_all_tests():
    tests = [test1, test2, test3, test4]
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
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\Nursery\\nursery.csv"
    name = 'Nursery'
    num_atts = 9
    l = 3

    times1 = []
    times2 = []
    N = 200
    print(f'-I- Running relation init {N} times')
    for i in range(N):
        tic1 = time.perf_counter()
        R = relation(path=path, name=name, num_atts=num_atts, l=l)
        toc1 = time.perf_counter()
        times1.append(toc1 - tic1)
        del R
        tic2 = time.perf_counter()
        S = relation(path=path, name=name, num_atts=num_atts, l=l, leapfrog_init=True)
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

if __name__ == '__main__':
    run_all_tests()