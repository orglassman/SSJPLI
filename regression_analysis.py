"""
designated set of tests to help maintainability
"""
import time

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

def run_all_tests():
    tests = [test1, test2, test3]
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

if __name__ == '__main__':
    run_all_tests()