import functools
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class A:
    def __init__(self):
        self._TIDs = {
            1: ['a', 'b', 'c'],
            2: ['d', 'e', 'f'],
            3: ['g', 'h', 'i'],
        }
        self.init_routine()

    def init_routine(self):
        execables = (
            functools.partial(self.subset_routine, subset, tids)
            for subset, tids in self._TIDs.items()
        )

        subset_res = {}
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execable) for execable in execables]
            for ftr in futures:
                subset, res = ftr.result()
                subset_res[subset] = res

    def subset_routine(self, subset, tids):
        attr = tids.keys()
        powerset = calc_powerset(attr, skip_empty=True)

        execables = (
            functools.partial(self.powerset_routine, tids, ps)
            for ps in powerset
        )

        res = {}
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(execable) for execable in execables]
            for ftr in futures:
                name, lfj = ftr.result()
                res[name] = lfj

        return subset, res

    def powerset_routine(self, tids, powerset):
        powerset = sorted(list(powerset))
        cover, target = self.get_target_tids(powerset, tids)
        lfj = self.LFJ(cover, target)
        return ','.join(cover), lfj


def calc_powerset(sequence, skip_empty=False):
    start = 0
    if skip_empty:
        start += 1
    nrange = range(start, len(sequence) + 1)
    g = itertools.chain.from_iterable(itertools.combinations(sequence, n) for n in nrange)
    yield from g