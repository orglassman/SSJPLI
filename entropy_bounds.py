import pandas as pd
from scipy.stats import entropy

from common import randomize_queries, convert_to_numeric
import matplotlib.pyplot as plt
from relation import Relation

class EntropyFrame():
    """
    container for results from entropy_framework method in Relation
    """
    def __init__(self, path, coverages):
        self.path = path
        self.coverages = coverages

        self.data = {
            'partial_H_sampled':[],         # partial entropy over sampled space S
            'H_sampled_normalized':[],      # normalize sampled space to distribution

            'E_exact':[],                   # (1-rho) * [ H(Q) - log(1-rho) ]
            'E1':[],                        # (1-rho) * [ log|Q| - log(1-rho) ]
            'E2':[],                        # (1-rho) * logN
            'E3':[],                        # (1-rho) * [ log(AxB - S - phi) - log(1-rho) ]
        }
        self.stats = {
            'times':[],
            'num_samples':[]
        }

        self.labels = {
            'xcoverage':'coverage %',
            'ytime':'time [sec]',
            'ysamples':'# samples',
            'yentropy':'entropy [bit/symbol]'
        }

    # def entropy(self, q):
    #     p = sorted(q)
    #     df = convert_to_numeric(pd.read_csv(self.path, usecols=p))
    #     dist = df.value_counts(normalize=True).values
    #     self.data['H'] = [entropy(dist, base=2)] * len(self.coverages)

    def update(self, data, stats):
        for key, value in data.items():
            self.data[key].append(value)

        for key, value in stats.items():
            self.stats[key].append(value)

    def plot_stats(self):
        fig, axes = plt.subplots(2,1)
        axes[0].plot(self.coverages, self.stats['times'])
        axes[0].set(xlabel=self.labels['xcoverage'], ylabel=self.labels['ytime'])
        axes[0].set_title('Sampling Time vs. Coverage')
        axes[1].plot(self.coverages, self.stats['num_samples'])
        axes[1].set(xlabel=self.labels['xcoverage'], ylabel=self.labels['ysamples'])
        axes[1].set_title('Sample Size vs. Coverage')
        plt.show()

    def plot_main(self):
        plt.figure()
        plt.xlabel('coverage %')
        for key, array in self.data.items():
            plt.plot(self.coverages, array)

        plt.legend(self.data.keys())
        plt.show()

        # # plot
        # fig, axes = plt.subplots(2, 1)
        # for d in datas:
        #     axes[0].plot(d.keys(), d.values())
        # axes[0].set(xlabel='coverage %', ylabel='entropy [bit/symbol]')
        # axes[0].set_title('Entropy vs. Coverage')
        # axes[0].legend(['true', 'sampled', 'upper1', 'upper2'])
        #
        # axes[1].plot(coverages, NSs)
        # axes[1].set(ylabel='# samples', xlabel='coverage %')
        # axes[1].set_title('Sample Size vs. Coverage')
        # fig.suptitle(r'Dataset {0}: $H\left({1}\right)$'.format(name, Qstr))
        # plt.show()


def plot_main(coverages, Hs, HSs, U1s, U2s, NSs, name, Qstr):
    # arrange data
    data1, data2, data3, data4 = {}, {}, {}, {}
    for i,val in enumerate(coverages):
        data1[val] = Hs[i]
        data2[val] = HSs[i]
        data3[val] = U1s[i]
        data4[val] = U2s[i]

    # sort keys
    datas = [data1, data2, data3, data4]
    for i,d in enumerate(datas):
        datas[i] = dict(sorted(d.items(), key=lambda item: item[0]))



def estimate_entropy_main():


    # population_sizes = range(1, 241)
    # queries = randomize_queries(R.get_attributes(), N=500)
    # coverages = [x/100 for x in range(1,100)]
    # HF = EntropyFrame(path=csv1, coverages=[1])

    # build database
    path = "C:\\Users\\orgla\\Desktop\\Study\\J_Divergence_ST_formulation\\Datasets\\School_Results\\school_results.csv"
    R = Relation(path=path, coverage=coverage)
    X = config.letter['X']
    data, stats = R.entropy_framework(X, coverage=1)

    # evaluate data
    # for c in coverages:
    #     data, stats = R.entropy_framework(Q, c)
    #     HF.update(data, stats)
    # HF.entropy(Q)
    # HF.plot_stats()
    print('hello')

if __name__ == '__main__':
    estimate_entropy_main()