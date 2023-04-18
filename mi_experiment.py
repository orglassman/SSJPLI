from argparse import ArgumentParser
from itertools import product
from math import factorial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mutual_info_score as mi
from scipy.optimize import fsolve
from itertools import cycle

from cubic import solve


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-card', help='A,B cardinality')

    args = parser.parse_args()
    return args


def load_data(card=10):
    A = list(range(card))
    B = list(range(card))

    df = pd.DataFrame(list(product(A, B)))
    df.columns = ['A', 'B']

    return df


def run_experiment(card=10):
    df = load_data(card=card)

    Is, cardAs, cardBs = [], [], []
    for i in range(df.shape[0]):
        # repeat for 100 steps
        for repeat in range(100):
            drop_indices = np.random.choice(df.index, i, replace=False)
            df_subset = df.drop(drop_indices)

            cardA = len(df_subset['A'].value_counts())
            cardB = len(df_subset['B'].value_counts())
            I = mi(df_subset['A'], df_subset['B'])

            Is.append(I)
            cardAs.append(cardA)
            cardBs.append(cardB)

        # print(f'-I- {i} tuples dropped; |A|={cardA}; |B|={cardB}; I(A;B)={I}')

    data = {
        'card_A': cardAs,
        'card_B': cardBs,
        'I': Is
    }

    return pd.DataFrame(data)


def aggregate_I(data):
    pairs = data[['card_A', 'card_B']].value_counts().index.to_flat_index()
    averages = {p: [] for p in pairs}
    for t in data.itertuples():
        key = (t.card_A, t.card_B)
        averages[key].append(t.I)

    for k, v in averages.items():
        total = sum(v)
        l = len(v)
        averages[k] = total / l

    return averages


def mi_experiment_main():
    args = parse_args()
    card = int(args.card)

    data = run_experiment(card=card)

    # average by (A, B) cardinalities
    averages = aggregate_I(data)

    z = np.zeros((card, card))
    for k, v in averages.items():
        x_coordinate = k[0] - 1
        y_coordinate = k[1] - 1
        z[x_coordinate][y_coordinate] = v

    fig, ax = plt.subplots()
    c = ax.imshow(z, cmap='RdBu')
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    fig.colorbar(c, ax=ax)

    plt.show()

    print('hello')


def plot_scan(Is, Is_approx):
    plt.figure()
    plt.plot(Is)
    plt.plot(Is_approx)
    plt.xlabel('Iteration')
    plt.ylabel('I(A;B) [bit/symbol]')
    plt.legend(['true', 'approx.'])
    plt.title(r'$I(A;B)$ Approximation')
    plt.show()


def taylor_e(x, n):
    res = 0
    for i in range(n):
        res += x**i / factorial(i)

    return res


def scan_I():
    args = parse_args()
    alpha = int(args.card)
    beta = alpha

    Is = [x / 10 for x in range(50)]
    Is_approx = []
    for i in Is:

        func = lambda tau: np.exp(i * tau / beta) - beta/tau
        tau_initial_guess = alpha + beta
        tau_solution = fsolve(func, tau_initial_guess)[0]
        cond_beta = np.int_(tau_solution)
        if cond_beta >= beta:
            print('-I- A,B independent, I(A;B)=0')
            continue
        try:
            PA, PB, PAB, I_approx = build_distributions(alpha, beta, cond_beta)
        except:
            print(f'-E- cond_beta={cond_beta}, DB not fully covered. Skipping')
            continue

        Is_approx.append(I_approx)

        print(f'-I- For desired I(A;B)={i}, cond_beta={cond_beta}; Approximated I {I_approx}')

    plot_scan(Is, Is_approx)
    print('-I- Finished')


def get_cond_beta(beta, I):
    a = (I / beta) ** 2
    b = (6 * I - I ** 2) / beta
    c = 6 * I + 12
    d = -12 * beta
    roots = solve(a, b, c, d)

    for root in roots:
        cond_beta = int(np.floor(np.real(root)))
        if cond_beta <= 0:
            continue

        I_approx = beta / cond_beta * np.log(beta / cond_beta)
        return cond_beta, I_approx

        # a = (i / beta) ** 2
        # b = (6 * i - i ** 2) / beta
        # c = 12 + 6 * i
        # d = -12 * beta

        # roots = solve(a, b, c, d)

        i_approx = tau_solution





        # for root in roots:
        #     cond_beta = np.floor(np.real(root))
        #     if cond_beta <= 0:
        #         continue
        #
        #     try:
        #         i_approx = beta / cond_beta * np.log(beta / cond_beta)
        #     except:
        #         print('hello')

def get_marginals(PAB):
    PA = {}
    PB = {}

    for k, v in PAB.items():
        a = k[0]
        b = k[1]

        if a in PA.keys():
            PA[a] += v
        else:
            PA[a] = v
        if b in PB.keys():
            PB[b] += v
        else:
            PB[b] = v

    return PA, PB

def compute_I(PA, PB, PAB):
    res = 0
    for a in PA:
        for b in PB:
            pa = PA[a]
            pb = PB[b]
            try:
                pab = PAB[(a, b)]
            except:
                # print(f'-I- Symbol {(a, b)} not in DAB')
                continue

            res += pab * np.log(pab / (pa * pb))

    return res

def build_distributions(alpha, beta, cond_beta):
    """
    build P(AB) over domains of size alpha, beta
        s.t.
    I(A;B) = I (approximately)
    """
    DA = np.arange(alpha)
    DB = np.arange(beta)

    # need to cover beta with conditional betas
    covered_b = {}
    PAB = {}

    pool = cycle(DB)    # circular iterator

    for a in DA:
        cond_bs = []

        for i in range(cond_beta):
            b = next(pool)
            cond_bs.append(b)
            covered_b[b] = 1

            symbol = (a, b)
            P = (1 / alpha) * (1 / cond_beta)
            PAB[symbol] = P

    PA, PB = get_marginals(PAB)
    I_final = compute_I(PA, PB, PAB)

    return PA, PB, PAB, I_final


if __name__ == '__main__':
    scan_I()
