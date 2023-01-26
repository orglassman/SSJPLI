import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import baseline_utilities as bu


#TODO: see if this can go in common
def make_boxplots(data):
    fig, ax1 = plt.subplots()

    box_plot_color(ax1, data['PLI'], 'red', 'tan')
    box_plot_color(ax1, data['PROJECT'], 'blue', 'cyan')
    box_plot_color(ax1, data['PS'], 'green', 'red')

    # plt.boxplot([x for x in X.values()], positions=range(len(X.keys())))
    # ax1.set_xticklabels(X.keys())
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.show()

def box_plot_color(ax, data, edge_color, fill_color):
    bp = ax.boxplot([x for x in data.values()], positions=range(1,len(data.keys())+1), patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

    return bp


#TODO: format logger
logger = logging.getLogger()
# format = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
# logging.basicConfig(format=format)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-cfg', help='path to cfg file')
    parser.add_argument('-out_dir', help='path to output directory')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    bu.run_baselines(args)

if __name__ == '__main__':
    main()
