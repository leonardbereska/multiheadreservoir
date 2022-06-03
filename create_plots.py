import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def name_to_path_name(name):
    return name.lower().replace('-', '_').replace(' ', '_')


def get_range_for_dataset(dataset):
    if dataset == 'VdP':
        range = slice(0, 4)
    elif dataset == 'L63':
        range = slice(4, 8)
    else:
        range = slice(8, 12)
    return range


def bar_plot(methods_to_plot, dataset, save_name=None):
    columns = ['mean', 'standard_error_mean', 'median', 'median_minus_one_sd', 'median_plus_one_sd']
    bar_width = 1 / len(methods_to_plot) - 1 / (len(methods_to_plot) + 1) / 4
    r = np.arange(3) + 0.7
    for method in methods_to_plot:
        dataset_range = get_range_for_dataset(dataset)
        df = pd.read_csv('results/{}.csv'.format(name_to_path_name(method)), names=columns)[dataset_range]
        plt.bar(r, df['mean'][0:3], width=bar_width, yerr=df['standard_error_mean'][0:3], label=method, capsize=2)
        r = [x + bar_width for x in r]
    plt.legend()
    plt.xticks([1, 2, 3])
    plt.xlabel('Environment')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title(dataset)
    if save_name is not None:
        plt.savefig('plots/barplots/{}-{}.pdf'.format(save_name, dataset))
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    datasets = ['VdP', 'L63', 'L96']

    continual = ['Continual LSTM-EWC', 'Continual Ours', 'Continual LSTM-Replay']
    for dataset in datasets:
        bar_plot(continual, dataset, save_name='continual')

    bounds = ['Single-Task LSTM', 'Single-Task RC', 'Continual Ours', 'Multi-Task LSTM', 'Multi-Task RC']
    for dataset in datasets:
        bar_plot(bounds, dataset, save_name='bounds')

