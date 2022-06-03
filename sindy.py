import argparse
from tqdm import tqdm
import torch as tc
import torch.nn as nn
import torch.optim as optim
from scipy.stats import sem
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from utils import load_data
import random
import numpy as np

import pysindy as ps


def plot(model, sequences):
    n_steps = 200
    sequence = sequences[0:1]
    predictions_one_step = model.simulate(sequence)
    predictions_n_steps = predict_n_steps(model, sequence, n_steps=n_steps)
    plot_sequence(sequence[0], 'sequences')
    plot_sequence(to_numpy(predictions_one_step)[0], 'reconstruction')
    plot_sequence(to_numpy(predictions_n_steps)[0], 'prediction')
    plt.legend()
    plt.show()
    plt.close()


def predict_n_steps(model, inputs, n_steps):
    n_steps_init = 10
    inputs = inputs[:, :n_steps_init]
    predictions = []
    for _ in range(n_steps):
        predictions_next_step = model.forward(inputs)
        next_step = predictions_next_step[0, -1].detach()
        predictions.append(next_step)
        inputs = tc.cat((inputs[0, 1:], next_step.unsqueeze(0))).unsqueeze(0)
    return tc.stack(predictions).unsqueeze(0)


def plot_sequence(data, label=None):
    if label is None:
        plt.plot(data[:, 0], data[:, -1])
    else:
        plt.plot(data[:, 0], data[:, -1], label=label)


class Sequence(Dataset):

    def __init__(self, data):
        self.data = tc.FloatTensor(data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get_sample(self, sample_size):
        sample_idx = random.choices(range(len(self)), k=sample_size)
        return [sequence for sequence in self.data[sample_idx]]



def get_loader(args, envs=(0, 1, 2, 3), shuffle=False, test=False):
    if test:
        trajs = slice(7, 10)
    else:
        trajs = slice(0, 7)
    dataset = Sequence(load_data(args.data_path, envs=envs, trajs=trajs))
    return DataLoader(dataset, args.batch_size, shuffle=shuffle)


def train_individually(args):
    print('Single-Task SINDy')
    for env_idx in (0, 1, 2, 3):
        train_loader = get_loader(args, envs=(env_idx,), shuffle=False)
        models = train(args, train_loader)
        evaluate(args, models, envs=(env_idx,))

#
# def train_continually(args):
#     print('Continual LSTM')
#     train_loader = get_loader(args, shuffle=False)
#     models = train(args, train_loader)
#     evaluate(args, models)


def train_jointly(args):
    args.n_epochs = args.n_reps
    args.n_reps = 1
    print('Multi-task SINDy')
    train_loader = get_loader(args, shuffle=True)
    models = train(args, train_loader)
    evaluate(args, models)


def train(args, train_loader):
    models = []
    for _ in range(args.n_exps):
        models.append(train_model(args, train_loader))
    return models


def train_model(args, train_loader):
    trainset = list(train_loader)
    # X = tc.cat(trainset, axis=1).squeeze().numpy()
    X = trainset[0].squeeze().numpy()

    model = ps.SINDy()
    t = np.arange(0, X.shape[0]*0.05, 0.05)

    model.fit(X, t=t)
    return model


def evaluate(args, models, envs=(0, 1, 2, 3)):
    for env_idx in envs:
        print('\tEnv {}'.format(env_idx))
        test_loader = get_loader(args, (env_idx,), test=True)
        squared_errors = []
        for model in models:
            squared_errors.append(get_squared_errors(model, test_loader))
        squared_errors = tc.stack(squared_errors)
        print_stats(squared_errors)


def get_squared_errors(model, data_loader):
    squared_errors = []
    n_steps = 1
    for step, sequences in enumerate(data_loader):
        targets = sequences[:, n_steps:]
        inputs = sequences[:, :-n_steps]
        t_step = np.arange(0, 0.05 + 0.05, 0.05)
        predictions = []
        for i in range(inputs.shape[1]):
            next_step = model.simulate(inputs[0, i], t=t_step)[-1]
            predictions.append(next_step)
        outputs = np.stack(predictions)

        squared_errors.append(tc.pow(targets - outputs, 2))
    return tc.stack(squared_errors).detach()


def print_stats(errors):
    mean = tc.mean(errors)
    standard_error_mean = sem(errors, axis=None)
    median = tc.median(errors)
    median_minus_one_sd = tc.quantile(errors, .5 - .6827 / 2)
    median_plus_one_sd = tc.quantile(errors, .5 + .6827 / 2)
    print_out = (mean, standard_error_mean, median, median_minus_one_sd, median_plus_one_sd)
    print('\t\tmean {:.2e} ± {:.1e}, \tmedian {:.2e}, [{:.2e}, {:.2e}])'.format(*print_out))


def to_numpy(tensor):
    return tensor.detach().numpy()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/Lorenz-96.npy')
    parser.add_argument('--n_hidden', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n_reps', type=int, default=500)
    parser.add_argument('--n_exps', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)
    parser.add_argument('--print_while_training', type=bool, default=False)
    args = parser.parse_args()
    args.device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    return args


def get_dataset_dimension(data_path):
    dataset = np.load(data_path)
    return dataset.shape[-1]


if __name__ == "__main__":
    tc.manual_seed(0)
    np.random.seed(0)
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']

    from datetime import datetime

    begin = datetime.now()
    for dataset in datasets:
        args = get_args()
        print(dataset)
        args.data_path = 'datasets/{}.npy'.format(dataset)
        dim = get_dataset_dimension(args.data_path)
        args.n_output = dim
        args.n_input = dim

        train_individually(args)
        print(args)
        train_jointly(args)
        print(args)

    total_time = datetime.now() - begin
