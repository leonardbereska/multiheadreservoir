import argparse
from tqdm import tqdm
import torch as tc
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.stats import sem
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from utils import load_data
import random
from tqdm.auto import trange, tqdm


def plot(model, sequences):
    n_steps = 200
    sequence = sequences[0:1]
    predictions_one_step = model.forward(sequence)
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


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(args.n_input, args.n_hidden)
        self.linear = nn.Linear(args.n_hidden, args.n_output)

    def forward(self, x):
        output, _ = self.lstm(x)
        predictions = self.linear(output)
        return predictions


def get_loader(args, envs=(0, 1, 2, 3), shuffle=False, test=False):
    if test:
        trajs = slice(7, 10)
    else:
        trajs = slice(0, 7)
    dataset = Sequence(load_data(args.data_path, envs=envs, trajs=trajs))
    return DataLoader(dataset, args.batch_size, shuffle=shuffle)


def train_continually(args):
    print('Continual LSTM')
    train_loader = get_loader(args, shuffle=False)
    models = train(args, train_loader)
    evaluate(args, models)


def train(args, train_loader):
    models = []

    for _ in range(args.n_exps):
        models.append(train_model(args, train_loader))
    return models


def get_task_id(idx):
    return int(idx / 7)  # seven sequences per task


def get_buffer_indices(task_id, buffer_size_per_task):
    buffer_indices = []
    for i in range(1, task_id + 1):
        for j in range(1, buffer_size_per_task + 1):
            buffer_indices.append(7 * i - j)  # take the last sequences for previous tasks
    return buffer_indices


def get_buffer(task_id, train_loader, buffer_size_per_task):
    indices = get_buffer_indices(task_id, buffer_size_per_task)
    buffer = []
    train_sequences = list(train_loader)
    for index in indices:
        buffer.append(train_sequences[index])
    return buffer


def get_inputs_and_targets_from_sequence(sequences):
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    return inputs, targets


def train_model(args, train_loader):
    model = LSTM(args).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    for _ in range(args.n_epochs):
        for sequence_idx, sequences in enumerate(train_loader):
            inputs, targets = get_inputs_and_targets_from_sequence(sequences)
            for _ in range(args.n_reps):
                task_id = get_task_id(sequence_idx)
                sample_from_buffer = np.random.rand() < args.buffer_probability and task_id > 0
                if sample_from_buffer:
                    buffer = get_buffer(task_id, train_loader, args.buffer_size_per_task)
                    buffer_sequence = random.choice(buffer)
                    inputs, targets = get_inputs_and_targets_from_sequence(buffer_sequence)
                optimizer.zero_grad()
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                optimizer.step()
    return model


def evaluate(args, models, envs=(0, 1, 2, 3), n_steps=1):
    for env_idx in envs:
        print('\tEnv {}'.format(env_idx))
        test_loader = get_loader(args, (env_idx,), test=True)
        squared_errors = []
        for model in models:
            squared_errors.append(get_squared_errors(model, test_loader, n_steps))
        squared_errors = tc.stack(squared_errors)
        print_stats(squared_errors)


def get_squared_errors(model, data_loader, n_steps):
    squared_errors = []
    for step, sequences in enumerate(data_loader):
        targets = sequences[:, n_steps:]
        inputs = sequences[:, :-n_steps]
        outputs = inputs
        for _ in range(n_steps):
            outputs = model.forward(outputs)
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
    parser.add_argument('--n_epochs', type=int, default=1)  # only one for continual schedule
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)
    parser.add_argument('--print_while_training', type=bool, default=False)

    parser.add_argument('--buffer_probability', type=float, default=0.01)
    parser.add_argument('--buffer_size_per_task', type=int, default=1)
    args = parser.parse_args()
    args.device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    return args


def get_dataset_dimension(data_path):
    import numpy as np
    dataset = np.load(data_path)
    return dataset.shape[-1]


def run():
    tc.manual_seed(0)
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    for dataset in datasets:
        args = get_args()
        print(dataset)
        args.data_path = 'datasets/{}.npy'.format(dataset)
        dim = get_dataset_dimension(args.data_path)
        args.n_output = dim
        args.n_input = dim

        train_continually(args)
        print(args)


if __name__ == "__main__":
    run()
