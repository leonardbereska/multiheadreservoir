# https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
from tqdm import tqdm
# from tqdm.auto import tqdm
from copy import deepcopy
from torch import nn
from torch.autograd import Variable
import torch as tc
import random
from torch import optim


def variable(t: tc.Tensor, use_cuda=True, **kwargs):
    if tc.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        criterion = nn.MSELoss()
        self.model.eval()
        for sequence in self.dataset:
            sequence = sequence.unsqueeze(0)
            sequence = variable(sequence)
            target = sequence[:, 1:]
            input = sequence[:, :-1]
            self.model.zero_grad()
            output = self.model(input)
            # label = output.max(1)[1].view(-1)
            # loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss = criterion(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0.
        for n, p in model.named_parameters():
            loss += (self._precision_matrices[n] * (p - self._means[n]) ** 2).sum()
        return loss


def train(model, optimizer, data_loader, loss_function):
    model.train()
    for sequences in data_loader:
        for _ in range(args.n_reps):
            input = sequences[:, :-1]
            target = sequences[:, 1:]
            optimizer.zero_grad()
            output = model(input)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
    return model


from lstm import get_squared_errors, print_stats, LSTM, get_args, get_loader


def run_test(args, models):
    envs = (0, 1, 2, 3)
    for env_idx in envs:
        print('\tEnv {}'.format(env_idx))
        test_loader = get_loader(args, envs=(env_idx,), test=True)
        all_errors = []
        for model in models:
            all_errors.append(get_squared_errors(model, test_loader, n_steps=1))
        errors = tc.stack(all_errors)
        print_stats(errors)


def ewc_process(args):
    importance = 1000
    n_samples = 200
    model = LSTM(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    envs = (0, 1, 2, 3)
    train_loaders = [get_loader(args, envs=(env_idx,)) for env_idx in envs]
    mse_loss = nn.MSELoss()

    def get_previous_envs(env_idx):
        previous_envs = []
        for previous_env in range(env_idx):
            previous_envs = previous_envs + train_loaders[previous_env].dataset.get_sample(n_samples)
        return random.sample(previous_envs, k=n_samples)

    def get_ewc_loss(model, env_idx):
        previous_envs = get_previous_envs(env_idx)
        ewc = EWC(model, previous_envs)

        def mse_with_ewc(output, target):
            ewc_loss = importance * ewc.penalty(model)
            return mse_loss(output, target) + ewc_loss

        return mse_with_ewc

    def get_mse_or_ewc(model, env_idx):
        if env_idx == 0:
            return mse_loss
        else:
            return get_ewc_loss(model, env_idx)

    def get_loss_function(model, env_idx, args):
        if args.use_ewc:
            return mse_loss
        else:
            return get_mse_or_ewc(model, env_idx)


    models = []
    for _ in range(args.n_exps):
        for env_idx in range(len(envs)):
            mse_or_ewc_loss = get_loss_function(model, env_idx, args)
            for _ in range(args.n_epochs):
                model = train(model, optimizer, train_loaders[env_idx], mse_or_ewc_loss)
        models.append(model)
    run_test(args, models)


def get_dataset_dimension(data_path):
    import numpy as np
    dataset = np.load(data_path)
    return dataset.shape[-1]


def process_datasets(func, args):
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    for dataset in datasets:
        print(dataset)
        args.data_path = 'datasets/{}.npy'.format(dataset)
        dim = get_dataset_dimension(args.data_path)
        args.n_output = dim
        args.n_input = dim
        func(args)


if __name__ == "__main__":
    tc.manual_seed(0)
    args = get_args()

    print('Continual LSTM+EWC')
    args.use_ewc = True
    process_datasets(ewc_process, args)

    # print('Continual LSTM')
    # args.use_ewc = False
    # process_datasets(ewc_process, args)
