import datetime
import argparse

import torch as tc
import numpy as np
from torch import optim
import torchvision as tv
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import utils
import dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tps_scale', type=float, default=1.)
    parser.add_argument('--rot_scale', type=float, default=0.)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--data_path', type=str, default='datasets/lv.npy')
    parser.add_argument('--n_particles', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=1000)
    args = parser.parse_args()

    now = datetime.datetime.now()
    now_str = now.strftime('%y%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir=f'output_dir/runs/{now_str}')
    args.writer = writer
    tc.manual_seed(args.seed)
    args.device = tc.device("cuda" if tc.cuda.is_available() else "cpu")


    data = tc.load(args.data_path)
    n_envs = data.shape[1]

    for env in range(n_envs):
        env_dataset = data[:, env]
        env_train_loader = DataLoader(dataset.SequenceDataset(env_dataset), batch_size=args.batch_size, shuffle=True)
        for epoch in range(1, args.epochs + 1):
            for idx, sequence in enumerate(env_train_loader):




        # print('Epoch {} Loss {:.1f}'.format(epoch, train_loss / len(train_loader.dataset)))
