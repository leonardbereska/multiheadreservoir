import datetime
import argparse
import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils import load_data


class Sequence(Dataset):

    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length + 1]
        sequence = tc.Tensor(sequence)
        return sequence

    def __len__(self):
        return len(self.data) - self.seq_length - 1


def collate_sequence(batch):
    inputs = tc.stack([item[:-1] for item in batch], dim=1)
    targets = tc.stack([item[1:] for item in batch], dim=1)
    return inputs, targets


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(args.n_input, args.n_hidden)
        self.linear = nn.Linear(args.n_hidden, args.n_output)

    def forward(self, x):
        output, _ = self.lstm(x)
        predictions = self.linear(output)
        return predictions


def train(args):
    data = load_data(args.data_path, envs=(0, 1), trajs=slice(0, 5))
    data = data.reshape(-1, data.shape[-1])
    dataset = Sequence(data, args.sequence_length)
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True, pin_memory=True, collate_fn=collate_sequence)

    args.n_output = data.shape[-1]
    args.n_input = data.shape[-1]

    model = LSTM(args).to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.n_epochs + 1):
        total_loss = 0.0
        for step, (inputs, targets) in enumerate(data_loader):
            model.train()
            optimizer.zero_grad()
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            predictions = model.forward(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

            total_loss += loss.item()

        num_batches = len(data_loader)
        log = f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}]\
                Epoch: {epoch}/{args.n_epochs}\
                Loss: {total_loss / num_batches:.2f}'
        print(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--data_path', type=str, default='datasets/lorenz63.npy')
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--n_hidden', type=int, default=1024)

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)

    args = parser.parse_args()
    args.device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    train(args)
