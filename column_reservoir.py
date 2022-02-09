import argparse
import numpy as np
import torch as tc
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg


class Reservoir:
    def __init__(self, args):
        self.radius = args.radius
        self.reservoir_size = args.reservoir_size
        self.sparsity = args.sparsity
        self.input_dim = args.input_dim
        self.weights_hidden = self.get_sparse_weights()
        self.weights_input = self.get_input_weights()
        self.regularization = args.regularization
        self.n_steps_prerun = args.n_steps_prerun
        self.n_prediction_heads = 10
        self.heads_deviation = np.zeros(self.n_prediction_heads)
        self.heads_active = np.ones(self.n_prediction_heads)

        self.weights_output = 0.01 * np.random.rand(self.n_prediction_heads, self.input_dim, self.reservoir_size)

        self.A = np.zeros((self.n_prediction_heads, self.reservoir_size, self.reservoir_size))
        self.B = np.zeros((self.n_prediction_heads, self.reservoir_size, self.input_dim))

    def get_sparse_weights(self):
        weights = sparse.random(self.reservoir_size, self.reservoir_size, density=self.sparsity)
        eigenvalues, _ = splinalg.eigs(weights)
        return weights / np.max(np.abs(eigenvalues)) * self.radius

    def get_input_weights(self):
        weights = np.zeros((self.reservoir_size, self.input_dim))
        q = int(self.reservoir_size / self.input_dim)
        for i in range(0, self.input_dim):
            weights[i * q:(i + 1) * q, i] = 2 * np.random.rand(q) - 1
        return weights

    def initialize_hidden(self, sequence):
        # pre-running on time series
        hidden = np.zeros((self.reservoir_size, 1))
        for t in range(self.n_steps_prerun):
            input = sequence[t].reshape(-1, 1)
            hidden = np.tanh(self.weights_hidden @ hidden + self.weights_input @ input)
        return hidden

    def augment_hidden(self, hidden):
        h_aug = hidden.copy()
        h_aug[::2] = pow(h_aug[::2], 2.0)
        return h_aug

    def activate_heads(self, hidden_states, targets):
        squared_deviation = np.power(np.einsum('hdm,nm->hnd', self.weights_output, hidden_states) - targets, 2)
        self.heads_deviation = np.mean(squared_deviation, axis=(1, 2))  # average over n samples and all dimensions
        mask = np.zeros(self.n_prediction_heads)
        idx = np.argsort(self.heads_deviation)[:3]
        mask[idx] = 1
        self.heads_active = mask
        print(self.heads_active)

    def update_AB(self, X, Y):
        # Federated Reservoir Computing - Bacciu et al. 2021
        self.A = self.A + np.einsum('a,ij->aij', self.heads_active, X.T @ X)
        self.B = self.B + np.einsum('a,ij->aij', self.heads_active, X.T @ Y)

    def update_weights(self):
        for head, (A, B) in enumerate(zip(self.A, self.B)):
            self.weights_output[head] = (np.linalg.inv(A + self.regularization * np.eye(self.reservoir_size)) @ B).T

    def train(self, data):
        assert len(data.shape) == 3  # shape: sequences, time steps, dimensions
        for idx, sequence in enumerate(data):
            hidden = self.initialize_hidden(sequence)
            hidden_states = []
            targets = []
            for t in range(self.n_steps_prerun, len(sequence) - 1):
                input = np.reshape(sequence[t], (-1, 1))
                target = np.reshape(sequence[t + 1], (-1, 1))
                hidden = np.tanh(self.weights_hidden @ hidden + self.weights_input @ input)
                hidden = self.augment_hidden(hidden)
                hidden_states.append(hidden)
                targets.append(target)

            hidden_states = np.squeeze(np.array(hidden_states))
            targets = np.squeeze(np.array(targets))

            if idx == 0:
                self.heads_active = np.ones(self.n_prediction_heads)
            else:
                self.activate_heads(hidden_states, targets)
            if idx == 10:
                print("10")
            if idx == 20:
                print("20")
            self.update_AB(hidden_states, targets)
            self.update_weights()

    def predict(self, sequence, n_steps_prediction):
        hidden = self.initialize_hidden(sequence)
        hidden_states = []
        input = np.reshape(sequence[0], (-1, 1))
        for t in range(self.n_steps_prerun, self.n_steps_prerun + n_steps_prediction):  # len data -1
            hidden = np.tanh(self.weights_hidden @ hidden + self.weights_input @ input)
            hidden = self.augment_hidden(hidden)
            hidden_states.append(hidden)
            input = self.weights_output @ hidden
        hidden_states = np.squeeze(np.array(hidden_states))
        output = np.array([self.weights_output @ hidden for hidden in hidden_states])
        return output


if __name__ == '__main__':
    args = argparse.Namespace()
    args.radius = 0.6
    args.sparsity = 0.01
    args.input_dim = 3
    args.reservoir_size = 1000
    args.n_steps_prerun = 5
    args.regularization = 1e-6
    env = 1
    test_idx = 0

    res = Reservoir(args)

    data = np.load('datasets/lorenz63.npy')
    assert len(data.shape) == 4  # environments, sequences, time steps, dimensions
    assert args.input_dim == data.shape[-1]
    print(data.shape)
    data = data[0:3, :]
    print(data.shape)
    data = np.concatenate(data, axis=0)  # concatenate all environments
    print(data.shape)
    res.train(data)
    sequence = data[0]

    predictions = res.predict(sequence, n_steps_prediction=100)

    plt.plot(sequence[:, 0], sequence[:, 2], label="sequence")
    sequence = data[10]
    plt.plot(sequence[:, 0], sequence[:, 2], label="sequence")
    plt.plot(predictions[:, 0], predictions[:, 2], label="predictions")
    plt.legend()
    plt.show()
