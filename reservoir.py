import argparse
import numpy as np
import torch as tc
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg
from sklearn.linear_model import Ridge


class Reservoir:
    def __init__(self, args):
        self.radius = args.radius
        self.reservoir_size = args.reservoir_size
        self.sparsity = args.sparsity
        self.input_dim = args.input_dim
        self.weights_hidden = self.get_sparse_weights()
        self.weights_input = self.get_input_weights()
        self.weights_output = np.zeros((self.input_dim, self.reservoir_size))
        self.regularization = args.regularization
        self.n_steps_prerun = args.n_steps_prerun

        self.A = np.zeros((self.reservoir_size, self.reservoir_size))
        self.B = np.zeros((self.reservoir_size, self.input_dim))

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

    # def train(self, data):
    #     hidden = self.initialize_hidden(data)
    #
    #     # run with teacher forcing (taking ground truth as input)
    #     hidden_states = []
    #     targets = []
    #     for t in range(self.n_steps_prerun, len(data) - 1):
    #         input = np.reshape(data[t], (-1, 1))
    #         target = np.reshape(data[t + 1], (-1, 1))
    #         hidden = np.tanh(self.weights_hidden @ hidden + self.weights_input @ input)
    #         hidden = self.augment_hidden(hidden)
    #         hidden_states.append(hidden)
    #         targets.append(target)
    #     hidden_states = np.squeeze(np.array(hidden_states))
    #     targets = np.squeeze(np.array(targets))
    #
    #     ridge = Ridge(alpha=self.regularization, fit_intercept=False, normalize=False, copy_X=True)
    #     ridge.fit(hidden_states, targets)
    #     self.weights_output = ridge.coef_

    # def update_weights(self):
    #     ridge = Ridge(alpha=self.regularization, fit_intercept=False, normalize=False, copy_X=True)
    #     ridge.fit(hidden_states, targets)
    #     self.weights_output = tc.matmul(tc.inverse(self.A + self.regularization * tc.eye()), self.B)
    def update_AB(self, new_X, new_Y):
        self.A = self.A + new_X.T @ new_X
        self.B = self.B + new_X.T @ new_Y

    def update_weights(self):
        self.weights_output = (np.linalg.inv(self.A + self.regularization * np.eye(self.reservoir_size)) @ self.B).T

    def train(self, data):
        assert len(data.shape) == 3  # shape: sequences, time steps, dimensions
        for sequence in data:
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
    # data = data[:2, :]
    data = data[2:4, :]
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

