import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg


def load_data(data_path, environments, trajectories):
    dataset = np.load(data_path)
    assert len(dataset.shape) == 4  # environments, sequences, time steps, dimensions
    assert args.input_dim == dataset.shape[-1]
    data_selection = dataset[environments, trajectories]
    data = np.concatenate(data_selection, axis=0)  # concatenate all environment sequences
    assert len(data.shape) == 3
    return data


def plot_pred_vs_target(predictions, targets):
    plt.plot(targets[:, 0], targets[:, 2], label='target')
    plt.scatter(targets[0, 0], targets[0, 2], label='start target')
    plt.plot(predictions[:, 0], predictions[:, 2], label='predictions')
    plt.scatter(predictions[0, 0], predictions[0, 2], label='start prediction')
    plt.legend()
    plt.show()


def plot_multiple_heads_and_target(outputs, targets):
    plt.plot(targets[:, 0], targets[:, 2], label='target')
    plt.scatter(targets[0, 0], targets[0, 2], label='start target')
    for predictions in outputs:
        plt.plot(predictions[:, 0], predictions[:, 2], label='predictions')
        plt.scatter(predictions[0, 0], predictions[0, 2], label='start prediction')
    plt.legend()
    plt.show()


class Reservoir:
    def __init__(self, args):
        self.radius = args.radius
        self.reservoir_size = args.reservoir_size
        self.sparsity = args.sparsity
        self.input_dim = args.input_dim
        self.regularization = args.regularization
        self.n_steps_prerun = args.n_steps_prerun

        self.n_prediction_heads = args.n_prediction_heads
        self.n_best_predictions = args.n_best_predictions
        self.activation_threshold = args.activation_threshold
        self.heads_active = np.zeros(self.n_prediction_heads)
        self.never_active_before_heads = np.ones(self.n_prediction_heads)

        self.weights_input = self.get_input_weights()
        self.weights_hidden = self.get_sparse_weights()

        self.weights_output = 0.001 * np.random.rand(self.n_prediction_heads, self.input_dim, self.reservoir_size)
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

    def forward_hidden(self, hidden, input):
        return np.tanh(self.weights_hidden @ hidden + self.weights_input @ input)

    def initialize_hidden(self, sequence):
        assert len(sequence) == self.n_steps_prerun
        hidden = np.zeros((self.reservoir_size, 1))
        for t in range(self.n_steps_prerun):
            input = sequence[t].reshape(-1, 1)
            hidden = self.forward_hidden(hidden, input)
        return hidden

    def augment_hidden(self, hidden):
        hidden_augmented = hidden.copy()
        hidden_augmented[::2] = pow(hidden_augmented[::2], 2.0)
        return hidden_augmented

    def initialize_head(self, sequence):
        hidden_states, targets = self.train_sequence(sequence)
        self.activate_heads(hidden_states, targets)
        return np.argmax(self.heads_active)

    def prediction_mse(self, hidden_states, targets):
        outputs = np.einsum('hdm,tm->htd', self.weights_output, hidden_states)
        return np.mean(np.power(outputs - targets, 2), axis=(1, 2))  # mean over dimensions

    def remember_active_heads(self):
        self.never_active_before_heads *= (1 - self.heads_active)

    def activate_random_heads_not_active_before(self):
        heads_not_active_before_indices = np.where(self.never_active_before_heads)[0]
        heads_active_indices = np.random.choice(heads_not_active_before_indices, self.n_best_predictions, replace=False)
        heads_active = np.zeros(self.n_prediction_heads)
        heads_active[heads_active_indices] = 1
        self.heads_active = heads_active

    def activate_best_prediction_heads(self, mse):
        heads_active = np.zeros(self.n_prediction_heads)
        best_prediction_heads_indices = np.argsort(mse)[:self.n_best_predictions]
        heads_active[best_prediction_heads_indices] = 1
        self.heads_active = heads_active

    def activate_heads(self, hidden_states, targets):
        mse = self.prediction_mse(hidden_states, targets)
        # print('mse {}'.format(np.array_str(mse, precision=2)))
        if np.all(mse > self.activation_threshold):
            self.activate_random_heads_not_active_before()
        else:
            self.activate_best_prediction_heads(mse)
        self.remember_active_heads()

    def update_AB(self, X, Y):
        # Federated Reservoir Computing - Bacciu et al. 2021
        self.A = self.A + np.einsum('a,ij->aij', self.heads_active, X.T @ X)
        self.B = self.B + np.einsum('a,ij->aij', self.heads_active, X.T @ Y)

    def update_output_weights(self):
        for head, (A, B) in enumerate(zip(self.A, self.B)):
            self.weights_output[head] = (np.linalg.inv(A + self.regularization * np.eye(self.reservoir_size)) @ B).T

    def train_sequence(self, sequence):
        hidden = self.initialize_hidden(sequence[:self.n_steps_prerun])
        hidden_states = []
        for t in range(self.n_steps_prerun, len(sequence) - 1):
            input = sequence[t:t + 1].T
            hidden = self.augment_hidden(self.forward_hidden(hidden, input))
            hidden_states.append(hidden)
        hidden_states = np.squeeze(np.array(hidden_states))
        targets = sequence[self.n_steps_prerun + 1:]
        return hidden_states, targets

    def train(self, data):
        assert len(data.shape) == 3  # shape: sequences, time steps, dimensions
        for idx, sequence in enumerate(data):
            hidden_states, targets = self.train_sequence(sequence)
            self.activate_heads(hidden_states, targets)
            self.update_AB(hidden_states, targets)
            self.update_output_weights()

    def predict(self, sequence, n_steps_predict):
        hidden = self.initialize_hidden(sequence[:self.n_steps_prerun])
        input = sequence[self.n_steps_prerun].reshape((-1, 1))
        head = self.initialize_head(sequence[:2 * self.n_steps_prerun])  # one init hidden, one init head
        outputs = []

        for t in range(self.n_steps_prerun, self.n_steps_prerun + n_steps_predict):
            hidden = self.augment_hidden(self.forward_hidden(hidden, input))
            output = self.weights_output[head] @ hidden
            input = output
            outputs.append(output)
        return np.array(outputs)

    def plot_predictions(self, sequence, n_steps_predict, title=None):
        predictions = self.predict(sequence, n_steps_predict=n_steps_predict)
        targets = sequence[args.n_steps_prerun + 1:]
        plt.plot(targets[:n_steps_predict, 0], targets[:n_steps_predict, 2], label='sequence')
        plt.plot(predictions[:, 0], predictions[:, 2], label='prediction')
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()

    def ahead_prediction_mse(self, sequence, n_steps_ahead):
        targets = sequence[args.n_steps_prerun + n_steps_ahead:]
        predictions = []
        for t in range(len(sequence) - args.n_steps_prerun - n_steps_ahead):
            prediction = self.predict(sequence[t:], n_steps_predict=n_steps_ahead)
            # plot_pred_vs_target(prediction, targets[t:t+10])
            assert len(prediction) == n_steps_ahead
            prediction = prediction[-1]  # take nth step
            predictions.append(prediction)
        predictions = np.squeeze(np.array(predictions))
        mse = np.mean(np.power(targets - predictions, 2), axis=(0, 1))  # mean over time steps and dimensions
        return mse


if __name__ == '__main__':
    args = argparse.Namespace()
    args.radius = 0.6
    args.sparsity = 0.01
    args.input_dim = 3
    args.reservoir_size = 1000
    args.n_steps_prerun = 10
    args.regularization = 1e-6
    args.n_prediction_heads = 10
    args.n_best_predictions = 1
    args.activation_threshold = 0.01
    args.data_path = 'datasets/lorenz63.npy'

    res = Reservoir(args)
    data = load_data(args.data_path, environments=(0, 1, 2, 3), trajectories=slice(0, 5))
    res.train(data)

    n_steps_predict = 100
    sequence_env_0 = load_data(args.data_path, environments=(0,), trajectories=slice(5, 6))[0]
    res.plot_predictions(sequence_env_0, n_steps_predict, title='env 0')
    sequence_env_1 = load_data(args.data_path, environments=(1,), trajectories=slice(5, 6))[0]
    res.plot_predictions(sequence_env_1, n_steps_predict, title='env 1')
    sequence_env_2 = load_data(args.data_path, environments=(2,), trajectories=slice(5, 6))[0]
    res.plot_predictions(sequence_env_2, n_steps_predict, title='env 2')
    sequence_env_3 = load_data(args.data_path, environments=(3,), trajectories=slice(5, 6))[0]
    res.plot_predictions(sequence_env_3, n_steps_predict, title='env 3')

    n_steps_ahead_prediction = 5
    # print('{} steps ahead prediction mse {:.3f}'.format(n_steps_ahead_prediction,
    #                                                     res.ahead_prediction_mse(sequence_env_0,
    #                                                                              n_steps_ahead_prediction)))
    # print('{} steps ahead prediction mse {:.2f}'.format(n_steps_ahead_prediction,
    #                                                     res.ahead_prediction_mse(sequence_env_1,
    #                                                                              n_steps_ahead_prediction)))
