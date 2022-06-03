from scipy.stats import sem
import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg

import utils


class Heads:
    def __init__(self, args):
        self.n_prediction_heads = args.n_prediction_heads
        self.n_best_predictions = args.n_best_predictions
        self.heads_active = np.zeros(self.n_prediction_heads)
        self.print_while_training = args.print_while_training
        self.never_active_before_heads = np.ones(self.n_prediction_heads)

    def activate_heads(self, mse):
        self.activation_rule(mse)
        self.remember_active_heads()
        if self.print_while_training:
            self.default_print(mse)

    def activation_rule(self, mse):
        return NotImplementedError

    def activate_best_prediction_heads(self, mse):
        best_prediction_heads_indices = np.argsort(mse)[:self.n_best_predictions]
        self.activate_heads_by_indices(best_prediction_heads_indices)

    def activate_heads_by_indices(self, heads_active_indices):
        heads_active = np.zeros(self.n_prediction_heads)
        heads_active[heads_active_indices] = 1
        self.heads_active = heads_active

    def activate_random_heads_not_active_before(self, mse):
        never_before_active_heads = np.where(self.never_active_before_heads)[0]
        if list(never_before_active_heads):
            heads_active_indices = np.random.choice(never_before_active_heads, self.n_best_predictions, replace=False)
            self.activate_heads_by_indices(heads_active_indices)
        else:
            self.activate_best_prediction_heads(mse)

    def remember_active_heads(self):
        self.never_active_before_heads *= (1 - self.heads_active)

    def default_print(self, mse):
        print('mse \t{}'.format(np.array_str(mse, precision=2)))
        print('heads \t{}'.format(self.heads_active))


class HeadsThreshold(Heads):
    def __init__(self, args):
        super(HeadsThreshold, self).__init__(args)
        self.activation_threshold = args.activation_threshold

    def activation_rule(self, mse):
        if np.all(mse > self.activation_threshold):
            self.activate_random_heads_not_active_before(mse)
        else:
            self.activate_best_prediction_heads(mse)


class HeadsRatio(Heads):
    def __init__(self, args):
        super(HeadsRatio, self).__init__(args)
        self.activation_ratio = args.activation_ratio
        self.last_mse = np.ones(self.n_prediction_heads)

    def activation_rule(self, mse):
        if self.environment_change_detected(mse):
            self.activate_random_heads_not_active_before(mse)
        else:
            self.activate_best_prediction_heads(mse)
        if self.print_while_training:
            relative_mse_increase = mse / self.last_mse
            print('ratio \t{}'.format(np.array_str(relative_mse_increase, precision=2)))
            print('pool \t{}'.format(self.never_active_before_heads))
        self.last_mse = mse

    def environment_change_detected(self, mse):
        last_heads_active = self.heads_active
        relative_mse_increase = mse / self.last_mse
        return (last_heads_active * relative_mse_increase > self.activation_ratio).any()


class Reservoir:
    def __init__(self, args):
        self.radius = args.radius
        self.sparsity = args.sparsity
        self.regularization = args.regularization

        self.reservoir_size = args.reservoir_size
        self.input_dim = args.input_dim

        self.n_steps_init_hidden = args.n_steps_init_hidden
        self.n_steps_init_head = args.n_steps_init_head

        self.heads = HeadsRatio(args)

        self.weights_input = self.get_input_weights()
        self.weights_hidden = self.get_sparse_weights_without_error()

        self.weights_output = np.zeros((args.n_prediction_heads, self.input_dim, self.reservoir_size))
        self.A = np.zeros((args.n_prediction_heads, self.reservoir_size, self.reservoir_size))
        self.B = np.zeros((args.n_prediction_heads, self.reservoir_size, self.input_dim))

        self.data_path = args.data_path
        self.n_envs = args.n_envs
        self.n_trajs = args.n_trajs
        self.n_trajs_train = args.n_trajs_train
        self.n_trajs_test = args.n_trajs_test
        self.n_trajs_val = args.n_trajs_val

        self.print_while_training = args.print_while_training

    def get_sparse_weights_without_error(self):
        try:
            weights = self.get_sparse_weights()
        except:
            weights = self.get_sparse_weights_without_error()
        return weights

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

    def augment_hidden(self, hidden):
        hidden_augmented = hidden.copy()
        hidden_augmented[::2] = pow(hidden_augmented[::2], 2.0)
        return hidden_augmented

    def initialize_hidden(self, sequence):
        hidden = np.zeros((self.reservoir_size, 1))
        for t in range(len(sequence)):
            input = sequence[t].reshape(-1, 1)
            hidden = self.forward_hidden(hidden, input)
        return hidden

    def initialize_head(self, sequence):
        assert len(sequence) == self.n_steps_init_hidden
        assert self.n_steps_init_hidden > self.n_steps_init_head
        hidden_states, targets = self.train_sequence(sequence, self.n_steps_init_hidden - self.n_steps_init_head)
        mse = self.prediction_mse(hidden_states, targets)
        self.heads.activate_best_prediction_heads(mse)
        return np.argmax(self.heads.heads_active)

    def prediction_mse(self, hidden_states, targets):
        outputs = np.einsum('hdm,tm->htd', self.weights_output, hidden_states)
        return np.mean(np.power(outputs - targets, 2), axis=(1, 2))  # mean over dimensions

    def update_AB(self, X, Y):
        # Federated Reservoir Computing - Bacciu et al. 2021
        self.A = self.A + np.einsum('a,ij->aij', self.heads.heads_active, X.T @ X)
        self.B = self.B + np.einsum('a,ij->aij', self.heads.heads_active, X.T @ Y)

    def update_output_weights(self):
        for head, (A, B) in enumerate(zip(self.A, self.B)):
            self.weights_output[head] = (np.linalg.inv(A + self.regularization * np.eye(self.reservoir_size)) @ B).T

    def train_sequence(self, sequence, n_steps_init_hidden=None):
        if n_steps_init_hidden is None:
            n_steps_init_hidden = self.n_steps_init_hidden
        hidden = self.initialize_hidden(sequence[:n_steps_init_hidden])
        hidden_states = []
        for t in range(n_steps_init_hidden, len(sequence) - 1):
            input = sequence[t:t + 1].T
            hidden = self.augment_hidden(self.forward_hidden(hidden, input))
            hidden_states.append(hidden)
        hidden_states = np.squeeze(np.array(hidden_states))
        targets = sequence[n_steps_init_hidden + 1:]
        return hidden_states, targets

    def train(self, data):
        assert len(data.shape) == 3  # shape: sequences, time steps, dimensions
        true_positives_and_false_negatives = []
        total_positives_and_negatives = []
        for idx, sequence in enumerate(data):
            hidden_states, targets = self.train_sequence(sequence)
            mse = self.prediction_mse(hidden_states, targets)
            # if idx % self.n_trajs_train == 0 and idx != 0:
            if idx % self.n_trajs_val == 0 and idx != 0:
                if self.print_while_training:
                    print('new environment')
                true_positives_and_false_negatives.append(self.heads.environment_change_detected(mse))
            total_positives_and_negatives.append(self.heads.environment_change_detected(mse))

            self.heads.activate_heads(mse)

            self.update_AB(hidden_states, targets)
            self.update_output_weights()
        return true_positives_and_false_negatives, total_positives_and_negatives

    def predict(self, sequence, n_steps_predict):
        hidden = self.initialize_hidden(sequence[:self.n_steps_init_hidden])
        input = sequence[self.n_steps_init_hidden].reshape((-1, 1))
        head = self.initialize_head(sequence[:self.n_steps_init_hidden])
        outputs = []

        for t in range(self.n_steps_init_hidden, self.n_steps_init_hidden + n_steps_predict):
            hidden = self.augment_hidden(self.forward_hidden(hidden, input))
            output = self.weights_output[head] @ hidden
            input = output
            outputs.append(output)
        return np.array(outputs)

    def plot_predictions(self, sequence, n_steps_predict, title=None):
        if sequence.shape[-1] > 3:
            self.plot_predictions_high_dimensional(sequence, n_steps_predict, title)
        else:
            self.plot_predictions_low_dimensional(sequence, n_steps_predict, title)

    def plot_predictions_low_dimensional(self, sequence, n_steps_predict, title=None):
        predictions = self.predict(sequence, n_steps_predict=n_steps_predict)
        targets = sequence[self.n_steps_init_hidden + 1:]
        plt.plot(targets[:n_steps_predict, 0], targets[:n_steps_predict, -1], label='sequence')
        plt.plot(predictions[:, 0], predictions[:, -1], label='prediction')
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_predictions_high_dimensional(self, sequence, n_steps_predict, title=None):
        predictions = self.predict(sequence, n_steps_predict=n_steps_predict)
        targets = sequence[self.n_steps_init_hidden + 1:]
        fig = plt.figure()
        fig.gca().set_axis_off()
        plt.title(title)
        axes_1 = fig.add_subplot(121)
        axes_1.title.set_text('targets')
        axes_1.set_ylabel('time steps')
        axes_1.set_xlabel('dimensions')
        axes_1.imshow(targets[:len(predictions)], aspect='auto')
        plt.gca().invert_yaxis()
        axes_2 = fig.add_subplot(122)
        axes_2.title.set_text('predictions')
        axes_2.set_xlabel('dimensions')
        axes_2.imshow(predictions, aspect='auto')
        plt.gca().invert_yaxis()
        plt.yticks([])
        plt.show()

    def ahead_prediction_mse(self, sequence, n_steps_ahead):
        n_steps_buffer = self.n_steps_init_hidden + n_steps_ahead
        targets = sequence[n_steps_buffer:]
        predictions = []
        for t in range(len(sequence) - n_steps_buffer):
            assert len(sequence[t:]) > n_steps_buffer
            prediction = self.predict(sequence[t:], n_steps_predict=n_steps_ahead)
            # utils.plot_pred_vs_target(prediction, targets[t:t+10])
            assert len(prediction) == n_steps_ahead
            prediction = prediction[-1]  # take nth step
            predictions.append(prediction)
        predictions = np.squeeze(np.array(predictions))
        se = np.power(targets - predictions, 2)
        return se

    def calculate_mses(self, env_idx, n_steps_to_calculate=(1, 5, 10, 50)):
        mses = []
        for n_steps in n_steps_to_calculate:
            mses_n = []
            for traj_idx in range(self.n_trajs_train, self.n_trajs_train + self.n_trajs_test):  # eval on test set
                sequence = utils.load_data(self.data_path, envs=(env_idx,), trajs=slice(traj_idx, traj_idx + 1))[0]
                mses_n.append(self.ahead_prediction_mse(sequence=sequence, n_steps_ahead=n_steps))
            mses.append(np.array(mses_n))
        return mses


def precision_recall_accuracy(true_positives_and_false_negatives,
                              total_positives_and_negatives):
    true_positives = sum(true_positives_and_false_negatives)
    false_positives = sum(total_positives_and_negatives) - true_positives
    false_negatives = sum([not positive for positive in true_positives_and_false_negatives])
    true_negatives = sum([not positive for positive in total_positives_and_negatives])

    if (true_positives + false_positives) == 0:
        precision = None
    else:
        precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_negatives + false_positives)
    print('\tprecision {:.2f}'.format(precision))
    print('\trecall {:.2f}'.format(recall))
    print('\taccuracy {:.2f}'.format(accuracy))
    return precision, recall, accuracy


def perform_accuracy_experiments(args, data, env_idx):
    all_mses = [[] for _ in range(len(args.mse_steps))]
    all_true_positives = []
    all_total_positives = []
    for i in range(args.n_exps):
        res = Reservoir(args)
        true_positives, total_positives = res.train(data)
        all_true_positives.extend(true_positives)
        all_total_positives.extend(total_positives)
        mses = res.calculate_mses(env_idx, n_steps_to_calculate=args.mse_steps)
        for j in range(len(args.mse_steps)):
            all_mses[j].append(mses[j])
    return all_mses, all_true_positives, all_total_positives


def perform_experiments(args, data, env_idx):
    all_mses = [[] for _ in range(len(args.mse_steps))]
    for i in range(args.n_exps):
        res = Reservoir(args)
        res.train(data)
        mses = res.calculate_mses(env_idx, n_steps_to_calculate=args.mse_steps)
        for j in range(len(args.mse_steps)):
            all_mses[j].append(mses[j])
    return all_mses


def print_statistics(all_mses, n_steps_to_calculate):
    all_mses = [np.array(mse_n) for mse_n in all_mses]
    means = [np.mean(mse_n) for mse_n in all_mses]
    sems = [sem(mse_n, axis=None) for mse_n in all_mses]
    medians = [np.median(mse_n) for mse_n in all_mses]
    medians_minus_one_sd = [np.quantile(mse_n, .5 - .6827 / 2) for mse_n in all_mses]
    medians_plus_one_sd = [np.quantile(mse_n, .5 + .6827 / 2) for mse_n in all_mses]
    for i, mse_n in enumerate(n_steps_to_calculate):
        print_out = (mse_n, means[i], sems[i], medians[i], medians_minus_one_sd[i], medians_plus_one_sd[i])
        print('\t\tMSE-{} \tmean {:.3e} ± {:.1e}; median {:.1e} [{:.1e} - {:.1e}])'.format(*print_out))


def experiment_one_per_env(data_path):
    print('Single-Task RC')
    args = get_hyperparameters(data_path)
    for env_idx in range(args.n_envs):
        print('\tEnvironment {}'.format(env_idx))
        data = utils.load_data(args.data_path, envs=(env_idx,), trajs=slice(0, args.n_trajs_train))
        args.n_prediction_heads = 1
        all_mses = perform_experiments(args, data, env_idx)
        print_statistics(all_mses, args.mse_steps)


def experiment_ours(data_path):
    print('Continual Ours')
    args = get_hyperparameters(data_path)
    data = utils.load_data(args.data_path, envs=(list(range(args.n_envs))), trajs=slice(0, args.n_trajs_train))
    res = Reservoir(args)  # TODO what does this change?
    res.train(data)
    for env_idx in range(args.n_envs):
        print('\tEnvironment {}'.format(env_idx))
        all_mses = perform_experiments(args, data, env_idx)
        print_statistics(all_mses, args.mse_steps)


def experiment_standard_rc(data_path):
    print('Multi-Task RC')
    args = get_hyperparameters(data_path)
    data = utils.load_data(args.data_path, envs=(list(range(args.n_envs))), trajs=slice(0, args.n_trajs_train))
    args.n_prediction_heads = 1
    res = Reservoir(args)
    res.train(data)
    for env_idx in range(args.n_envs):
        print('\tEnvironment {}'.format(env_idx))
        all_mses = perform_experiments(args, data, env_idx)
        print_statistics(all_mses, args.mse_steps)


def get_dataset_specific_hyperparameters(data_path):
    args = argparse.Namespace()
    dataset = np.load(data_path)
    args.data_path = data_path
    args.input_dim = dataset.shape[-1]
    args.n_trajs_train = 7
    args.n_trajs_test = 3
    args.n_trajs_val = 5
    args.n_envs = dataset.shape[0]
    args.n_trajs = dataset.shape[1]
    assert args.n_trajs == args.n_trajs_train + args.n_trajs_test + args.n_trajs_val
    args.regularization = get_regularization(data_path)
    return args


def get_regularization(data_path):
    if data_path == 'datasets/Lorenz-63.npy':
        regularization = 1e-6
    elif data_path == 'datasets/Lorenz-96.npy':
        regularization = 1.  # originally used 5.
    elif data_path == 'datasets/Van-der-Pol.npy':
        regularization = 1e-6  # 1e-1
    else:
        regularization = NotImplementedError
        print('Unknown dataset, determine regularization first!')
    return regularization


def get_hyperparameters(data_path):
    args = get_dataset_specific_hyperparameters(data_path)
    args.regularization = get_regularization(data_path)
    args.radius = 0.6
    args.sparsity = 0.01
    args.reservoir_size = 1000
    args.n_steps_init_hidden = 10
    args.n_steps_init_head = 5
    args.regularization = 0.1
    args.n_prediction_heads = 10
    args.n_best_predictions = 1
    args.activation_ratio = 20.
    args.print_while_training = False

    args.n_exps = 1
    args.mse_steps = (1,)
    return args


def run_experiments():
    np.random.seed(42)
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    from datetime import datetime
    begin = datetime.now()
    for dataset in datasets:
        data_path = 'datasets/{}.npy'.format(dataset)
        print(dataset)
        experiment_one_per_env(data_path)
        experiment_ours(data_path)
        experiment_standard_rc(data_path)
    total_time = datetime.now() - begin
    print('Total time elapsed {}'.format(total_time))


def determine_regularization():
    np.random.seed(42)
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    for dataset in datasets:
        data_path = 'datasets/{}.npy'.format(dataset)
        args = get_hyperparameters(data_path)
        exponents = list(range(-11, 8))
        regularizations = [float('1e{}'.format(exponent)) for exponent in exponents]
        validation_measures = []
        validation_measure_errors = []
        for regularization in regularizations:
            args.regularization = regularization
            env_idx = 0  # validate on first environment (realistic assumption)
            validation_slice = slice(args.n_trajs_train + args.n_trajs_test, args.n_trajs)
            data = utils.load_data(args.data_path, envs=(env_idx,), trajs=validation_slice)
            args.n_prediction_heads = 1
            all_mses = perform_experiments(args, data, env_idx)
            means = [np.mean(mse_n) for mse_n in all_mses]
            sems = [sem(mse_n, axis=None) for mse_n in all_mses]
            validation_measures.append(means[0])
            validation_measure_errors.append(sems[0])
            # print('{:.1e}\t{:.3e}'.format(regularization, means[0]))  # MSE-1 as validation measure
        print('Best regularization on {}: {}'.format(dataset, regularizations[np.argmin(validation_measures)]))
        plt.errorbar(regularizations, validation_measures, yerr=validation_measure_errors)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('1-step ahead prediction MSE')
        plt.xlabel('regularization parameter')
        plt.title('Determine optimal regularization on {}'.format(dataset))
        plt.savefig('determine_regularization_{}.pdf'.format(dataset))
        plt.show()


def determine_threshold():
    np.random.seed(42)
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    for dataset in datasets:
        print(dataset)
        data_path = 'datasets/{}.npy'.format(dataset)
        args = get_hyperparameters(data_path)
        args.print_while_training = False
        thresholds = [1., 1.1, 1.2, 1.5, 2., 5., 10., 20., 50., 100.]
        accuracies = []
        validation_measures = []
        validation_measure_errors = []
        for threshold in thresholds:
            args.activation_ratio = threshold
            env_idx = 0  # validate mse on first environment
            validation_slice = slice(args.n_trajs_train + args.n_trajs_test, args.n_trajs)
            data = utils.load_data(args.data_path, envs=(list(range(args.n_envs))), trajs=validation_slice)
            all_mses, all_true_positives, all_total_positives = perform_accuracy_experiments(args, data, env_idx)
            precision, recall, accuracy = precision_recall_accuracy(all_true_positives, all_total_positives)
            means = [np.mean(mse_n) for mse_n in all_mses]
            sems = [sem(mse_n, axis=None) for mse_n in all_mses]
            accuracies.append(accuracy)
            validation_measures.append(means[0])
            validation_measure_errors.append(sems[0])
            print('\tthreshold {}\t MSE {:.3e}'.format(threshold, means[0]))  # MSE-1 as validation measure

        print('Best (MSE) activation ratio {}: {}'.format(dataset, thresholds[np.argmin(validation_measures)]))
        print('Best (accuracy) activation ratio {}: {}'.format(dataset, thresholds[np.argmin(accuracies)]))

        plt.errorbar(thresholds, validation_measures, yerr=validation_measure_errors)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('1-step ahead prediction MSE')
        plt.xlabel('drift detection parameter')
        plt.title('Determine optimal drift detection on {}'.format(dataset))
        plt.savefig('determine_threshold_mse_{}.pdf'.format(dataset))
        plt.show()

        plt.plot(thresholds, accuracies)
        plt.xscale('log')
        plt.ylabel('accuracy')
        plt.xlabel('drift detection parameter')
        plt.title('Determine optimal drift detection on {}'.format(dataset))
        plt.savefig('determine_threshold_accuracy_{}.pdf'.format(dataset))
        plt.show()


def estimate_accuracy():
    np.random.seed(0)
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    for dataset in datasets:
        print(dataset)
        data_path = 'datasets/{}.npy'.format(dataset)
        args = get_hyperparameters(data_path)
        data = utils.load_data(args.data_path, envs=(list(range(args.n_envs))), trajs=slice(0, args.n_trajs_train))
        res = Reservoir(args)
        true_positives, total_positives = res.train(data)
        precision_recall_accuracy(true_positives, total_positives)


def run_ours(data_path):
    args = get_hyperparameters(data_path)
    args.print_while_training = False
    args.heads_simple = False
    data = utils.load_data(args.data_path, envs=(0, 1, 2, 3), trajs=slice(0, args.n_trajs_train))
    print(data.shape)
    res = Reservoir(args)
    res.train(data)

    n_steps_predict = 200
    for env_idx in range(args.n_envs):
        sequence = utils.load_data(args.data_path, envs=(env_idx,), trajs=slice(7, 8))[0]
        res.plot_predictions(sequence, n_steps_predict, title='Environment {}'.format(env_idx))


def determine_memory():
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    for dataset in datasets:
        print(dataset)
        data_path = 'datasets/{}.npy'.format(dataset)
        args = get_hyperparameters(data_path)
        res = Reservoir(args)
        print(res.A.shape)
        print(res.B.shape)


def determine_active_heads():
    np.random.seed(42)
    datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    for dataset in datasets:
        data_path = 'datasets/{}.npy'.format(dataset)
        args = get_hyperparameters(data_path)
        n_active_heads_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        validation_measures = []
        validation_measure_errors = []
        for n_active_heads in n_active_heads_list:
            # args.n_best_predictions = n_active_heads
            env_idx = 0  # validate on first environment (realistic assumption)
            validation_slice = slice(args.n_trajs_train + args.n_trajs_test, args.n_trajs)
            data = utils.load_data(args.data_path, envs=(env_idx,), trajs=validation_slice)
            args.n_prediction_heads = 1
            all_mses = perform_experiments(args, data, env_idx)
            means = [np.mean(mse_n) for mse_n in all_mses]
            sems = [sem(mse_n, axis=None) for mse_n in all_mses]
            validation_measures.append(means[0])
            validation_measure_errors.append(sems[0])
            # print('{:.1e}\t{:.3e}'.format(regularization, means[0]))  # MSE-1 as validation measure
        print('Best regularization on {}: {}'.format(dataset, n_active_heads_list[np.argmin(validation_measures)]))
        plt.errorbar(n_active_heads_list, validation_measures, yerr=validation_measure_errors)
        # plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('1-step ahead prediction MSE')
        plt.xlabel('regularization parameter')
        plt.title('Determine optimal regularization on {}'.format(dataset))
        plt.savefig('determine_regularization_{}.pdf'.format(dataset))
        plt.show()


if __name__ == '__main__':
    # determine_regularization()
    # determine_threshold()
    # estimate_accuracy()
    # determine_memory()
    # determine_active_heads()


    run_experiments()

    # datasets = ['Van-der-Pol', 'Lorenz-63', 'Lorenz-96']
    # for dataset in datasets:
    # dataset = datasets[1]
    # data_path = 'datasets/{}.npy'.format(dataset)
    # run_ours(data_path)
