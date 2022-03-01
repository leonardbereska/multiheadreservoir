import numpy as np
from matplotlib import pyplot as plt


def load_data(data_path, envs, trajs):
    dataset = np.load(data_path)
    assert len(dataset.shape) == 4  # environments, sequences, time steps, dimensions
    data_selection = dataset[envs, trajs]
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


def plot_environment(sequences):
    for idx, sequence in enumerate(sequences):
        plt.plot(sequence[:, 0], sequence[:, -1], label=str(idx))
    plt.legend()
    plt.title('Environment sequences')
    plt.show()