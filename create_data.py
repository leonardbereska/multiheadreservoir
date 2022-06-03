import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


def normalize(x, axis=(0,)):
    return (x - np.mean(x, axis=axis)) / np.std(x, axis=axis)


class DynamicalSystem:
    def __init__(self):
        self.save_directory = 'datasets'
        self.transient_cutoff = 100

        self.sequences_per_environment = 15
        self.sequence_length = 200
        self.step_size = 0.05

        self.save_name = ''
        self.environments = list()
        self.n_dimensions = NotImplementedError

    def get_equations(self, par):
        return NotImplementedError

    def get_initial_state(self):
        return np.random.rand(self.n_dimensions)

    def integrate_system(self, par):
        equations = self.get_equations(par)
        t = np.arange(0.0, self.sequence_length * self.step_size + self.transient_cutoff, self.step_size)
        x = odeint(equations, self.get_initial_state(), t)
        x = x[int(self.transient_cutoff / self.step_size):]
        return x

    def get_data(self):
        """
        :return: data with shape (number environments, number trajectories per environment, time steps, dimensions)
        """
        sequences = []
        for environment in self.environments:
            env_trajectories = [self.integrate_system(environment) for i in range(self.sequences_per_environment)]
            sequences.append(np.array(env_trajectories))
        sequences = np.array(sequences)  # shape (environment, number trajectories, length trajectory, dimension)
        sequences = normalize(sequences, axis=(0, 1, 2))  # normalize environments jointly
        return sequences

    def save_data(self):
        data = self.get_data()
        self.plot(data)
        np.save('{}/{}.npy'.format(self.save_directory, self.save_name), data)

    def plot_trajectory(self, trajectory, label=None):
        if label is None:
            plt.plot(trajectory[:, 0], trajectory[:, -1])
        else:
            plt.plot(trajectory[:, 0], trajectory[:, -1], label=label)

    def plot(self, data=None):
        if data is None:
            data = self.get_data()
        i = len(data)
        for env in reversed(data):
            i -= 1
            trajectory = env[0]
            dict_ = self.environments[i]
            greek_letters = ('mu', 'sigma', 'rho', 'beta')
            key_repr = [u'$\{}$'.format(key) if key in greek_letters else key for key in dict_.keys()]
            print_queue = ['{}={}'.format(key_repr[i], dict_[key]) for i, key in enumerate(dict_.keys())]
            print_out = ', '.join(print_queue)
            # plt.title('{} ({})'.format(self.save_name, print_out))
            plt.title('{}'.format(self.save_name))
            self.plot_trajectory(trajectory, label=print_out)
            # plt.show()
            plt.legend()
            name = 'plots/{}_{}.pdf'.format(self.save_name, print_out)
            print(name)
            plt.savefig(name)
        plt.close()


class Lorenz63(DynamicalSystem):
    def __init__(self):
        super(Lorenz63, self).__init__()
        self.save_name = 'Lorenz-63'
        self.n_dimensions = 3
        self.environments = [dict(rho=28., sigma=10., beta=2.66),  # "normal"
                             dict(rho=36., sigma=8.5, beta=3.5),  # chaos
                             dict(rho=35., sigma=21., beta=1.),  # limit cycle
                             dict(rho=60., sigma=20., beta=8.)  # fixed points
                             ]

    def get_equations(self, par):
        def lorenz63(x, t):
            return par['sigma'] * (x[1] - x[0]), x[0] * (par['rho'] - x[2]) - x[1], x[0] * x[1] - par['beta'] * x[2]

        return lorenz63


class Lorenz96(DynamicalSystem):
    def __init__(self, n_dimensions=10):
        super(Lorenz96, self).__init__()
        self.save_name = 'Lorenz-96'
        self.n_dimensions = n_dimensions
        self.environments = [dict(F=5),
                             dict(F=10),
                             dict(F=20),
                             dict(F=50),
                             ]

    def plot_trajectory(self, trajectory, **kwargs):
        plt.imshow(trajectory, aspect='auto')
        plt.ylabel('time steps')
        plt.gca().invert_yaxis()
        plt.xlabel('dimensions')

    def get_equations(self, par):
        def lorenz96(x, t):
            x_next = np.zeros(self.n_dimensions)
            for i in range(self.n_dimensions):
                x_next[i] = (x[(i + 1) % self.n_dimensions] - x[i - 2]) * x[i - 1] - x[i] + par['F']
            return x_next

        return lorenz96


class VanderPol(DynamicalSystem):
    def __init__(self):
        super(VanderPol, self).__init__()
        self.save_name = 'Van-der-Pol'
        self.n_dimensions = 2
        self.environments = [dict(mu=0.5),
                             dict(mu=1.),
                             dict(mu=2.),
                             dict(mu=4.),
                             ]

    def get_equations(self, par):
        def vanderpol(x, t):
            (x, y) = x
            return [y, (par['mu'] * (1. - x * x) * y - x)]

        return vanderpol


if __name__ == "__main__":
    np.random.seed(0)
    Lorenz63().save_data()
    VanderPol().save_data()
    Lorenz96().save_data()