import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


def normalize(x, axis=0):
    return (x - np.mean(x, axis)) / np.std(x, axis)


def integrate_system(equations, config):
    t = np.arange(0.0, config['sequence_length'] * config['step_size'] + 10, config['step_size'])
    x = odeint(equations, config['initial_state'], t)
    x = x[int(10 / config['step_size']):]
    # x = normalize(x)
    return x


def get_trajectory_lorenz63(par, config):
    def lorenz_equations(state, t):
        x, y, z = state
        return par['sigma'] * (y - x), x * (par['rho'] - z) - y, x * y - par['beta'] * z

    return integrate_system(lorenz_equations, config)


if __name__ == "__main__":
    np.random.seed(0)
    envs = [dict(rho=28., sigma=10., beta=8. / 3),  # "normal"
            dict(rho=36., sigma=8.5, beta=3.5),  # chaos
            dict(rho=35., sigma=21., beta=1.),  # limit cycle
            dict(rho=37.5, sigma=34., beta=4.)  # fixed points
            ]
    trajectories_per_environment = 10
    trajectories = []
    for env in envs:
        env_trajectories = []
        for i in range(trajectories_per_environment):
            initial_state = [0.4, 0.4, 23.6] + [7.9, 9.0, 8.6] * np.random.rand(3,)
            config = dict(initial_state=initial_state, sequence_length=200, step_size=0.05)
            env_trajectories.append(get_trajectory_lorenz63(env, config))
        trajectories.append(np.array(env_trajectories))
    trajectories = np.array(trajectories)  # shape (environment, number trajectories, length trajectory, dimensionality)
    np.save('datasets/lorenz63.npy', trajectories)

    # plt.plot(trajectory[:, 0], trajectory[:, 2])
    # plt.show()
    # plt.plot(trajectory)
    # plt.show()

    # for i, env in enumerate(trajectories):
    #         plt.plot(env[0, :, 0], env[0, :, 2])
    # plt.legend(list(range(4)))
    # plt.title('Multiple environments for Lorenz-63')
    # plt.show()
