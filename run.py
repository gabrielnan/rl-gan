from ast import literal_eval as make_tuple

import argparse
import gan
import gym
import numpy as np
import os
from tqdm import tqdm

from gan import check_dirs

exp = ['CartPole-v0', 'Pendulum-v0', 'MountainCar-v0', 'SpaceInvaders-v0']


def main(args):
    data = read(args.data)
    env = gym.make(args.env)
    create_set_state_method(env.__class__)

    gen = gan.Generator(data, args.noise_dim)
    samples = gen.generate(1000)
    # print sample

    env.reset()
    total_error = dataset_error(env, samples)
    print('Avg Error:', list(total_error))
    # print(samples[0])

    random_samples = np.random.uniform(-1, 1, size=samples.shape)
    random_total_error = dataset_error(env, random_samples)
    print('Avg Random Error:', list(random_total_error))
    # print(random_samples[0])


def create_set_state_method(cls):
    def set_state(self, state):
        self.reset()
        if self.state.shape == state.shape:
            self.state = state
        else:
            cls.logger.error('Parameter state does not have same dimensions as this environment state.')

    cls.set_state = set_state


def dataset_error(env, data):
    total_error = 0
    for sample in data:
        sample = round_actions(sample)
        total_error += sample_norm_MSE(env, sample)
    return total_error / len(data)


def round_actions(sample):
    sample[:, -1] = np.round(1 / (1 + np.exp(sample[:, -1])))
    return sample


def sample_error(env, sample):
    initial_state = sample[0, :-1]
    total_error = np.zeros(initial_state.shape)
    action = int(sample[0, -1])
    env.set_state(initial_state)
    # print(env.state, '=====', initial_state)

    for step in sample[1:]:
        env.step(action)
        state = step[:-1]
        # print(env.state, '\n', state, '\n\n')
        step_error = state_error(env.state, state)
        total_error += step_error
        action = int(step[-1])
    return total_error


def sample_norm_MSE(env, sample):
    total_error = sample_error(env, sample)
    state_range = np.max(sample[:, :-1], axis=0) - np.min(sample[:, :-1], axis=0)
    return total_error / state_range


def state_error(true_state, pred_state):
    return np.abs(true_state - pred_state)


def format_dataset_name(env_name, shape):
    return '{}_{}_dataset.csv'.format(env_name, shape)


def record(nb_samples, nb_steps, env_name, extend_steps=False, nb_max_steps=200):
    # Check for directories
    check_dirs('datasets')
    dataset_filename = 'datasets/' + format_dataset_name(env_name, (nb_samples, nb_steps))
    if os.path.isfile(dataset_filename):
        # TODO idk lol
        pass

    env = gym.make(env_name)
    state_dim = len(env.reset()) + 1

    if extend_steps:
        nb_max_steps = int(nb_max_steps / nb_steps) * nb_steps
        nb_traj_samples = int(np.ceil(float(nb_samples) * nb_steps / nb_max_steps))
        data = get_data(env, nb_traj_samples, nb_max_steps, state_dim)
        data = data.flatten()[:nb_samples * nb_steps * state_dim]
        data = data.reshape((nb_samples, nb_steps, state_dim))
    else:
        data = get_data(env, nb_samples, nb_steps, state_dim)

    np.savetxt('datasets/dataset.csv', data.flatten(), delimiter=',', header=str(data.shape))
    np.savetxt(dataset_filename, data.flatten(), delimiter=',', header=str(data.shape))


def get_data(env, nb_samples, nb_steps, state_dim):
    data = np.array([]).reshape(0, nb_steps, state_dim)

    for _ in tqdm(range(nb_samples)):
        sample = np.array([]).reshape(0, state_dim)
        step = env.reset()
        for _ in range(nb_steps):
            action = env.action_space.sample()
            step = np.append(step, action)
            sample = np.vstack((sample, step))
            step = env.step(action)[0]
        data = np.vstack((data, sample.reshape(1, nb_steps, state_dim)))
    return data


def read(filename):
    data = np.genfromtxt(filename, dtype='float', delimiter=',', skip_header=1)
    with open(filename, 'r') as data_file:
        header = data_file.readline()
    return data.reshape(make_tuple(header[2:]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default='datasets/dataset.csv')
    parser.add_argument('--nb_samples', type=int, default=10000)
    parser.add_argument('--nb_steps', type=int, default=6)
    parser.add_argument('--noise_dim', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
    # record(args.nb_samples, args.nb_steps, args.env, extend_steps=True)
