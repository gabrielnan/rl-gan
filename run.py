from ast import literal_eval as make_tuple
from tqdm import tqdm
import cartpoleMod
import gym
import numpy as np

exp = ['CartPole-v0','Pendulum-v0', 'MountainCar-v0', 'SpaceInvaders-v0']

def main():
    filename = 'dataset.csv'
    data = read(filename)
    sample = data[0]
    env = cartpoleMod.CartPoleModEnv()

    env.reset()
    total_error = compute_error(env, sample)
    print(total_error)

def main_record():
    record()

def create_GAN():


    return None

def compute_error(env, sample):
    initial_state = sample[0, :-1]
    total_error = np.zeros(initial_state.shape)
    action = int(sample[0, -1])
    env.set_state(initial_state)
    print(env.state, '=====', initial_state)

    for step in sample[1:]:
        env.step(action)
        state = step[:-1]
        print(env.state, '=====', state)
        total_error += state_error(env.state, state)
        action = int(step[-1])
    return total_error

def state_error(true_state, pred_state):
    return np.abs(true_state - pred_state)


def record(nb_samples=2000, nb_steps=100, env_name='CartPole-v0'):
    env = gym.make(env_name)
    state_dim = len(env.reset()) + 1

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
    np.savetxt('dataset.csv', data.flatten(), delimiter=',', header=str(data.shape))
    # np.savetxt('dataset_{}.csv'.format(env_name[:-3]), data.flatten(), delimiter=',', header=str(data.shape))


def read(filename):
    data = np.genfromtxt(filename, dtype='float', delimiter=',', skip_header=1)
    with open(filename, 'r') as file:
        header = file.readline()
    return data.reshape(make_tuple(header[2:]))

if __name__ == '__main__':
    main()
    # main_record()
