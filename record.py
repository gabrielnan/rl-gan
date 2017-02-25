import numpy as np
import gym, os
from tqdm import tqdm

exp = ['CartPole-v0', 'MountainCar-v0', 'SpaceInvaders-v0']
nb_samples = 2000
nb_steps = 100

def main():
    env = gym.make(exp[0])
    state_dim = len(env.reset()) + 1

    data = np.array([]).reshape(0, nb_steps, state_dim)

    for _ in tqdm(range(nb_samples)):
        sample = np.array([]).reshape(0, state_dim)
        for _ in range(nb_steps):
            env.reset()
            action = env.action_space.sample()
            step = env.step(action)[0]
            step = np.append(step, action)
            sample = np.vstack((sample, step))
        data = np.vstack((data, sample.reshape(1, nb_steps, state_dim)))
    np.savetxt('dataset.csv', data, delimiter=',')

if __name__ == '__main__':
    main()
