import numpy as np
import gym

exp = ['CartPole-v0', 'MountainCar-v0', 'SpaceInvaders-v0', 'Breakout-v0', 'Pendulum-v0']
env = gym.make(exp[4])
observation = env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    

