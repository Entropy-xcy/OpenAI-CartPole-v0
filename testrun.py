import gym
import numpy as np

env = gym.make('CartPole-v0')
env.reset()

def getNoise():
    return np.random.rand(4) * 2 - 1

def run(parameters):
    observation = env.reset()
    Sreward = 0

    for _ in range(200):
        env.render()
        if np.matmul(parameters, observation) < 0:
            action = 0
        else:
            action = 1

        observation, reward, done, info = env.step(action)

        Sreward += reward
        if done:
            break
    return Sreward

def train():
    noise_scale = 0.4
    Mreward = 0

    parameters = getNoise()

    while True:
        newparams = parameters + getNoise() * noise_scale
        reward = run(newparams)

        if reward>Mreward:
            Mreward = reward
            parameters = newparams

        print('Reward:', reward, 'Max reward:', Mreward)

        if reward == 200:
            break
    act(parameters)

def act(parameters):
    print('Train step Over Now, Entering Test step')

    Sreward = 0
    for _ in range(10):
        Sreward += run(parameters)

    print('Final Average Score:', Sreward / 10)

if __name__ == '__main__':
    train()
