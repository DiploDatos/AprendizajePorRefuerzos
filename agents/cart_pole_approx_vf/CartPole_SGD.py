# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
from sklearn.linear_model import SGDRegressor


class SGDCartPoleSolver():
    def __init__(self, n_episodes=1000, max_env_steps=None, gamma=0.9, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.005, alpha=0.0001, batch_size=128, monitor=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = SGDRegressor(
                alpha=self.alpha,
                learning_rate='optimal',
                shuffle=False,
                warm_start=True)

        # initialize model
        self.model.partial_fit(self.preprocess_state(self.env.reset(), 0), [0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        qsa = np.asarray([self.model.predict(self.preprocess_state(state, a)) for a in range(self.env.action_space.n)]).flatten()
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(qsa)

    def get_epsilon(self, t):
        return max(self.epsilon_min, self.epsilon*math.exp(-self.epsilon_decay*t))

    @staticmethod
    def preprocess_state(state, action):
        return np.reshape(np.hstack((state, action)), [1, 5])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            qsa_s_prime = np.asarray([self.model.predict(self.preprocess_state(next_state, a)) for a in range(self.env.action_space.n)])
            qsa_s = reward if done else reward + self.gamma * np.max(qsa_s_prime)
            x_batch.append(self.preprocess_state(state, action)[0])
            y_batch.append(qsa_s)

        self.model.partial_fit(np.array(x_batch), np.array(y_batch))

    def run(self):
        scores100 = deque(maxlen=100)
        scores = []
        for e in range(self.n_episodes):
            state = self.env.reset()
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

            scores100.append(i)
            scores.append(i)
            mean_score = np.mean(scores100)
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)

        return scores


if __name__ == '__main__':
    agent = SGDCartPoleSolver()
    agent.run()