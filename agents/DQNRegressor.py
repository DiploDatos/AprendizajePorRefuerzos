import math
import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

"""
Adapted from implementation of Victor Mayoral Vilches

Original text:

Q-learning with value function approximation (sklearn estimator) for different RL problems 

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

    @author: Juan Cruz Barsce <jbarsce@frvm.utn.edu.ar>
    @author: Ezequiel Beccaría <ebeccaria@frvm.utn.edu.ar>
    @author: Jorge Andres Palombarini <jpalombarini@frvm.utn.edu.ar>
"""


class DQNRegressor():

    def __init__(self, actions_n, alpha, alpha_decay, epsilon_init, epsilon_decay, epsilon_min, gamma, scaler):
        self.tuples = {}
        self._epsilon = epsilon_init  # exploration constant
        self._epsilon_decay = epsilon_decay  # exploration constant decay
        self._epsilon_min = epsilon_min  # exploration constant min
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self._gamma = gamma      # discount factor
        self._actions = range(actions_n)
        self._scaler = scaler

        # Init model
        self._value_function = Sequential()
        self._value_function.add(Dense(24, input_dim=4, activation='tanh'))
        self._value_function.add(Dense(48, activation='tanh'))
        self._value_function.add(Dense(2, activation='linear'))
        self._value_function.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def get_q(self, state):
        """
        Gets the tabular Q-value for the specified state and action pair. Returns 0 if there is no value for such pair
        """
        return self._value_function.predict(self.pre_process(state))

    def get_epsilon(self, t):
        return max(self._epsilon_min, min(self._epsilon, 1.0 - math.log10((t + 1) * self._epsilon_decay)))

    def learn(self, minibatch):
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in minibatch:
            # get current Qsa value
            qsa = self._value_function.predict(self.pre_process(state))
            qsa[0][action] = reward if done else reward + self._gamma * np.max(self._value_function.predict(self.pre_process(next_state))[0])

            x_batch.append(self.pre_process(state)[0])
            y_batch.append(qsa[0])

        self._value_function.fit(np.array(x_batch), np.array(y_batch), batch_size=len(minibatch), verbose=0)

        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def choose_action(self, state, episode):
        """
        Chooses an action according to the learning previously performed
        """
        # get Q values for each action choice
        q = self.get_q(state)

        if random.random() < self.get_epsilon(episode):
            action = random.choice(self._actions)  # a random action is returned
        else:
            # get max q values indexes
            max_q_i = np.argwhere(q == np.amax(q, 0)).flatten()

            # In case there're several state-action max values
            # we select a random one among them
            i = random.choice(max_q_i)

            action = self._actions[i]

        return action

    def pre_process(self, state):
        """
        Gets State-Action tuple
        """
        scaled_state_action = self._scaler.transform(state.reshape(1, -1))
        return scaled_state_action
