import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder

"""
Adapted from implementation of Victor Mayoral Vilches

Original text:

Q-learning with value function approximation (sklearn estimator) for different RL problems 

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

    @author: Juan Cruz Barsce <jbarsce@frvm.utn.edu.ar>
    @author: Ezequiel Beccaría <ebeccaria@frvm.utn.edu.ar>
    @author: Jorge Andres Palombarini <jpalombarini@frvm.utn.edu.ar>
"""


class QLearningSGDRegressor():

    def __init__(self, init_state, actions_n, epsilon, epsilon_min, epsilon_decay, gamma, vf, scaler):
        self.tuples = {}
        self._epsilon = epsilon  # exploration constant
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._gamma = gamma      # discount factor
        self._actions = range(actions_n)
        # OneHotEncoder for actions
        self._action_encoder = OneHotEncoder()
        self._action_encoder.fit(np.expand_dims(np.arange(actions_n), axis=1))
        self._value_function = vf
        self._scaler = scaler
        # init value function predictor
        for a in self._actions:
            # fit with initial state and reward 0S
            vf.partial_fit(self.get_state_action_tuple(init_state, a), np.zeros(1))

    def get_q(self, state, action):
        """
        Gets the tabular Q-value for the specified state and action pair. Returns 0 if there is no value for such pair
        """
        return self._value_function.predict(self.get_state_action_tuple(state, action))

    def get_epsilon(self, t):
        return max(self._epsilon_min, min(self._epsilon, 1.0 - np.log10((t + 1) * self._epsilon_decay)))

    def learn(self, minibatch):
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in minibatch:
            # get next max Qsa value
            new_max_q = max([self.get_q(next_state, a) for a in self._actions])
            # update new Qsa
            # qsa = old_value + self._alpha * (reward + self._gamma * new_max_q - old_value)
            qsa = np.float64(reward) if done else reward + self._gamma * new_max_q[0]

            x_batch.append(self.get_state_action_tuple(state, action).flatten())
            y_batch.append(qsa)

        self._value_function.partial_fit(np.array(x_batch), np.array(y_batch))

        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def choose_action(self, state, episode):
        """
        Chooses an action according to the learning previously performed
        """
        # get Q values for each action choice
        q = np.asarray([self.get_q(state, a) for a in self._actions]).flatten()

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

    def get_state_action_tuple(self, state, action):
        """
        Gets State-Action tuple
        """
        # transform action representation to OneHotEncoder
        enc_a = self._action_encoder.transform(np.array([action]).reshape(1, -1)).toarray()[0, :]
        scaled_state_action = self._scaler.transform(np.hstack((state, enc_a)).reshape(1, -1))
        return scaled_state_action
