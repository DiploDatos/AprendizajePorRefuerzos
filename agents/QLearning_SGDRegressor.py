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

    def __init__(self, init_state, actions_n, epsilon, alpha, gamma, vf):
        self.tuples = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = range(actions_n)
        # OneHotEncoder for actions
        self.action_encoder = OneHotEncoder()
        self.action_encoder.fit(np.expand_dims(np.arange(actions_n), axis=1))
        self.value_function = vf
        # init value function predictor
        for a in self.actions:
            enc_a = self.action_encoder.transform(np.array([a]).reshape(1, -1)).toarray()[0, :]
            sa = np.hstack((init_state, enc_a))
            vf.partial_fit(sa.reshape(1, -1), np.zeros(1))

    def get_q(self, state, action):
        """
        Gets the tabular Q-value for the specified state and action pair. Returns 0 if there is no value for such pair
        """
        sa = np.hstack((state, self.action_encoder.transform(action)))
        return self.value_function.predict(sa)

    def learn(self, state, action, reward, next_state):
        """
        Performs a Q-learning update for a given state transition

        Q-learning update:
        Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        """
        new_max_q = max([self.get_q(next_state, a) for a in self.actions])
        old_value = self.get_q(state, action)

        qsa = old_value + self.alpha * (reward + self.gamma * new_max_q - old_value)
        sa = np.hstack((state, self.action_encoder.transform(action)))
        self.value_function.partial_fit(sa, qsa)

    def choose_action(self, state, return_q=False):
        """
        Chooses an action according to the learning previously performed
        """
        q = [self.get_q(state, a) for a in self.actions]
        max_q = max(q)

        if random.random() < self.epsilon:
            return random.choice(self.actions)  # a random action is returned

        count = q.count(max_q)

        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == max_q]
            i = random.choice(best)
        else:
            i = q.index(max_q)

        action = self.actions[i]

        if return_q:  # if they want it, give it!
            return action, q
        return action
