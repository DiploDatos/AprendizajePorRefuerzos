import random

"""

Adapted from implementation of Victor Mayoral Vilches

Original text:

Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
"""


class QLearning:

    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def get_q(self, state, action):
        """
        Gets the tabular Q-value for the specified state and action pair. Returns 0 if there is no value for such pair
        """
        return self.q.get((state, action), 0.0)

    def learn(self, state, action, reward, next_state):
        """
        Performs a Q-learning update for a given state transition

        Q-learning update:
        Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        """
        new_max_q = max([self.get_q(next_state, a) for a in self.actions])
        old_value = self.get_q(state, action)

        self.q[(state, action)] = old_value + self.alpha * (reward + self.gamma * new_max_q - old_value)

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
