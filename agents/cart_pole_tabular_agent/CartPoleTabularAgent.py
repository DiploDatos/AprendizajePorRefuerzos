import gym
import six
import numpy as np
import pandas as pd
from functools import reduce

"""
Module adapted from Victor Mayoral Vilches <victor@erlerobotics.com>
"""

# functions that split the state -------------------------------------------


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

#  -------------------------------------------------------------------------


class CartPoleTabularAgent:

    def __init__(self):

        # basic configuration
        self._environment_name = "CartPole-v0"
        self._environment_instance = None  # (note that the "None" variables have values yet to be assigned)
        self.random_state = None  # type: np.random.RandomState
        self._cutoff_time = None
        self._hyper_parameters = None

        # number of features in the state
        self._number_of_features = None

        # Dictionary of Q-values
        self.q = {}

        # list that contains the amount of time-steps the cart had the pole up during the episode. It is used as a way
        # to score the performance of the agent. It has a maximum value of 200 time-steps
        self._last_time_steps = None

        # whether ot not to display a video of the agent execution at each episode
        self.display_video = True

        # attributes that controls the state space reduction. For example, 8 bins over a state feature means that the
        # feature is divided in 8 parts, where all of them share the same size.
        self.n_bins = 8
        self.n_bins_angle = 10

        # attribute initialization
        self._cart_position_bins = None
        self._pole_angle_bins = None
        self._cart_velocity_bins = None
        self._angle_rate_bins = None

        # Dictionary of Q-values
        self.q = {}

        # default hyper-parameters for Q-learning
        self._alpha = 0.5
        self._gamma = 0.9
        self._epsilon = 0.1
        self.episodes_to_run = 3000  # amount of episodes to run for each run of the agent
        self.actions = None

        # matrix with 3 columns, where each row represents the action, reward and next state obtained from the agent
        # executing an action in the previous state
        self.action_reward_state_trace = []

    def set_cutoff_time(self, cutoff_time):
        """
        Method that sets a maximum number of time-steps for each agent episode.
        :param cutoff_time:
        """
        self._cutoff_time = cutoff_time

    def set_hyper_parameters(self, hyper_parameters):
        """
        Method that passes the hyper_parameter configuration vector to the RL agent.
        :param hyper_parameters: a list containing the hyper-parameters that are to be set in the RL algorithm.
        """
        self._hyper_parameters = hyper_parameters

        for key, value in six.iteritems(hyper_parameters):
            if key == 'alpha':  # Learning-rate
                self._alpha = value

            if key == 'gamma':
                self._gamma = value

            if key == 'epsilon':
                self._epsilon = value

    def init_agent(self):
        """
        Initializes the reinforcement learning agent with a default configuration.
        """
        self._environment_instance = gym.make(self._environment_name)

        # environment is seeded
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        if self.display_video:
            # video_callable=lambda count: count % 10 == 0)
            self._environment_instance = gym.wrappers.Monitor(self._environment_instance,
                                                              '/tmp/cartpole-experiment-1',
                                                              force=True)

        # the number of features is obtained
        self._number_of_features = self._environment_instance.observation_space.shape[0]

        # Number of states is huge so in order to simplify the situation
        # we discretize the space to: 10 ** number_of_features
        self._cart_position_bins = pd.cut([-2.4, 2.4], bins=self.n_bins, retbins=True)[1][1:-1]
        self._pole_angle_bins = pd.cut([-2, 2], bins=self.n_bins_angle, retbins=True)[1][1:-1]
        self._cart_velocity_bins = pd.cut([-1, 1], bins=self.n_bins, retbins=True)[1][1:-1]
        self._angle_rate_bins = pd.cut([-3.5, 3.5], bins=self.n_bins_angle, retbins=True)[1][1:-1]

        self.actions = range(self._environment_instance.action_space.n)

    def restart_agent_learning(self):
        """
        Restarts the reinforcement learning agent so it starts learning from scratch, in order to avoid bias with
        previous learning experience.
        """
        # last run is cleared
        self._last_time_steps = []
        self.action_reward_state_trace = []

        # q values are restarted
        self.q = {}

    def run(self):
        """
        Runs the reinforcement learning agent with a given configuration.
        """
        for i_episode in range(self.episodes_to_run):
            # an instance of an episode is run until it fails or until it reaches 200 time-steps

            # resets the environment, obtaining the first state observation
            observation = self._environment_instance.reset()

            # the state is split into the different variables
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

            # a number of four digits representing the actual state is obtained
            state = build_state([to_bin(cart_position, self._cart_position_bins),
                                 to_bin(pole_angle, self._pole_angle_bins),
                                 to_bin(cart_velocity, self._cart_velocity_bins),
                                 to_bin(angle_rate_of_change, self._angle_rate_bins)])

            for t in range(self._cutoff_time):

                # Pick an action based on the current state
                action = self.choose_action(state)
                # Execute the action and get feedback
                observation, reward, done, info = self._environment_instance.step(action)

                # current state transition is saved
                self.action_reward_state_trace.append([action, reward, observation])

                # Digitize the observation to get a state
                cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

                next_state = build_state([to_bin(cart_position, self._cart_position_bins),
                                          to_bin(pole_angle, self._pole_angle_bins),
                                          to_bin(cart_velocity, self._cart_velocity_bins),
                                          to_bin(angle_rate_of_change, self._angle_rate_bins)])

                if not done:
                    self.learn(state, action, reward, next_state)
                    state = next_state
                else:
                    if t < self._cutoff_time - 1:  # tests whether the pole fell
                        reward = -200  # the pole fell, so a negative reward is computed to avoid failure
                    self.learn(state, action, reward, next_state)
                    self._last_time_steps = np.append(self._last_time_steps, [int(t + 1)])
                    break

        last_time_steps_list = list(self._last_time_steps)
        last_time_steps_list.sort()
        print("Overall score: {:0.2f}".format(self._last_time_steps.mean()))
        print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y,
                                                      last_time_steps_list[-100:]) / len(last_time_steps_list[-100:])))

        return self._last_time_steps.mean()

    def choose_action(self, state):
        """
        Chooses an action according to the learning previously performed
        """
        q = [self.q.get((state, a), 0) for a in self.actions]
        max_q = max(q)

        if self.random_state.uniform() < self._epsilon:
            return self.random_state.choice(self.actions)  # a random action is returned

        count = q.count(max_q)

        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == max_q]
            i = self.random_state.choice(best)
        else:
            i = q.index(max_q)

        action = self.actions[i]

        return action

    def learn(self, state, action, reward, next_state):
        """
        Performs a Q-learning update for a given state transition

        Q-learning update:
        Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        """
        new_max_q = max([self.q.get((next_state, a), 0.0) for a in self.actions])
        old_value = self.q.get((state, action), 0.0)

        self.q[(state, action)] = old_value + self._alpha * (reward + self._gamma * new_max_q - old_value)

    def destroy_agent(self):
        """
        Destroys the reinforcement learning agent, in order to instantly release the memory the agent was using.
        """
        self._environment_instance.close()
