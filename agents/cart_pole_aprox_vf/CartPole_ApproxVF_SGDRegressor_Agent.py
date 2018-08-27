import gym
import six
import numpy as np
from sklearn.linear_model import SGDRegressor

from agents.QLearning_SGDRegressor import QLearningSGDRegressor
from functools import reduce
from agents.RLAgent import RLAgent

"""
Module adapted from Victor Mayoral Vilches <victor@erlerobotics.com>
"""


class CartPoleApproxVFSGDRegressorAgent(RLAgent):

    def __init__(self):

        # basic configuration
        self._environment_name = "CartPole-v0"
        self._environment_instance = None  # (note that the "None" variables have values yet to be assigned)
        self._random_state = None
        self._cutoff_time = None
        self._hyper_parameters = None

        # number of features in the state
        self._number_of_features = None

        # list that contains the amount of time-steps the cart had the pole up during the episode. It is used as a way
        # to score the performance of the agent. It has a maximum value of 200 time-steps
        self._last_time_steps = None

        # whether ot not to display a video of the agent execution at each episode
        self.display_video = True

        # attribute initialization
        self._cart_position_bins = None
        self._pole_angle_bins = None
        self._cart_velocity_bins = None
        self._angle_rate_bins = None

        # the Q-learning algorithm
        self._q_learning = None
        self._value_function = None

        # default hyper-parameters
        self._alpha = 0.5
        self._gamma = 0.9
        self._epsilon = 0.1
        self.episodes_to_run = 3000  # amount of episodes to run for each run of the agent

        # matrix with 3 columns, where each row represents the action, reward and next state obtained from the agent
        # executing an action in the previous state
        self.action_reward_state_trace = []

    def set_random_state(self, random_state):
        """
        Method that sets a previously created numpy.RandomState object in order to share the same random seed.
        :param random_state: an instantiated np.RandomState object
        """
        self._random_state = random_state

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
        if self._random_state is not None:
            self._environment_instance.seed(self._random_state.randint(0, 1e10))

        if self.display_video:
            # video_callable=lambda count: count % 10 == 0)
            self._environment_instance = gym.wrappers.Monitor(self._environment_instance,
                                                              '/tmp/cartpole-experiment-1',
                                                              force=True)

    def restart_agent_learning(self):
        """
        Restarts the reinforcement learning agent so it starts learning from scratch, in order to avoid bias with
        previous learning experience.
        """
        # last run is cleared
        self._last_time_steps = []
        self.action_reward_state_trace = []

        # a new q_learning agent is created
        del self._q_learning

        # a new Q-learning object is created to replace the previous object
        self._q_learning = QLearningSGDRegressor(
            init_state=self._environment_instance.reset(),
            actions_n=self._environment_instance.action_space.n,
            alpha=self._alpha,
            gamma=self._gamma,
            epsilon=self._epsilon,
            vf=SGDRegressor())

        # the number of features is obtained
        self._number_of_features = self._environment_instance.observation_space.shape[0]
        # Init Ridge Regression Value function approximation
        self._value_function = SGDRegressor()

    def run(self):
        """
        Runs the reinforcement learning agent with a given configuration.
        """
        for i_episode in range(self.episodes_to_run):
            # an instance of an episode is run until it fails or until it reaches 200 time-steps

            # resets the environment, obtaining the first state observation
            state = self._environment_instance.reset()

            for t in range(self._cutoff_time):

                # Pick an action based on the current state
                action = self._q_learning.choose_action(state)
                # Execute the action and get feedback
                observation, reward, done, info = self._environment_instance.step(action)

                # current state transition is saved
                self.action_reward_state_trace.append([action, reward, observation])

                next_state = observation

                if not done:
                    self._q_learning.learn(state, action, reward, next_state)
                    state = next_state
                else:
                    # Q-learn stuff
                    reward = -200  # a negative reward is computed so as to avoid failure
                    self._q_learning.learn(state, action, reward, next_state)
                    self._last_time_steps = np.append(self._last_time_steps, [int(t + 1)])
                    break

        last_time_steps_list = list(self._last_time_steps)
        last_time_steps_list.sort()
        print("Overall score: {:0.2f}".format(self._last_time_steps.mean()))
        print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y,
                                                      last_time_steps_list[-100:]) / len(last_time_steps_list[-100:])))

        return self._last_time_steps.mean()

    def destroy_agent(self):
        """
        Destroys the reinforcement learning agent, in order to instantly release the memory the agent was using.
        """
        self._environment_instance.close()