import gym
import six
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from agents.QLearning_RandomForestRegressor import QLearningRandomForestRegressor
from agents.RLAgent import RLAgent

"""
Module adapted from Victor Mayoral Vilches <victor@erlerobotics.com>
"""


class CartPoleApproxRandomForestRegressorAgent(RLAgent):

    def __init__(self):

        # basic configuration
        self._environment_name = "CartPole-v0"
        self._environment_instance = None  # (note that the "None" variables have values yet to be assigned)
        self._state_action_min_values = np.array([-2.4, np.finfo(np.float64).min, -41.8, np.finfo(np.float64).min, 0, 0])
        self._state_action_max_values = np.array([2.4, np.finfo(np.float64).max, 41.8, np.finfo(np.float64).max, 1, 1])
        boundaries = np.vstack((self._state_action_min_values, self._state_action_max_values))
        self._scaler = StandardScaler()
        self._scaler.fit(boundaries)
        self._random_state = None
        self._cutoff_time = None
        self._hyper_parameters = None

        # number of features in the state
        self._number_of_features = None

        # list that contains the amount of time-steps the cart had the pole up during the episode. It is used as a way
        # to score the performance of the agent. It has a maximum value of 200 time-steps
        self.timesteps_of_episode = None
        self.reward_of_episode = None

        # whether ot not to display a video of the agent execution at each episode
        self.display_video = False

        # the Q-learning algorithm
        self._learning_algorithm = None
        self._value_function = None

        # default hyper-parameters
        self._alpha = None
        self._gamma = None
        self._epsilon = None
        self._batch_size = None

        self.episodes_to_run = 3000  # amount of episodes to run for each run of the agent

        # matrix with 3 columns, where each row represents the action, reward and next state obtained from the agent
        # executing an action in the previous state
        self._memory = []

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
            if key == 'gamma':  # Discount factor
                self._gamma = value
            if key == 'epsilon':  # Exploration rate
                self._epsilon = value
            if key == 'batch_size':
                self._batch_size = value

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
        self.timesteps_of_episode = []
        self.reward_of_episode = []
        self._memory = []

        # a new q_learning agent is created
        del self._learning_algorithm

        # a new Q-learning object is created to replace the previous object
        self._learning_algorithm = QLearningRandomForestRegressor(
            init_state=self._environment_instance.reset(),
            actions_n=self._environment_instance.action_space.n,
            gamma=self._gamma,
            epsilon=self._epsilon,
            vf=RandomForestRegressor(n_estimators=1, warm_start=True),
            scaler=self._scaler)

    def run(self):
        """
        Runs the reinforcement learning agent with a given configuration.
        """
        for i_episode in range(self.episodes_to_run):
            # an instance of an episode is run until it fails or until it reaches 200 time-steps

            # resets the environment, obtaining the first state observation
            state = self._environment_instance.reset()
            cum_reward = 0

            for t in range(self._cutoff_time):

                # Pick an action based on the current state
                action = self._learning_algorithm.choose_action(state)
                # Execute the action and get feedback
                observation, reward, done, info = self._environment_instance.step(action)

                next_state = observation

                if not done:
                    cum_reward += reward
                    # current state transition is saved
                    self._memory.append((state, action, reward, next_state, done))
                    state = next_state
                else:
                    if t < self._cutoff_time - 1:  # tests whether the pole fell
                        reward = -200  # the pole fell, so a negative reward is computed to avoid failure
                    # current state transition is saved
                    self._memory.append((state, action, reward, next_state, done))

                    self.timesteps_of_episode = np.append(self.timesteps_of_episode, [int(t + 1)])
                    cum_reward += reward
                    self.reward_of_episode = np.append(self.reward_of_episode, cum_reward)
                    break
            print("Episode {:0d} finish. Reward: {:0.2f}".format(i_episode, cum_reward))

            # train value function predictor with a minibach
            self._learning_algorithm.learn(random.sample(self._memory, min(len(self._memory), self._batch_size)))

        return self.reward_of_episode.mean()

    def destroy_agent(self):
        """
        Destroys the reinforcement learning agent, in order to instantly release the memory the agent was using.
        """
        self._environment_instance.close()


