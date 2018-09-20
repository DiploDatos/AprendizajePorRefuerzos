import gym
import six
import numpy as np
from gym.envs.registration import register


class FrozenLakeAgent:

    def __init__(self):

        # basic configuration
        self._environment_name = "FrozenLake-v0"
        self._environment_instance = None  # (note that the "None" variables have values yet to be assigned)
        self.random_state = None  # type: np.random.RandomState
        self._cutoff_time = None
        self._hyper_parameters = None

        # whether ot not to display a video of the agent execution at each episode
        self.display_video = True

        # list that contains the amount of time-steps of the episode. It is used as a way to score the performance of
        # the agent.
        self.timesteps_of_episode = None

        # list that contains the amount of reward given to the agent in each episode
        self.reward_of_episode = None

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

    def restart_agent_learning(self):
        """
        Restarts the reinforcement learning agent so it starts learning from scratch, in order to avoid bias with
        previous learning experience.
        """
        # last run is cleared
        self.timesteps_of_episode = []
        self.reward_of_episode = []
        self.action_reward_state_trace = []

        # q values are restarted
        self.q = {}

    def init_agent(self, is_slippery=False):
        """
        Initializes the reinforcement learning agent with a default configuration.
        """

        if is_slippery:
            self._environment_instance = gym.make('FrozenLake-v0')
        else:
            # A Frozen Lake environment is registered with Slippery turned as False so it is deterministic
            register(id='FrozenLakeNotSlippery-v0',
                     entry_point='gym.envs.toy_text:FrozenLakeEnv',
                     kwargs={'map_name': '4x4', 'is_slippery': False},
                     max_episode_steps=100,
                     reward_threshold=0.78)

            self._environment_instance = gym.make('FrozenLakeNotSlippery-v0')

        self.actions = range(self._environment_instance.action_space.n)

        # environment is seeded
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        if self.display_video:
            # video_callable=lambda count: count % 10 == 0)
            self._environment_instance = gym.wrappers.Monitor(self._environment_instance,
                                                              '/tmp/frozenlake-experiment-1',
                                                              force=True)

    def run(self):
        """
        Runs the reinforcement learning agent with a given configuration.
        """
        for i_episode in range(self.episodes_to_run):
            # an instance of an episode is run until it fails or until it reaches 200 time-steps

            # resets the environment, obtaining the first state observation
            observation = self._environment_instance.reset()

            # a number of four digits representing the actual state is obtained
            state = observation

            for t in range(self._cutoff_time):

                # Pick an action based on the current state
                action = self.choose_action(state)
                # Execute the action and get feedback
                observation, reward, done, info = self._environment_instance.step(action)

                # current state transition is saved
                self.action_reward_state_trace.append([action, reward, observation])

                # Digitize the observation to get a state
                next_state = observation

                if not done:
                    self.learn(state, action, reward, next_state)
                    state = next_state
                else:
                    if reward == 0:  # episode finished because the agent fell into a hole

                        # the default reward can be overrided by a hand-made reward (below) for example to punish the
                        # agent for falling into a hole
                        reward = 0  # replace this number to override the reward

                    self.learn(state, action, reward, next_state)
                    self.timesteps_of_episode = np.append(self.timesteps_of_episode, [int(t + 1)])
                    self.reward_of_episode = np.append(self.reward_of_episode, reward)
                    break

        return self.reward_of_episode.mean()

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
