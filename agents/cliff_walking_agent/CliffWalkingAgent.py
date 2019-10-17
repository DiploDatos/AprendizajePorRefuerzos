import gym
import six
import numpy as np


class CliffWalkingAgent:

    def __init__(self):

        # basic configuration
        self._environment_name = "CliffWalking-v0"
        self._environment_instance = None
        self.random_state = None
        self._cutoff_time = None
        self._hyper_parameters = None

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
        self.episodes_to_run = 500  # amount of episodes to run for each run of the agent
        self.actions = None

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

        # q values are restarted
        self.q = {}

    def init_agent(self):
        """
        Initializes the reinforcement learning agent with a default configuration.
        """

        self._environment_instance = gym.make(self._environment_name)

        self.actions = range(self._environment_instance.action_space.n)

        # environment is seeded
        if self.random_state is None:
            self.random_state = np.random.RandomState()

    def run(self):
        """
        Runs the reinforcement learning agent with a given configuration.
        """
        for i_episode in range(self.episodes_to_run):
            # an instance of an episode is run until it fails or until it reaches 200 time-steps

            # resets the environment, obtaining the first state observation
            state = self._environment_instance.reset()

            episode_reward = 0
            done = False
            t = 0

            while not done:

                # Pick an action based on the current state
                action = self.choose_action(state)
                # Execute the action and get feedback
                observation, reward, done, info = self._environment_instance.step(action)

                if done:
                    reward = +1
                elif reward == -1:
                    reward = 0
                
                episode_reward += reward

                next_state = observation

                if not done:
                    self.learn(state, action, reward, next_state)
                    state = next_state
                else:
                    self.learn(state, action, reward, next_state)
                    self.timesteps_of_episode = np.append(self.timesteps_of_episode, [int(t + 1)])
                    self.reward_of_episode = np.append(self.reward_of_episode, max(episode_reward, -100))

                t += 1

        return self.reward_of_episode.mean()

    def choose_action(self, state):
        """
        Chooses an action according to the learning previously performed
        """
        q = [self.q.get((state, a), 0.0) for a in self.actions]
        max_q = max(q)

        if self.random_state.uniform() < self._epsilon:
            return self.random_state.choice(self.actions)  # a random action is selected

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
        new_max_q = max([self.q.get((next_state, a), 0.0) for a in self.actions]) if next_state else 0
        old_value = self.q.get((state, action), 0.0)

        self.q[(state, action)] = old_value + self._alpha * (reward + self._gamma * new_max_q - old_value)

    def destroy_agent(self):
        """
        Destroys the reinforcement learning agent, in order to instantly release the memory the agent was using.
        """
        self._environment_instance.close()
