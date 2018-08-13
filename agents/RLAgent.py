
import abc


class RLAgent(object):
    """
    Abstract base class that provides an interface of a RL agent that receives a set of hyper-parameters and a
    configuration as input and returns a measure of the run as an output.
    """
    __metaclass__ = abc.ABCMeta

    # core methods -------------------------------------------------------------

    @abc.abstractmethod
    def set_random_state(self, random_state):
        """
        Method that sets a previously created numpy.RandomState object in order to share the same random seed.
        :param random_state: an instantiated np.RandomState object
        """
        raise NotImplementedError('this is an abstract method; must be overriden')

    @abc.abstractmethod
    def set_cutoff_time(self, cutoff_time):
        """
        Method that sets a maximum number of time-steps for each agent episode.
        :param cutoff_time:
        """
        raise NotImplementedError('this is an abstract method; must be overriden')

    @abc.abstractmethod
    def set_hyper_parameters(self, hyper_parameters):
        """
        Method that passes the hyper_parameter configuration vector to the RL agent.
        :param hyper_parameters: a list containing the hyper-parameters that are to be set in the RL algorithm.
        """
        raise NotImplementedError('this is an abstract method; must be overriden')

    @abc.abstractmethod
    def init_agent(self):
        """
        Initializes the reinforcement learning agent with a default configuration.
        """
        raise NotImplementedError('this is an abstract method; must be overriden')

    @abc.abstractmethod
    def restart_agent_learning(self):
        """
        Restarts the reinforcement learning agent so it starts learning from scratch, in order to avoid bias with
        previous learning experience.
        """
        raise NotImplementedError('this is an abstract method; must be overriden')

    @abc.abstractmethod
    def run(self):
        """
        Runs the reinforcement learning agent with a previously given configuration.
        """
        raise NotImplementedError('this is an abstract method; must be overriden')

    @abc.abstractmethod
    def destroy_agent(self):
        """
        Destroys the reinforcement learning agent, in order to instantly release the memory the agent was using.
        """
        raise NotImplementedError('this is an abstract method; must be overriden')

    # --------------------------------------------------------------------------
