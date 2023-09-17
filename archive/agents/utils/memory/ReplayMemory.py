import random
import numpy as np
from agents.utils.memory.Transition import Transition


class ReplayMemory(object):

    def __init__(self, capacity):
        """Create Replay buffer.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._capacity = capacity
        self._memory = []
        self._position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self._memory) < self._capacity:
            self._memory.append(None)
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        return random.sample(self._memory, min(len(self._memory), batch_size)), np.ones(batch_size)

    def sample_all(self):
        return self._memory

    def clear(self):
        self._memory = []
        self._position = 0

    def __len__(self):
        return len(self._memory)
