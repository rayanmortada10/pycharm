import collections
import random

class ReplayBuffer:
    def __init__(self, capacity=5000):
        # Create a deque with a maxlen
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, transition):
        """Appends a transition tuple to the buffer."""
        self.buffer.append(transition)  # Efficiently adds to the right (end) of buffer

    def sample(self, batch_size):
        """
        Returns a random mini-batch of transitions.
        Usage: my_batch = buffer.sample(32)
        """
        return random.sample(self.buffer, batch_size)

    def __getitem__(self, idx):
        """
        Enables buffer[idx] indexing, similar to lists.
        """
        return self.buffer[idx]

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)