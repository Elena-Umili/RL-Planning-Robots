from collections import namedtuple, deque
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 #field_names=['state', 'action', 'reward', 'done', 'next_state'])
                                 field_names=['s_0', 'a_1', 'r_1', 'd_1', 's_1', 'a_2', 'r_2', 'd_2', 's_2'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, s_0, a_1, r_1, d_1, s_1, a_2, r_2, d_2, s_2):
        self.replay_memory.append(
            self.Buffer(s_0, a_1, r_1, d_1, s_1, a_2, r_2, d_2, s_2))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size