from collections import namedtuple, deque
import numpy as np
import warnings
import random

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=1000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)

        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def consecutive_sample(self, batch_size=32):
        samples = []
        for i in range(int(batch_size/2)):
            rand_num = random.randint(0, len(self.replay_memory)-2)
            samples.append(rand_num)
            samples.append(rand_num+1)
        #print(samples)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size
