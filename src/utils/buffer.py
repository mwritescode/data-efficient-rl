import random
import numpy as np
from math import log, ceil
from collections import deque
from abc import ABC, abstractmethod
from src.utils.sum_tree import SumTree

class AbstractBuffer(ABC):
    def __init__(self, buffer_size):
        super().__init__()
        self.memory = deque(maxlen=buffer_size)

    def append(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
    
    @abstractmethod
    def sample(self, batch_size, **kwrags):
        pass
    
    def __len__(self):
        return len(self.memory)


class ReplayBuffer(AbstractBuffer):
    def __init__(self, buffer_size):
        super().__init__(buffer_size=buffer_size)
    
    def sample(self, batch_size, **kwargs):
        idxs = np.random.choice(len(self.memory), size=batch_size, replace=False)
        out = [np.stack([self.memory[idx][elem] for idx in idxs]) for elem in range(5)]
        return idxs, out, np.ones(shape=(batch_size,))

class PrioritizedReplayBuffer(AbstractBuffer):
    def __init__(self, buffer_size, alpha=0.6, eps=0.01):
        super().__init__(buffer_size=buffer_size)
        capacity = 2 ** ceil(log(buffer_size, 2))
        self.sum_tree = SumTree(capacity=capacity)
        self.max_priority = 1.0
        self.current_idx = 0
        self.eps = eps
        self.alpha = alpha

    def append(self, state, action, reward, next_state, done):
        super().append(state, action, reward, next_state, done)

        # When the transition is first added to the buffer, it should have max priority
        self.sum_tree.leaf_nodes[self.current_idx].update_value(value=self.max_priority ** self.alpha)
        self.current_idx = (self.current_idx + 1) % len(self.memory)

    def sample(self, batch_size, beta=0.4):
        idxs, weights = self._get_idxs_and_weights(batch_size=batch_size, beta=beta)
        out = [np.stack([self.memory[idx][elem] for idx in idxs]) for elem in range(5)]
        return idxs, out, weights

    def _get_idxs_and_weights(self, batch_size, beta):
        p_sum = self.sum_tree.head.value
        bounds = np.linspace(0, p_sum, batch_size + 1)
        idxs = []
        weights = []
        for lower, upper in zip(bounds[::1], bounds[1::1]):
            sampled_val = random.uniform(lower, upper)
            node = self.sum_tree.retrieve_node(sampled_val)
            idxs.append(node.idx)
            weights.append((node.value/p_sum) * len(self.memory))
        weights = np.array(weights) ** (-beta)
        weights = weights / np.max(weights)
        return idxs, weights

    def update_priorities(self, indexes, new_priorities):
        for idx, priority in zip(indexes, new_priorities):
            self.sum_tree.leaf_nodes[idx].update_value(value=(priority + self.eps) ** self.alpha)