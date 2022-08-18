import os
import pandas as pd
import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.current_frame = 0
        self.action_space_dim = self.env.action_space.n

    def evaluate(self, seed, frames, callbacks=[]):

        print(f'Evaluating for seed {seed}')

        _ = self.env.reset(seed=seed)
        
        ep_return = 0.0
        ep_len = 0
        for self.current_frame in range(frames+1):
            action = np.random.randint(low=0, high=self.action_space_dim)
            _, reward, done, _ = self.env.step(action)
            ep_len += 1
            ep_return += reward
            if done:
                for callback in callbacks:
                    callback.on_eval_episode_end(logs={
                        'eval_episode_return': ep_return,
                        'eval_episode_len': ep_len
                    })
                ep_len = 0
                ep_return = 0.0
                _ = self.env.reset()
