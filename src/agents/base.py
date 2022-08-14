import tqdm
import numpy as np
from abc import ABC, abstractmethod
from ..utils.exploration import EpsGreedy
 
class RLBaseAgent(ABC):
    def __init__(self, record_video=False, log_table=False, log_table_period=100, evaluate_after=300, evaluation_episodes=5, eps_start=1.0, eps_end=0.1, eps_eval=0.05, max_train_frames=1000000, max_episode_frames=108000, warmup_frames=1600, batch_size=32, annealing_steps=50000):
        super().__init__()
        self.current_frame_num = 0
        self.eval_period = evaluate_after
        self.n_eval_episodes = evaluation_episodes
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_eval = eps_eval
        self.batch_size = batch_size
        self.max_train_frames = max_train_frames
        self.max_frames = max_train_frames + warmup_frames
        self.max_episode_frames = max_episode_frames
        self.warmup_frames = warmup_frames
        self.annealing_steps = annealing_steps
        self.record_video = record_video
        self.log_table = log_table
        self.log_table_period = log_table_period
    
    @abstractmethod
    def train_step(self, max_frames):
        # This is a single training step
        pass
    
    @abstractmethod
    def evaluation_episode(self, state, max_frames=None, current_frame=0):
        # This is a whole episode evaluation
        pass

    @abstractmethod
    def execute_one_action(self, state, train=True):
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_path):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path):
        pass
    
    def fit(self, seed=None, callbacks=[]):
        for callback in callbacks:
            callback.on_train_begin()
        additional_logs = {'episode_num': 0, 'episode_len': 0, 'episode_return': 0.0}
        state = self.env.reset(seed=seed)
        # Repeat until max_frames is reached
        for self.current_frame_num in tqdm.tqdm(range(self.max_frames+1)):
            # Execute one step in the environment
            next_state, reward, done, info = self.execute_one_action(state)
            additional_logs['episode_return'] += reward
            additional_logs['episode_len'] += 1

            # If warmup is done train for one step
            if self.current_frame_num > self.warmup_frames:
                logs = self.train_step()

                if self.log_table and \
                    (self.current_frame_num - self.warmup_frames) % self.log_table_period == 0:
                    logs['table']['actual_frame'] = info['rgb']
                    logs['table']['seen_frame'] = np.array(state)[-1]
                for callback in callbacks:
                    callback.on_train_step_end(logs)

            if done or additional_logs['episode_len'] > self.max_episode_frames:
                if self.current_frame_num > self.warmup_frames:
                    for callback in callbacks:
                        callback.on_train_episode_end({**logs, **additional_logs})
                additional_logs['episode_num'] += 1
                additional_logs['episode_len'] = 0
                additional_logs['episode_return'] = 0.0
                state = self.env.reset()    
            else:
                state = next_state

            if (self.current_frame_num - self.warmup_frames) % self.eval_period == 0 and \
                self.current_frame_num > self.warmup_frames:
                for callback in callbacks:
                    callback.on_eval_begin()

                state = self.env.reset()
                episode_returns = []
                for ep_num in tqdm.tqdm(range(self.n_eval_episodes)):
                    ep_return, _ = self.evaluation_episode(state, is_recording=ep_num==1 and self.record_video)
                    state = self.env.reset()
                    episode_returns.append(ep_return)
                
                eval_logs = {
                    'mean_episode_return': np.mean(episode_returns),
                    'max_episode_return': max(episode_returns),
                    'min_episode_return': min(episode_returns),
                    'log_videos': self.record_video,
                    'train_frame': self.current_frame_num - self.warmup_frames}
                
                for callback in callbacks:
                    callback.on_eval_end(eval_logs)

        for callback in callbacks:
            callback.on_train_end()

    def evaluate(self, frames, seeds, callbacks):
        logs = {
            'average_return': [],
            'max_return': [],
            'min_return': [],
            'num_episodes': []
        }
        for seed in tqdm.tqdm(seeds):
            episode_num = 1
            frame_n = 0
            episode_returns = []
            state = self.env.reset(seed=seed)
            while frame_n < frames:
                episode_return, frame_n = self.evaluation_episode(state, max_frames=frames, current_frame=frames)
                episode_num += 1
                episode_returns.append(episode_return)
                state = self.env.reset()
            
            logs['average_return'].append(np.mean(episode_returns))
            logs['max_return'].append(max(episode_returns))
            logs['min_return'].append(min(episode_returns))
            logs['num_episodes'].append(episode_num)
        
        for callback in callbacks:
            callback.on_eval_end(logs)
        
  
    
    def compile(self, environment, optimizer, loss):
        self.env = environment
        self.loss_fn = loss
        self.output_dim = environment.action_space.n
        self.optimizer = optimizer
        self.exploration_policy = EpsGreedy(
            self.output_dim, 
            eps_start=self.eps_start, 
            eps_end=self.eps_end, 
            eps_eval=self.eps_eval, 
            annealing_steps=self.annealing_steps, 
            warmup_steps=self.warmup_frames)
        