from abc import ABC, abstractmethod
from ..utils.exploration import EpsGreedy

# TODO: maybe re-name evaluation_step in run_eval_episode and train_step in run_train_episode (?)
 
class RLBaseAgent(ABC):
    def __init__(self, evaluate_after=0, eps_start=1.0, eps_end=0.1, eps_eval=0.05, max_train_frames=1000000, max_episode_frames=108000, warmup_frames=1600, batch_size=32, annealing_steps=50000):
        super().__init__()
        self.current_frame_num = 0
        self.current_eval_frame_num = 0
        self.evaluate_period = evaluate_after
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_eval = eps_eval
        self.batch_size = batch_size
        self.max_train_frames = max_train_frames
        self.max_frames = max_train_frames + warmup_frames
        self.max_episode_frames = max_episode_frames
        self.warmup_frames = warmup_frames
        self.annealing_steps = annealing_steps
 
    @abstractmethod
    def train_step(self, max_frames):
        # This is a whole episode training
        pass
    
    @abstractmethod
    def evaluation_step(self):
        # This is a whole episode evaluation
        pass
    
    def fit(self, seed=None, callbacks=[]):
        for callback in callbacks:
            callback.on_train_begin()
        episode_num = 1
        state = self.env.reset(seed=seed)
        while self.current_frame_num < self.max_frames:
            logs = self.train_step(init_state=state, episode_num=episode_num)
            state = self.env.reset()
            for callback in callbacks:
                callback.on_train_episode_end(logs)
            episode_num += 1
        for callback in callbacks:
            callback.on_train_end()

    def evaluate(self, frames, callbacks=[]):
        for callback in callbacks:
            callback.on_eval_begin()
        self.current_eval_frame_num = 0
        episode_num = 1
        while self.current_eval_frame_num < frames:
            state = self.env.reset()
            logs = self.evaluation_step(state)
            for callback in callbacks:
                callback.on_eval_episode_end(logs)
            episode_num += 1
        for callback in callbacks:
            callback.on_eval_end()
        # TODO: write evaluation step after the train one, as done in dr.Q
    
    def compile(self, environment, optimizer, loss):
        self.env = environment
        self.loss_fn = loss
        self.output_dim = environment.action_space.n
        self.optimizer = optimizer
        self.exploration_policy = EpsGreedy(self.output_dim, eps_start=self.eps_start, eps_end=self.eps_end, eps_eval=self.eps_eval, annealing_steps=self.annealing_steps, warmup_steps=self.warmup_frames)
        