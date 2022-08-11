from abc import ABC
import wandb

class RLCallback(ABC):
    def __init__(self):
        super().__init__()
    
    def on_train_begin(self, logs=None):
        return logs
    
    def on_eval_begin(self, logs=None):
        return logs
    
    def on_train_episode_end(self, logs=None):
        return logs
    
    def on_eval_episode_begin(self, logs=None):
        return logs

    def on_train_end(self, logs=None):
        return logs
    
    def on_eval_end(self, logs=None):
        return logs

class WandbCallback(RLCallback):
    def __init__(self, project, name, group, config):
        super().__init__()
        self.run = wandb.init(
            project=project,
            name=name,
            group=group,
            reinit=True,
            config=config
        )
    
    def on_train_episode_end(self, logs=None):
        if not logs['is_warmup']:
            episode = logs['episode']
            logs_to_wandb = {key: value for key, value in logs.items() if key != 'episode' and value is not None}
            for score_name, score_value in logs_to_wandb.items():
                wandb.log({score_name: score_value, 'episode': episode}, step=episode)
        return logs

    def on_eval_end(self, logs=None):
        self.run.finish()
        return logs

