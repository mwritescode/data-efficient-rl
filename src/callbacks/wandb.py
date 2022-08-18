import pandas as pd
from abc import ABC
import wandb
import glob

class RLCallback(ABC):
    def __init__(self):
        super().__init__()
    
    def on_train_begin(self, logs=None):
        return logs
    
    def on_eval_begin(self, logs=None):
        return logs
    
    def on_train_episode_end(self, logs=None):
        return logs
    
    def on_eval_episode_end(self, logs=None):
        return logs

    def on_train_step_end(self, logs=None):
        return logs

    def on_train_end(self, logs=None):
        return logs
    
    def on_eval_end(self, logs=None):
        return logs

class WandbTrainingCallback(RLCallback):
    def __init__(self, project, name, group, config):
        super().__init__()
        self.run = wandb.init(
            project=project,
            name=name,
            group=group,
            reinit=True,
            config=config
        )
        self.vid_name = group
    
    def on_train_step_end(self, logs=None):
        frame = logs['train_frame']
        logs_to_wandb = {key: value for key, value in logs.items() if \
            key not in ['train_frame', 'is_warmup', 'table'] and value is not None}

        if 'table' in logs.keys():
            table = wandb.Table(columns=list(logs['table'].keys()))

            for key in logs['table'].keys():
                if 'frame' in key:
                    if len(logs['table'][key].shape) == 4:
                        logs['table'][key] = [wandb.Image(img[-1]) for img in logs['table'][key]]
                    else:
                        logs['table'][key] = wandb.Image(logs['table'][key])
            table.add_data(*[value for _, value in logs['table'].items()])
            logs_to_wandb['table'] = table

        wandb.log(logs_to_wandb, step=frame)
        return logs

    def on_train_episode_end(self, logs=None):
        logs_to_wandb = {key:value for key, value in logs.items() if 'episode' in key}
        wandb.log(logs_to_wandb)
        return logs

    def on_eval_end(self, logs=None):
        logs_to_wandb = {key: value for key, value in logs.items () if key not in ['train_frame', 'log_videos']}
        wandb.log(logs_to_wandb)
        
        if logs['log_videos']:
            all_videos = glob.glob(f'videos/current_ep_{self.vid_name}.mp4')
            wandb.log({"gameplays": wandb.Video(
                all_videos[0], 
                caption=f'evaluation episode after {logs["train_frame"]} training frames', 
                fps=24, 
                format="gif"), "step": logs['train_frame']})

    def on_train_end(self, logs=None):
        self.run.finish()
        return logs


class WandbEvaluationCallback(RLCallback):
    def __init__(self, project, name, group, config):
        super().__init__()
        self.run = wandb.init(
            project=project,
            name=name,
            group=group,
            reinit=True,
            config=config
        )
        wandb.define_metric("eval_episode_return", summary="mean")
        wandb.define_metric("eval_episode_len", summary="mean")
    
    def on_eval_episode_end(self, logs=None):
        wandb.log(logs)
        return logs
    
    def on_eval_end(self, logs=None):
        self.run.finish()
        return logs


