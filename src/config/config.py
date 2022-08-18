from yacs.config import CfgNode as CN
import os

_C = CN()

_C.ENV = CN()
_C.ENV.TERMINAL_ON_LIFE_LOSS = False
_C.ENV.RENDER_MODE = 'rgb_array'
_C.ENV.STICKY_ACTION_PROB = 0.0

_C.AGENT = CN()

_C.AGENT.PLAY_RANDOMLY = False

_C.AGENT.USE_TARGET = True
_C.AGENT.UPDATE_TAGRET_AFTER = 1
_C.AGENT.TAU = 0.99
_C.AGENT.DOUBLE = False
_C.AGENT.PRIORITIZED_REPLAY = False
_C.AGENT.DUELING = False
_C.AGENT.NOISY_NETS = False
_C.AGENT.BETA = 0.4
_C.AGENT.AUGMENT = False
_C.AGENT.DISCOUNT_FACTOR = 0.99
_C.AGENT.CLIP_REWARDS = True
_C.AGENT.RECORD_VIDEO = True
_C.AGENT.LOG_TABLE = True
_C.AGENT.LOG_TABLE_PERIOD = 100
_C.AGENT.EVALUATE_AFTER = 1000
_C.AGENT.EVALUATION_EPISODES = 5
_C.AGENT.EPS_START = 1.0
_C.AGENT.EPS_END = 0.1
_C.AGENT.EPS_EVAL = 0.05
_C.AGENT.EPS_ANNEALING_STEPS = 50000
_C.AGENT.BUFFER_SIZE = 100000
_C.AGENT.MAX_TRAIN_FRAMES = 100000
_C.AGENT.MAX_EPISODE_FRAMES = 108000
_C.AGENT.WARMUP_FRAMES = 1600
_C.AGENT.BATCH_SIZE = 32
_C.AGENT.NUM_UPDATES_PER_STEP = 1


_C.TRAINING = CN()
_C.TRAINING.SEEDS = [1, 2, 3]
_C.TRAINING.LEARNING_RATE = 0.0001
_C.TRAINING.CLIPNORM = 10.0
_C.TRAINING.LOSS = 'huber' # can also be 'mse'
_C.TRAINING.CHECKPOINT_PATH = 'checkpoints/run-name'

_C.TRAINING.WANDB = CN()
_C.TRAINING.WANDB.PROJECT = 'data-efficient-rl' 
_C.TRAINING.WANDB.GROUP = 'group-name'
_C.TRAINING.WANDB.NAME = 'run-name'


_C.EVALUATION = CN()
_C.EVALUATION.SEEDS = [42, 24, 48, 140, 768]
_C.EVALUATION.CHECKPOINT_PATH = 'checkpoints/run-name'
_C.EVALUATION.MAX_FRAMES = 125000
_C.EVALUATION.WANDB = CN()
_C.EVALUATION.WANDB.PROJECT = 'data-efficient-rl' 
_C.EVALUATION.WANDB.GROUP = 'group-name'
_C.EVALUATION.WANDB.NAME = 'run-name'


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  return _C.clone()

def save_cfg_default():
    """Save in a YAML file the default version of the configuration file, in order to provide a template to be modified."""
    dirpath = 'src/config/experiments'
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, 'dqn_default.yaml'), 'w') as f:
        f.write(_C.dump())
        f.flush()
        f.close()

if __name__ == '__main__':
    save_cfg_default()