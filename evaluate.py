import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser

import gym
from gym.wrappers import FrameStack,  AtariPreprocessing

from src.config import config
from src.agents.dqn import DQNAgent
from src.callbacks.wandb import WandbEvaluationCallback

def run_evaluation(cfg):
    env_orig = gym.make("ALE/MsPacman-v5", frameskip=1, render_mode=cfg.ENV.RENDER_MODE)

    env = AtariPreprocessing(env_orig, terminal_on_life_loss=cfg.ENV.TERMINAL_ON_LIFE_LOSS)
    env = FrameStack(env, num_stack=4)

    tf.keras.backend.set_image_data_format('channels_first')
    if not cfg.ENV.PLAY_RANDOMLY:
        agent = DQNAgent(
            use_target=cfg.AGENT.USE_TARGET,
            update_target_after=cfg.AGENT.UPDATE_TAGRET_AFTER,
            tau=cfg.AGENT.TAU,
            double=cfg.AGENT.DOUBLE,
            prioritized_replay=cfg.AGENT.PRIORITIZED_REPLAY,
            dueling=cfg.AGENT.DUELING,
            noisy_nets=cfg.AGENT.NOISY_NETS,
            beta=cfg.AGENT.BETA,
            augment=cfg.AGENT.AUGMENT,
            discount_factor=cfg.AGENT.DISCOUNT_FACTOR,
            clip_rewards=cfg.AGENT.CLIP_REWARDS,
            record_video=cfg.AGENT.RECORD_VIDEO,
            log_table=cfg.AGENT.LOG_TABLE,
            log_table_period=cfg.AGENT.LOG_TABLE_PERIOD,
            evaluate_after=cfg.AGENT.EVALUATE_AFTER,
            evaluation_episodes=cfg.AGENT.EVALUATION_EPISODES,
            eps_start=cfg.AGENT.EPS_START,
            eps_end=cfg.AGENT.EPS_END,
            eps_eval=cfg.AGENT.EPS_EVAL,
            annealing_steps=cfg.AGENT.EPS_ANNEALING_STEPS,
            buffer_size=cfg.AGENT.BUFFER_SIZE,
            max_train_frames=cfg.AGENT.MAX_TRAIN_FRAMES,
            max_episode_frames=cfg.AGENT.MAX_EPISODE_FRAMES,
            warmup_frames=cfg.AGENT.WARMUP_FRAMES,
            batch_size=cfg.AGENT.BATCH_SIZE)
        agent.online_network.summary()

        loss_fn = keras.losses.Huber if cfg.TRAINING.LOSS == 'huber' else keras.losses.MeanSquaredError

        agent.compile(
            environment=env, 
            optimizer=keras.optimizers.Adam(
                learning_rate=cfg.TRAINING.LEARNING_RATE, 
                clipnorm=cfg.TRAINING.CLIPNORM), 
            loss=loss_fn(reduction=tf.keras.losses.Reduction.NONE))

        wandb_config = cfg.clone()
        del wandb_config['TRAINING']
        wandb_callback = WandbEvaluationCallback(cfg.EVALUATION.WANDB.PROJECT, cfg.EVALUATION.WANDB.NAME, cfg.EVALUATION.WANDB.GROUP, config=wandb_config)

        agent.load_checkpoint(cfg.EVALUATION.CHECKPOINT_PATH)
        agent.evaluate(callbacks=[wandb_callback], seeds=cfg.EVALUATION.SEEDS, frames=cfg.EVALUATION.MAX_FRAMES)
        
    env.close()
    env_orig.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_path', help='Path to the agent\'s configuration file')
    args = parser.parse_args()

    cfg = config.get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.freeze()
    run_evaluation(cfg)