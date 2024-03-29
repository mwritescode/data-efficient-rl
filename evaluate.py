import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser

import gym
from gym.wrappers import FrameStack,  AtariPreprocessing

from src.config import config
from src.agents.dqn import DQNAgent
from src.agents.random import RandomAgent
from src.callbacks.wandb import WandbEvaluationCallback

def run_evaluation(cfg):
    for seed in cfg.EVALUATION.SEEDS:
        env_orig = gym.make(
            "ALE/MsPacman-v5", 
            frameskip=1, 
            render_mode=cfg.ENV.RENDER_MODE,
            repeat_action_probability=cfg.ENV.STICKY_ACTION_PROB)

        env = AtariPreprocessing(env_orig, terminal_on_life_loss=cfg.ENV.TERMINAL_ON_LIFE_LOSS)
        env = FrameStack(env, num_stack=4)

        cfg['EVALUATION']['SEED'] = seed
        wandb_config = cfg.clone()
        del wandb_config['TRAINING']

        if cfg.AGENT.PLAY_RANDOMLY:
            agent = RandomAgent(env=env)
            keys_to_delete = [key for key in wandb_config['AGENT'] if key != 'PLAY_RANDOMLY']
            for key in keys_to_delete:
                del wandb_config['AGENT'][key]
        else:
            tf.keras.backend.set_image_data_format('channels_first')
            agent = DQNAgent(
                name=cfg.EVALUATION.WANDB.GROUP,
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
                batch_size=cfg.AGENT.BATCH_SIZE,
                num_updates_per_step=cfg.AGENT.NUM_UPDATES_PER_STEP)
            agent.online_network.summary()

            loss_fn = keras.losses.Huber if cfg.TRAINING.LOSS == 'huber' else keras.losses.MeanSquaredError

            agent.compile(
                environment=env, 
                optimizer=keras.optimizers.Adam(
                    learning_rate=cfg.TRAINING.LEARNING_RATE, 
                    clipnorm=cfg.TRAINING.CLIPNORM), 
                loss=loss_fn(reduction=tf.keras.losses.Reduction.NONE))

            agent.load_checkpoint(cfg.EVALUATION.CHECKPOINT_PATH)

        wandb_callback = WandbEvaluationCallback(
            cfg.EVALUATION.WANDB.PROJECT, 
            cfg.EVALUATION.WANDB.NAME + f'-{seed}', 
            cfg.EVALUATION.WANDB.GROUP, 
            config=wandb_config)

        agent.evaluate(callbacks=[wandb_callback], seed=cfg.EVALUATION.SEED, frames=cfg.EVALUATION.MAX_FRAMES)
            
        env.close()
        env_orig.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_path', help='Path to the agent\'s configuration file')
    args = parser.parse_args()

    cfg = config.get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.freeze()
    if cfg.AGENT.PLAY_RANDOMLY:
        run_evaluation(cfg)
    else:
        for seed in cfg.TRAINING.SEEDS:
            new_cfg = cfg.clone()
            old_name = cfg.TRAINING.CHECKPOINT_PATH.split('/')[1]
            new_name = old_name + f'-{seed}'
            new_cfg['TRAINING']['SEED'] = seed
            new_cfg['EVALUATION']['CHECKPOINT_PATH'] = new_cfg['EVALUATION']['CHECKPOINT_PATH'].replace(old_name, new_name)
            new_cfg['EVALUATION']['WANDB']['NAME'] += str(seed)
            run_evaluation(new_cfg)