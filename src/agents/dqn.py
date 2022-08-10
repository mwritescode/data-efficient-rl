import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow import keras
from ..utils.nets import get_network
from ..utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from .base import RLBaseAgent

#TODO: check the default values against those in the literature

class DQNAgent(RLBaseAgent):
    def __init__(self, buffer_size=100000, double=False, prioritized_replay=False, dueling=False, noisy_nets=False, use_target=False, update_target_after=10000, discount_factor=0.99, clip_rewards=True, beta=0.4, **kwargs):
        super().__init__(**kwargs)
        network_name = 'dueling' if dueling else 'dqn'
        if noisy_nets:
            network_name += '_noisy'
        self.noisy_nets = noisy_nets
        self.online_network = get_network(network_name)
        self.use_target = True if double else use_target
        if self.use_target:
            self.target_network = get_network(network_name)
            self.set_target_weights()
            self.update_taraget_after = update_target_after
        self.prioritized_replay = prioritized_replay
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size) if not prioritized_replay else PrioritizedReplayBuffer(buffer_size=buffer_size)
        self.gamma = discount_factor
        self.clip_rewards = clip_rewards
        self.double = double
        self.beta = beta
        self.beta_decay = (1.0 - beta) / self.max_train_frames
    
    def set_target_weights(self):
        self.target_network.set_weights(deepcopy(self.online_network.get_weights()))

    def train_step(self, init_state, episode_num):
        logs = {'train_loss': None, 'train_episode_return': 0.0, 'episode': episode_num, 'episode_length': 1}
        state = init_state
        while logs['episode_length'] < self.max_episode_frames:
            next_state, reward, done, _ = self.execute_one_action(state)
            logs['train_episode_return'] += reward
            logs['episode_length'] += 1
            if done:
                break
            elif self.current_frame_num > self.warmup_frames:
                # 1. Sample batch experiences
                idxs, infos, weights = self.replay_buffer.sample(batch_size=self.batch_size, beta=self.beta)
                states, actions, rewards, next_states, dones = infos
                if self.prioritized_replay:
                    self.beta += self.beta_decay

                # Clip rewards to be +1 if positive and -1 if negative
                if self.clip_rewards:
                    rewards = np.sign(rewards)

                # 2. Estimate target Q-values
                if not self.double:
                    if self.use_target:
                        next_q_values = self.target_network.predict(next_states, verbose=0)
                    else:
                        next_q_values = self.online_network.predict(next_states, verbose=0)
                    target_q = tf.where(dones, rewards, rewards + self.gamma * np.max(next_q_values, axis=1))
                else:
                    target_next_q = self.target_network.predict(next_states, verbose=0)
                    online_next_q = self.online_network.predict(next_states, verbose=0)
                    col_ids = tf.argmax(online_next_q, axis=1)
                    row_ids = tf.range(self.batch_size, dtype=tf.int64)
                    ids = tf.stack([row_ids, col_ids], axis=1)
                    target_q = tf.where(dones, rewards, rewards + self.gamma * tf.gather_nd(target_next_q, ids))
                mask = tf.one_hot(actions, self.output_dim)   

                # 3. Compute current Q-values         
                with tf.GradientTape() as tape:
                    current_q = tf.boolean_mask(self.online_network(states), mask)
                    elementwise_loss = self.loss_fn(tf.expand_dims(target_q, axis=-1), tf.expand_dims(current_q, axis=-1))
                    loss = tf.reduce_mean(elementwise_loss*weights)

                # 4. Update network weights
                trainable_vars = self.online_network.trainable_variables
                grads = tape.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients(zip(grads, trainable_vars))
                if self.prioritized_replay:
                    self.replay_buffer.update_priorities(idxs, new_priorities=elementwise_loss.numpy())
                logs['train_loss'] = logs['train_loss'] + loss.numpy() if logs['train_loss'] else loss.numpy()

                if  self.use_target and self.current_frame_num % self.update_taraget_after == 0:
                    self.set_target_weights()
                
                if self.noisy_nets:
                    self.online_network.reset_noise()
                    if self.use_target:
                        self.target_network.reset_noise()
                print('Training frame')
            else:
                print('Warmup frame')
            state = next_state

        if logs['train_loss']:
            logs['train_loss'] /= logs['episode_length']
        return logs

        
    def evaluation_step(self):
        pass

    def execute_one_action(self, state):
        state = np.array(state).squeeze()
        q_values = self.online_network.predict(state[np.newaxis, :], verbose=0)
        if self.noisy_nets:
            action = np.argmax(q_values)
        else:
            action = self.exploration_policy.select_action(q_values[0], step=self.current_frame_num)
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.append(state, action, reward, np.array(next_state).squeeze(), done)
        self.current_frame_num += 1
        return next_state, reward, done, info
    

        