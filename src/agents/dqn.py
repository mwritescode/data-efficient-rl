import os
import numpy as np
import tensorflow as tf
from copy import deepcopy
import moviepy.editor as mpy
from ..utils.nets import get_network
from ..utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from ..utils.augment import augment_batch
from .base import RLBaseAgent

#TODO: check the default values against those in the literature

class DQNAgent(RLBaseAgent):
    def __init__(
        self, 
        buffer_size=100000, 
        double=False, 
        prioritized_replay=False, 
        dueling=False, 
        noisy_nets=False, 
        augment=False, 
        use_target=False, 
        update_target_after=10000, 
        tau=None, 
        discount_factor=0.99, 
        clip_rewards=True, 
        beta=0.4, 
        **kwargs):
        super().__init__(**kwargs)
        network_name = 'dueling' if dueling else 'dqn'
        if noisy_nets:
            network_name += '_noisy'
        self.noisy_nets = noisy_nets
        self.online_network = get_network(network_name)
        self.use_target = True if double else use_target
        self.tau = tau
        if self.use_target:
            self.target_network = get_network(network_name)
            self.set_target_weights()
            self.update_taraget_after = update_target_after
        self.prioritized_replay = prioritized_replay
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size) if not prioritized_replay else PrioritizedReplayBuffer(buffer_size=buffer_size)
        self.gamma = tf.cast(discount_factor, tf.float64)
        self.clip_rewards = clip_rewards
        self.double = double
        self.beta = beta
        self.augment = augment
        self.beta_decay = (1.0 - beta) / self.max_train_frames
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = os.path.normpath(checkpoint_path)

        # Load online network weights
        self.online_network.load_weights(checkpoint_path)        

    def save_checkpoint(self, checkpoint_path):
        checkpoint_path = os.path.normpath(checkpoint_path)
        folder_name = os.path.dirname(checkpoint_path)
        os.makedirs(folder_name, exist_ok=True)

        # Save online network weights
        self.online_network.save_weights(checkpoint_path)
    
    def set_target_weights(self):
        online_net_weights = deepcopy(self.online_network.get_weights())
        if self.tau is not None:
            new_weights = [self.tau * target_w + (1-self.tau) * online_w for \
                target_w, online_w in zip(self.target_network.get_weights(), online_net_weights)]
        else:
            new_weights = online_net_weights
        self.target_network.set_weights(new_weights)

    def train_step(self):
        logs = {
            'train_loss': 0.0, 
            'target_q': 0.0}

        if self.log_table and \
            (self.current_frame_num - self.warmup_frames) % self.log_table_period == 0:
            logs['table'] = {
                'beta': self.beta
            }

        # 1. Sample batch experiences
        if self.noisy_nets:
            self.online_network.reset_noise()
            if self.use_target:
                self.target_network.reset_noise()

        idxs, infos, weights = self.replay_buffer.sample(batch_size=self.batch_size, beta=self.beta)
        states, actions, rewards, next_states, dones = infos

        if self.augment:
            old_states = states
            states = augment_batch(states)
            next_states = augment_batch(next_states)
            if self.log_table and \
                (self.current_frame_num - self.warmup_frames) % self.log_table_period == 0:
                logs['table']['augmented_frames'] = states.numpy()[:5,:]
                logs['table']['non_augmented_frames'] = old_states[:5,:]

        if self.prioritized_replay:
            self.beta += self.beta_decay

        # Clip rewards to be +1 if positive and -1 if negative
        if self.clip_rewards:
            rewards = np.sign(rewards)

        loss, elementwise_loss, target_q = self._use_gradient_tape(
            states=tf.convert_to_tensor(states), 
            actions=tf.convert_to_tensor(actions), 
            rewards=tf.convert_to_tensor(rewards, tf.float64), 
            next_states=tf.convert_to_tensor(next_states), 
            dones=tf.convert_to_tensor(dones), 
            weights=tf.convert_to_tensor(weights, tf.float64))

        logs['target_q'] = target_q[0]

        if self.prioritized_replay:
            self.replay_buffer.update_priorities(idxs, new_priorities=elementwise_loss.numpy())
        logs['train_loss'] += loss.numpy()

        if  self.use_target and self.current_frame_num % self.update_taraget_after == 0:
            self.set_target_weights()
        
        logs['train_frame'] = self.current_frame_num - self.warmup_frames
        return logs
    
    @tf.function
    def _use_gradient_tape(self, states, actions, rewards, next_states, dones, weights):
        # 2. Estimate target Q-values
        if not self.double:
            if self.use_target:
                next_q_values = tf.cast(self.target_network(next_states), tf.float64)
            else:
                next_q_values = tf.cast(self.online_network(next_states), tf.float64)
            target_q = tf.where(dones, rewards, rewards + self.gamma * tf.reduce_max(next_q_values, axis=1))
        else:
            target_next_q = tf.cast(self.target_network(next_states), tf.float64)
            online_next_q = self.online_network(next_states)
            col_ids = tf.argmax(online_next_q, axis=1)
            row_ids = tf.range(self.batch_size, dtype=tf.int64)
            ids = tf.stack([row_ids, col_ids], axis=1)
            target_q = tf.where(dones, rewards, rewards + self.gamma * tf.gather_nd(target_next_q, ids))
        mask = tf.one_hot(actions, self.output_dim)

        # 3. Compute current Q-values 
        with tf.GradientTape() as tape:
            current_q = tf.boolean_mask(self.online_network(states), mask)
            elementwise_loss = self.loss_fn(tf.expand_dims(target_q, axis=-1), tf.expand_dims(current_q, axis=-1))
            loss = tf.reduce_mean(tf.cast(elementwise_loss, tf.float64)*weights)

        # 4. Update network weights
        trainable_vars = self.online_network.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss, elementwise_loss, target_q
        
    def evaluation_episode(self, state, max_frames=None, current_frame=0, is_recording=False):
        episode_return = 0.0
        max_frames = self.max_episode_frames if max_frames is None else max_frames
        if is_recording:
            os.makedirs('videos', exist_ok=True)
            frames = []
        while current_frame < max_frames:
            next_state, reward, done, info = self.execute_one_action(state, train=False)
            if is_recording:
                frames.append(info['rgb'])
            current_frame += 1
            episode_return += reward
            if done:
                break
            state = next_state

        if is_recording:
            episode_vid = mpy.ImageSequenceClip(frames, fps=24)
            episode_vid.write_videofile("videos/current_ep.mp4", logger=None)
            
        return episode_return, current_frame

    def execute_one_action(self, state, train=True):
        state = np.array(state).squeeze()
        q_values = self.online_network.predict(state[np.newaxis, :], verbose=0)
        if self.current_frame_num < self.warmup_frames:
            action = np.random.randint(low=0, high=self.output_dim)
        elif self.noisy_nets and train: 
            action = np.argmax(q_values)
        else:
            action = self.exploration_policy.select_action(
                q_values[0], 
                step=self.current_frame_num - self.warmup_frames, 
                train=train)
        next_state, reward, done, info = self.env.step(action)
        if train:
            self.replay_buffer.append(state, action, reward, np.array(next_state).squeeze(), done)

        return next_state, reward, done, info
    

        