import tensorflow as tf
import numpy as np
import os
from .models import QNetwork, GaussianPolicy
from .utils import soft_update, hard_update

class SAC:
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.policy_type = args['policy']
        self.target_update_interval = args['target_update_interval']
        self.lr = args['lr']

        self.critic = QNetwork(num_inputs, action_space.shape[0], args['hidden_size'])
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args['hidden_size'])
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args['hidden_size'], action_space)
        self.policy.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

    def select_action(self, state, evaluate=False):
        state = tf.convert_to_tensor(state, dtype=tf.float32)[tf.newaxis, :]
        action, _, _ = self.policy.sample(state)
        return action.numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)[:, tf.newaxis]
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)[:, tf.newaxis]

        with tf.GradientTape(persistent=True) as tape:
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = tf.minimum(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

            qf1, qf2 = self.critic(state_batch, action_batch)
            qf1_loss = tf.reduce_mean(tf.square(qf1 - next_q_value))
            qf2_loss = tf.reduce_mean(tf.square(qf2 - next_q_value))
            qf_loss = qf1_loss + qf2_loss

            pi, log_pi, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = tf.minimum(qf1_pi, qf2_pi)
            policy_loss = tf.reduce_mean(self.alpha * log_pi - min_qf_pi)

        critic_grads = tape.gradient(qf_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.numpy(), qf2_loss.numpy(), policy_loss.numpy()

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = f"checkpoints/sac_checkpoint_{env_name}_{suffix}"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        self.policy.save_weights(f"{ckpt_path}_policy.weights.h5")
        self.critic.save_weights(f"{ckpt_path}_critic.weights.h5")
        self.critic_target.save_weights(f"{ckpt_path}_critic_target.weights.h5")

    def load_checkpoint(self, ckpt_path, evaluate=False):
        self.policy.load_weights(f"{ckpt_path}_policy.weights.h5")
        self.critic.load_weights(f"{ckpt_path}_critic.weights.h5")
        self.critic_target.load_weights(f"{ckpt_path}_critic_target.weights.h5")

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.policy.save_weights(os.path.join(save_dir, "actor.weights.h5"))
        self.critic.save_weights(os.path.join(save_dir, "critic.weights.h5"))