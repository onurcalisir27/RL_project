import numpy as np
import tensorflow as tf
from scipy.optimize import minimize, LinearConstraint
import os
import random
from tqdm import tqdm
from models import RNetwork

class Method:
    def __init__(self, state_dim, objs, wp_id, save_name, config):
        self.state_dim = state_dim
        self.objs = np.array(objs, dtype=np.float32)
        self.wp_id = wp_id
        self.best_wp = []
        self.n_inits = 5
        self.lr = 1e-3
        self.hidden_size = 128
        self.n_models = 10
        self.models = []
        self.learned_models = []
        self.n_eval = 100

        self.action_dim = config['task']['task']['action_space']
        self.exploration_epoch = config['task']['task']['exploration_epoch']
        self.ensemble_sampling_epoch = config['task']['task']['ensemble_sampling_epoch']
        self.averaging_noise_epoch = config['task']['task']['averaging_noise_epoch']

        for _ in range(self.n_models):
            critic = RNetwork(self.state_dim * self.wp_id + len(objs), hidden_dim=self.hidden_size)
            critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                         loss='mse')
            self.models.append(critic)

        save_dir = 'models/' + save_name

        for wp_id in range(1, self.wp_id):
            models = []
            for idx in range(self.n_models):
                critic = RNetwork(wp_id * self.state_dim + len(objs), hidden_dim=self.hidden_size)
                critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
                # Build the model before loading weights
                dummy_input = tf.zeros((1, wp_id * self.state_dim + len(objs)))
                critic(dummy_input)
                critic.load_weights(os.path.join(save_dir, f'model_{wp_id}_{idx}.weights.h5'))
                models.append(critic)
            self.learned_models.append(models)

        self.best_traj = self.action_dim * (np.random.rand(self.state_dim * self.wp_id) - 0.5)
        self.best_reward = -np.inf
        self.lincon = LinearConstraint(np.eye(self.state_dim), -self.action_dim, self.action_dim)

    def traj_opt(self, i_episode, objs):
        self.reward_idx = random.choice(range(self.n_models))
        self.objs = np.array(objs, dtype=np.float32)
        self.curr_episode = i_episode

        self.traj = []
        for idx in range(1, self.wp_id + 1):
            min_cost = np.inf

            self.load_model = idx != self.wp_id
            self.curr_wp = idx - 1

            if idx == self.wp_id and i_episode <= self.exploration_epoch:
                self.best_wp = self.action_dim * (np.random.rand(self.state_dim) - 0.5)
            else:
                for t_idx in range(self.n_inits):
                    xi0 = (np.copy(self.best_traj[self.curr_wp * self.state_dim:self.curr_wp * self.state_dim + self.state_dim]) +
                           np.random.normal(0, 0.1, size=self.state_dim) if t_idx != 0 else
                           np.copy(self.best_traj[self.curr_wp * self.state_dim:self.curr_wp * self.state_dim + self.state_dim]))

                    res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lincon,
                                 options={'eps': 1e-6, 'maxiter': 1e6})
                    if res.fun < min_cost:
                        min_cost = res.fun
                        self.best_wp = res.x
                if idx == self.wp_id and np.random.rand() < 0.5 and i_episode > self.ensemble_sampling_epoch and i_episode < self.averaging_noise_epoch:
                    tqdm.write("NOISE ADDED TO GRIPPER")
                    self.best_wp[-1] *= -1
                if idx == self.wp_id and np.random.rand() < 0.5 and i_episode > self.ensemble_sampling_epoch and i_episode < self.averaging_noise_epoch:
                    tqdm.write("NOISE ADDED TO POSE")
                    self.best_wp[:3] += np.random.normal(0, 0.05, 3)

            self.traj.append(self.best_wp)
        return np.array(self.traj).flatten()

    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)

    def get_cost(self, traj):
        traj_learnt = np.array(self.traj).flatten()
        traj_combined = np.concatenate((traj_learnt, traj))
        traj_combined = np.concatenate((traj_combined, self.objs))
        traj_tensor = tf.convert_to_tensor(traj_combined, dtype=tf.float32)[tf.newaxis, :]
        reward = self.get_reward(traj_tensor)
        return -reward

    def get_reward(self, traj):
        if self.load_model:
            models = self.learned_models[self.curr_wp]
            loss = 0
            for critic in models:
                loss += critic(traj).numpy()[0, 0]
            return loss / self.n_models
        else:
            if self.curr_episode < self.ensemble_sampling_epoch:
                critic = self.models[self.reward_idx]
                return critic(traj).numpy()[0, 0]
            else:
                loss = 0
                for critic in self.models:
                    loss += critic(traj).numpy()[0, 0]
                return loss / self.n_models

    def get_avg_reward(self, traj):
        traj = np.concatenate((traj, self.objs))
        traj_tensor = tf.convert_to_tensor(traj, dtype=tf.float32)[tf.newaxis, :]
        reward = 0
        for critic in self.models:
            reward += critic(traj_tensor).numpy()[0, 0]
        return reward / self.n_models

    def update_parameters(self, memory, batch_size):
        loss = np.zeros(self.n_models)
        for idx, critic in enumerate(self.models):
            loss[idx] = self.update_critic(critic, memory, batch_size)
        return np.mean(loss)

    def update_critic(self, critic, memory, batch_size):
        trajs, rewards = memory.sample(batch_size)
        trajs = tf.convert_to_tensor(trajs, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)[:, tf.newaxis]

        with tf.GradientTape() as tape:
            rhat = critic(trajs)
            loss = tf.reduce_mean(tf.square(rhat - rewards))
        grads = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(grads, critic.trainable_variables))
        return loss.numpy()

    def reset_model(self, idx):
        critic = RNetwork(self.wp_id * self.state_dim + len(self.objs), hidden_dim=self.hidden_size)
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        self.models[idx] = critic
        tqdm.write(f"RESET MODEL {idx}")

    def save_model(self, save_name):
        save_dir = os.path.join('models', save_name)
        os.makedirs(save_dir, exist_ok=True)
        for idx, critic in enumerate(self.models):
            # Build the model with a dummy input before saving
            dummy_input = tf.zeros((1, self.wp_id * self.state_dim + len(self.objs)))
            critic(dummy_input)  # This builds the model
            critic.save_weights(os.path.join(save_dir, f'model_{self.wp_id}_{idx}.weights.h5'))