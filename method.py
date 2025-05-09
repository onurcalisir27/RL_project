import numpy as np
import tensorflow as tf
from scipy.optimize import minimize, LinearConstraint
import os
import random
from models import RNetwork
from tqdm import tqdm

class Method(object):
    def __init__(self, state_dim, objs, wp_id, save_name, config):
        self.state_dim = state_dim
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

        # Initialize models
        for _ in range(self.n_models):
            critic = RNetwork(self.state_dim*(self.wp_id) + len(objs), hidden_dim=self.hidden_size)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.models.append((critic, optimizer))

        save_dir = 'models/' + save_name

        # Load previously trained models
        for wp_id in range(1, self.wp_id):
            models = []
            for idx in range(self.n_models):
                critic = RNetwork(wp_id*self.state_dim + len(objs), hidden_dim=self.hidden_size)
                # Load weights from saved model
                critic.load_weights(save_dir + '/model_' + str(wp_id) + '_' + str(idx) + '.weights.h5')
                models.append(critic)
            self.learned_models.append(models)

        self.best_traj = self.action_dim*(np.random.rand(self.state_dim*self.wp_id) - 0.5)
        self.best_reward = -np.inf
        self.lincon = LinearConstraint(np.eye(self.state_dim), -self.action_dim, self.action_dim)

    def traj_opt(self, i_episode, objs):
        self.reward_idx = random.choice(range(self.n_models))
        self.objs = tf.convert_to_tensor(objs, dtype=tf.float32)
        self.curr_episode = i_episode

        self.traj = []
        for idx in range(1, self.wp_id+1):
            min_cost = np.inf

            self.load_model = True if idx!=self.wp_id else False
            self.curr_wp = idx-1

            if idx == self.wp_id and i_episode <= self.exploration_epoch:
                self.best_wp = self.action_dim*(np.random.rand(self.state_dim) - 0.5)
            else:
                for t_idx in range(self.n_inits):
                    if t_idx != 0:
                        xi0 = np.copy(self.best_traj[self.curr_wp*self.state_dim:self.curr_wp*self.state_dim+self.state_dim]) + np.random.normal(0, 0.1, size=self.state_dim)
                    else:
                        xi0 = np.copy(self.best_traj[self.curr_wp*self.state_dim:self.curr_wp*self.state_dim+self.state_dim])

                    res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lincon, options={'eps': 1e-6, 'maxiter': 1e6})
                    if res.fun < min_cost:
                        min_cost = res.fun
                        self.best_wp = res.x
                
                # Add noise for exploration
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
        traj_tensor = tf.convert_to_tensor(traj, dtype=tf.float32)
        traj_learnt = tf.convert_to_tensor(np.array(self.traj).flatten(), dtype=tf.float32)
        full_traj = tf.concat([traj_learnt, traj_tensor], axis=0)
        full_traj_with_objs = tf.concat([full_traj, self.objs], axis=0)
        reward = self.get_reward(full_traj_with_objs)
        return -reward
    
    def get_reward(self, traj):
        loss = 0
        if self.load_model:
            models = self.learned_models[self.curr_wp]
            for idx in range(self.n_models):
                critic = models[idx]
                loss += critic(tf.expand_dims(traj, 0)).numpy()[0][0]
            return loss/self.n_models
        else:
            if self.curr_episode < self.ensemble_sampling_epoch:
                critic, _ = self.models[self.reward_idx]
                return critic(tf.expand_dims(traj, 0)).numpy()[0][0]
            else:
                for idx in range(self.n_models):
                    critic, _ = self.models[idx]
                    loss += critic(tf.expand_dims(traj, 0)).numpy()[0][0]
                return loss/self.n_models

    def get_avg_reward(self, traj):
        reward = 0
        traj_tensor = tf.convert_to_tensor(traj, dtype=tf.float32)
        traj_with_objs = tf.concat([traj_tensor, self.objs], axis=0)
        for idx in range(self.n_models):
            critic, _ = self.models[idx]
            reward += critic(tf.expand_dims(traj_with_objs, 0)).numpy()[0][0]
        return reward/self.n_models

    def update_parameters(self, memory, batch_size):
        loss_values = np.zeros((self.n_models))
        for idx, (critic, optim) in enumerate(self.models):
            loss_values[idx] = self.update_critic(critic, optim, memory, batch_size)
        return np.mean(loss_values)

    def update_critic(self, critic, optimizer, memory, batch_size):
        trajs, rewards = memory.sample(batch_size)
        trajs_tensor = tf.convert_to_tensor(trajs, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = critic(trajs_tensor)
            loss = tf.reduce_mean(tf.square(predictions - tf.expand_dims(rewards_tensor, -1)))
        
        gradients = tape.gradient(loss, critic.trainable_variables)
        optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
        
        return loss.numpy()

    def reset_model(self, idx):
        critic = RNetwork(self.wp_id*self.state_dim + len(self.objs), hidden_dim=self.hidden_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.models[idx] = (critic, optimizer)
        tqdm.write("RESET MODEL {}".format(idx))

    def save_model(self, save_name):
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx in range(self.n_models):
            critic, _ = self.models[idx]  # Unpack the tuple to get just the critic
            # Build the model with a dummy input to ensure it's built
            dummy_input = tf.zeros((1, self.wp_id*self.state_dim + len(self.objs)))
            critic(dummy_input)  # This will build the model
            critic.save_weights(save_dir + '/model_' + str(self.wp_id) + '_' + str(idx) + '.weights.h5')