import tensorflow as tf
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

class QNetwork(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.q1 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(num_inputs + num_actions,),
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(1)
        ])
        # Q2 architecture
        self.q2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(num_inputs + num_actions,),
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, state, action):
        xu = tf.concat([state, action], axis=1)
        q1 = self.q1(xu)
        q2 = self.q2(xu)
        return q1, q2

class GaussianPolicy(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.base = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(num_inputs,),
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros')
        ])
        self.mean_linear = tf.keras.layers.Dense(num_actions,
                                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                              bias_initializer='zeros')
        self.log_std_linear = tf.keras.layers.Dense(num_actions,
                                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                 bias_initializer='zeros')

        # Action rescaling
        if action_space is None:
            self.action_scale = tf.constant(1.0, dtype=tf.float32)
            self.action_bias = tf.constant(0.0, dtype=tf.float32)
        else:
            self.action_scale = tf.constant((action_space.high - action_space.low) / 2.0, dtype=tf.float32)
            self.action_bias = tf.constant((action_space.high + action_space.low) / 2.0, dtype=tf.float32)

    def call(self, state):
        x = self.base(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        normal = tf.random.normal(tf.shape(mean))
        x_t = mean + std * normal  # Reparameterization trick
        y_t = tf.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = -0.5 * ((x_t - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        log_prob -= tf.math.log(self.action_scale * (1 - y_t ** 2) + EPSILON)
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        mean = tf.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean