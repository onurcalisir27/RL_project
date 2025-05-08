import tensorflow as tf
import numpy as np

class RNetwork(tf.keras.Model):
    def __init__(self, traj_dim, hidden_dim):
        super(RNetwork, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(traj_dim,), activation='leaky_relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_dim, activation='leaky_relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, traj):
        return self.model(traj)

class WeightClipper(tf.keras.callbacks.Callback):
    def on_batch_end(self):
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.kernel
                weights.assign(tf.clip_by_value(weights, 0, float('inf')))