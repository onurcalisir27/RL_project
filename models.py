import tensorflow as tf
import numpy as np

class WeightClipper(tf.keras.callbacks.Callback):
    def __init__(self, frequency=5):
        # self.frequency = frequency
        pass
    
    def on_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                w = layer.kernel.numpy()
                w = np.clip(w, 0, np.inf)
                layer.kernel.assign(w)

def weights_init(shape, dtype=None):
    return tf.keras.initializers.GlorotUniform()(shape)

def bias_init(shape, dtype=None):
    return tf.zeros(shape)

class RNetwork(tf.keras.Model):
    def __init__(self, traj_dim, hidden_dim):
        super(RNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, 
                                           kernel_initializer=weights_init,
                                           bias_initializer=bias_init)
        self.dense2 = tf.keras.layers.Dense(hidden_dim, 
                                           kernel_initializer=weights_init,
                                           bias_initializer=bias_init)
        self.dense3 = tf.keras.layers.Dense(1, 
                                           kernel_initializer=weights_init,
                                           bias_initializer=bias_init)
        self.relu = tf.keras.layers.LeakyReLU()

    def call(self, traj):
        x = self.relu(self.dense1(traj))
        x = self.relu(self.dense2(x))
        return self.dense3(x)


class Actor(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.ReLU()
        ])
    
    def call(self, x):
        return self.model(x)


class GRU(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # TensorFlow handles multi-layer GRUs differently
        self.gru_layers = []
        for i in range(num_layers):
            self.gru_layers.append(tf.keras.layers.GRU(hidden_dim, 
                                                      return_sequences=(i < num_layers-1),
                                                      return_state=True))
        
        self.linear1 = tf.keras.layers.Dense(output_dim)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, h=None):
        # Initial hidden state handling is different in TF
        states = []
        current_input = x
        
        for i, gru_layer in enumerate(self.gru_layers):
            if i == len(self.gru_layers) - 1:
                # For the last layer, get sequences and state
                seq, state = gru_layer(current_input, initial_state=h[i] if h is not None else None)
                current_input = seq
            else:
                # For intermediate layers
                current_input, state = gru_layer(current_input, initial_state=h[i] if h is not None else None)
            states.append(state)
            
        output = self.relu(self.linear1(current_input[:, -1]))
        return output, states

    def init_hidden(self, batch_size):
        return [tf.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]


class AE(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(AE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim

        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(int(self.hidden_dim/2), activation='relu'),
            tf.keras.layers.Dense(self.output_dim)
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(int(self.hidden_dim/2), activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, x):
        wp = self.encoder(x)
        reward = self.decoder(wp)
        return wp, reward