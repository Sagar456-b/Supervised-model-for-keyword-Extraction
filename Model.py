# Import TensorFlow for building and running neural networks
import tensorflow as tf

# Import Keras classes and functions for model construction
from keras.optimizers.legacy import Adam
from keras.layers import Layer
from keras import Input
from keras.models import Model, Sequential
from keras.layers import TimeDistributed, Dense, Embedding, Dropout, Bidirectional, MultiHeadAttention

# Custom GRU cell layer to handle gated recurrent operations
class CustomGRUCell(Layer):
    def __init__(self, units, **kwargs):
        super(CustomGRUCell, self).__init__(**kwargs)
        self.units = units  # Number of units in the GRU cell

    def build(self, input_shape):
        # Create weights for update, reset, and candidate operations within the GRU
        self.Wz = self.add_weight(shape=(input_shape[-1] + self.units, self.units), initializer="glorot_uniform", name="Wz")
        self.Wr = self.add_weight(shape=(input_shape[-1] + self.units, self.units), initializer="glorot_uniform", name="Wr")
        self.Wh = self.add_weight(shape=(input_shape[-1] + self.units, self.units), initializer="glorot_uniform", name="Wh")

    def call(self, inputs, states):
        # GRU logic to compute the next state
        h_prev = states[0]
        x_h = tf.concat([inputs, h_prev], axis=-1)
        z = tf.sigmoid(x_h @ self.Wz)
        r = tf.sigmoid(x_h @ self.Wr)
        h_hat = tf.tanh(tf.concat([inputs, r * h_prev], axis=-1) @ self.Wh)
        new_h = z * h_prev + (1 - z) * h_hat
        return new_h, [new_h]

# Custom layer for processing sequences with the custom GRU cell
class CustomGRULayer(Layer):
    def __init__(self, units, return_sequences=True, return_state=False, go_backwards=False, **kwargs):
        super(CustomGRULayer, self).__init__(**kwargs)
        self.units = units  # Number of units in each GRU cell
        self.return_sequences = return_sequences  # Whether to return all outputs or just the last one
        self.return_state = return_state  # Whether to return the last state
        self.go_backwards = go_backwards  # Process input sequences in reverse order
        self.cell = CustomGRUCell(units)

    def call(self, inputs):
        # Processing logic for the GRU layer, with options to reverse sequences and manage states
        if self.go_backwards:
            inputs = tf.reverse(inputs, axis=[1])
        batch_size = tf.shape(inputs)[0]
        initial_state = tf.zeros((batch_size, self.units))
        states = [initial_state]
        outputs = tf.TensorArray(dtype=tf.float32, size=tf.shape(inputs)[1], dynamic_size=True)
        time = tf.constant(0)
        def condition(time, outputs, states):
            return time < tf.shape(inputs)[1]
        def body(time, outputs, states):
            x_t = inputs[:, time, :]
            output, states = self.cell(x_t, states)
            outputs = outputs.write(time, output)
            return time + 1, outputs, states
        _, outputs, last_state = tf.while_loop(condition, body, [time, outputs, states])
        outputs = outputs.stack()
        if self.return_sequences:
            outputs = tf.transpose(outputs, [1, 0, 2])
        else:
            outputs = outputs[-1]
        if self.return_state:
            return outputs, last_state[0]
        return outputs

    def get_config(self):
        # Serialization support for the custom GRU layer
        config = super(CustomGRULayer, self).get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards
        })
        return config

# Class for building a keyword extraction model with custom GRU layers and multi-head attention
class KeywordExtractionModel:
    def __init__(self, hidden_unit, word_index, embedding_dim, embedding_matrix, learning_rate):
        self.hidden_unit = hidden_unit  # Number of units in each GRU cell
        self.word_index = word_index  # Vocabulary index map
        self.embedding_matrix = embedding_matrix  # Pre-trained word embeddings
        self.learning_rate = learning_rate  # Learning rate for the optimizer
        self.vocab_size = len(self.word_index) + 1  # Vocabulary size
        self.embedding_dim = embedding_dim  # Dimensionality of word embeddings
        self.model = self.build_model()  # Construct the model

    def build_model(self):
        # Construct a model with embedding, custom GRU, and multi-head attention layers
        input_seq = Input(shape=(None,))
        # This layer transforms the integer-encoded vocabulary into dense vector embeddings.
        x = Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix], trainable=False)(input_seq)
        # A bidirectional GRU (Gated Recurrent Unit) processes the data in both forward and backward directions. This can help the model to capture dependencies from both past and future and is particularly useful in sequence-to-sequence tasks.
        x = Bidirectional(CustomGRULayer(self.hidden_unit, return_sequences=True))(x)
        # This layer randomly sets a fraction (30% here) of input units to 0 at each update during training time, which helps to prevent overfitting.
        x = Dropout(0.3)(x)
        # Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With num_heads=2, the attention mechanism is performed in parallel, allowing the model to learn more complex dependencies.
        #  The attention layer processes queries, keys, and values, all set to x here, implying self-attention.
        attention_output = MultiHeadAttention(num_heads=2, key_dim=self.hidden_unit)(x, x, x)
        # Further dropout for regularization followed by another bidirectional GRU to process the sequence again, which might enhance the model’s ability to understand more complex patterns or longer dependencies in the data.
        x = Dropout(0.5)(attention_output)
        x = Bidirectional(CustomGRULayer(self.hidden_unit, return_sequences=True))(x)
        x = Dropout(0.5)(x)
        # This applies a dense layer to every temporal slice of the input. Each dense layer uses the softmax activation function to output probabilities for two classes for each time step.
        # that squesh the value
        output = TimeDistributed(Dense(2, activation="softmax"))(x)
        # The model uses the Adam optimizer with a specified learning rate. It is compiled with categorical crossentropy as the loss function, which is typical for multi-class classification problems, and accuracy as the metric for performance evaluation
        optimizer = Adam(learning_rate=self.learning_rate)
        model = Model(inputs=input_seq, outputs=output)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)  # Fit the model with given data

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)  # Evaluate the model

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)  # Predict using the model

    def to_json(self):
        return self.model.to_json()  # Serialize model to JSON

    def save_weights(self, filepath, *args, **kwargs):
        self.model.save_weights(filepath, *args, **kwargs)  # Save model weights
