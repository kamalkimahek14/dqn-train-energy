# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Slack predictor in TensorFlow/Keras: a feed-forward net over the flattened history window. Same
structure, line for line, as torch_slack.py."""
import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_net(n_in: int, hidden: tuple) -> keras.Model:
    """MLP: inputs -> one logit for 'a slack event happens in the next decision'."""
    layers = [keras.Input(shape=(n_in,))]
    for h in hidden:
        layers.append(keras.layers.Dense(h, activation="relu"))
    layers.append(keras.layers.Dense(1))
    return keras.Sequential(layers)


class TFSlackNet:
    def __init__(self, n_in: int, hidden: tuple, seed: int, lr: float):
        keras.utils.set_random_seed(seed)
        self.net = build_net(n_in, hidden)
        self.opt = keras.optimizers.Adam(learning_rate=lr)
        self.opt.build(self.net.trainable_variables)

    @tf.function
    def _forward(self, x):
        return tf.sigmoid(self.net(x))[:, 0]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._forward(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()

    def loss(self, x, y, pos_weight: float) -> tf.Tensor:
        logits = self.net(tf.convert_to_tensor(x, dtype=tf.float32))[:, 0]
        # Weighted cross-entropy: each positive counts pos_weight times, against the class imbalance.
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            tf.convert_to_tensor(y, dtype=tf.float32), logits, pos_weight))

    def eval_loss(self, x, y, pos_weight: float) -> float:
        return float(self.loss(x, y, pos_weight))

    @tf.function
    def _train_step(self, x, y, pos_weight):
        with tf.GradientTape() as tape:
            loss = self.loss(x, y, pos_weight)
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))
        return loss

    def update(self, x, y, pos_weight: float) -> float:
        return float(self._train_step(tf.convert_to_tensor(x, dtype=tf.float32),
                                      tf.convert_to_tensor(y, dtype=tf.float32), tf.constant(pos_weight)))

    def get_weights(self) -> list:
        return [w.copy() for w in self.net.get_weights()]

    def set_weights(self, weights: list):
        self.net.set_weights(weights)

    def save(self, path: str):
        self.net.save_weights(path + ".weights.h5")

    def load(self, path: str):
        self.net.load_weights(path + ".weights.h5")
