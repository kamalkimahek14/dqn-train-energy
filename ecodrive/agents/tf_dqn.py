# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Double DQN in TensorFlow/Keras. Same structure, line for line, as torch_dqn.py."""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..config import DQNConfig


def build_q_net(obs_dim: int, n_actions: int, hidden: tuple) -> keras.Model:
    """MLP: observation -> one Q-value per action."""
    layers = [keras.Input(shape=(obs_dim,))]
    for h in hidden:
        layers.append(keras.layers.Dense(h, activation="relu"))
    layers.append(keras.layers.Dense(n_actions))
    return keras.Sequential(layers)


class TFDQN:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig):
        keras.utils.set_random_seed(cfg.seed)
        self.cfg = cfg
        self.q = build_q_net(obs_dim, n_actions, cfg.hidden)          # online network, trained
        self.q_target = build_q_net(obs_dim, n_actions, cfg.hidden)   # target network, copied
        self.sync_target()
        # global_clipnorm clips the whole gradient vector, like torch's clip_grad_norm_.
        self.opt = keras.optimizers.Adam(learning_rate=cfg.lr, global_clipnorm=cfg.grad_clip)
        self.opt.build(self.q.trainable_variables)  # create Adam's slots outside tf.function
        self.huber = keras.losses.Huber(delta=1.0)

    @tf.function
    def _forward(self, obs):
        return self.q(obs)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self._forward(tf.convert_to_tensor(obs, dtype=tf.float32)).numpy()

    def greedy(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.q_values(obs[None])[0]))

    def td_target(self, rewards, next_obs, dones) -> tf.Tensor:
        """Double DQN: the online net picks the next action, the target net scores it."""
        next_a = tf.argmax(self.q(next_obs), axis=1)
        next_q = tf.gather(self.q_target(next_obs), next_a, batch_dims=1)
        return tf.stop_gradient(rewards + self.cfg.gamma * (1.0 - dones) * next_q)

    def loss(self, batch: dict) -> tf.Tensor:
        b = {k: tf.convert_to_tensor(v) for k, v in batch.items()}
        target = self.td_target(b["rewards"], b["next_obs"], b["dones"])
        q = tf.gather(self.q(b["obs"]), b["actions"], batch_dims=1)  # Q(s, a) of the taken action
        return self.huber(target, q)  # Huber loss with delta = 1

    @tf.function
    def _train_step(self, batch):
        with tf.GradientTape() as tape:
            loss = self.loss(batch)
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss

    def update(self, batch: dict) -> float:
        """One gradient step on a replay batch; returns the loss."""
        return float(self._train_step({k: tf.convert_to_tensor(v) for k, v in batch.items()}))

    def sync_target(self):
        self.q_target.set_weights(self.q.get_weights())

    def save(self, path: str):
        self.q.save_weights(path + ".weights.h5")

    def load(self, path: str):
        self.q.load_weights(path + ".weights.h5")
        self.sync_target()
