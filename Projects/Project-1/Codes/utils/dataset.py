"""
dataset.py
"""

import os

import tensorflow as tf


class GSVDataLoader:
    def __init__(self, train_dir, test_dir, seed=42):
        self.seed = seed
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_dims = (400, 300)
        tf.random.set_seed(seed)
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    def load_train(self, batch_size=32):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_dir,
            label_mode="int",
            image_size=self.image_dims,
            batch_size=batch_size,
        )
