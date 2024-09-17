"""
dataset.py
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image


class GSVDataLoader:
    def __init__(self, train_dir, test_dir, seed=42):
        self.seed = seed
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_dims = (400, 300)
        tf.random.set_seed(seed)
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    def load_train_tf(self, batch_size=32):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_dir,
            label_mode="int",
            image_size=self.image_dims,
            batch_size=batch_size,
        )

    def load_train(self):
        train_ds = []
        labels = []
        label_map = {"A": 1, "B": 2, "C": 3, "D": 4, "S": 5}
        for label in os.listdir(self.train_dir):
            if not os.path.isdir(os.path.join(self.train_dir, label)):
                continue
            for img in os.listdir(os.path.join(self.train_dir, label)):
                img_path = os.path.join(self.train_dir, label, img)
                if not img_path.endswith(".jpg"):
                    continue
                image = Image.open(img_path)
                image = np.array(image)
                train_ds.append(image)
                labels.append(label_map[label])
        self.train_ds = np.array(train_ds)
        self.labels = np.array(labels)
