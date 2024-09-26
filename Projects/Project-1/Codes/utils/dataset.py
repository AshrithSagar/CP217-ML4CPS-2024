"""
dataset.py
"""

import abc
import os

import numpy as np
import tensorflow as tf
from PIL import Image


class GSVDataLoaderBase(abc.ABC):
    def __init__(self, train_dir, test_dir, seed=42):
        self.seed = seed
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_dims = (400, 300)
        self.train_ds = None
        self.labels = None
        self.test_ds = None

    @abc.abstractmethod
    def load_train(self):
        pass

    @abc.abstractmethod
    def load_test(self):
        pass

    @abc.abstractmethod
    def select_subset(self, num_samples):
        pass


class GSVDataLoaderSklearn(GSVDataLoaderBase):
    def __init__(self, train_dir, test_dir, seed=42):
        super().__init__(train_dir, test_dir, seed)

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
                if image.mode != "RGB":
                    image = image.convert("RGB")  # Convert to RGB
                image = np.array(image)
                train_ds.append(image)
                labels.append(label_map[label])
        self.train_ds = np.array(train_ds)
        self.labels = np.array(labels)

    def load_test(self):
        test_ds = []
        for img in os.listdir(self.test_dir):
            img_path = os.path.join(self.test_dir, img)
            if not img_path.endswith(".jpg"):
                continue
            image = Image.open(img_path)
            image = np.array(image)
            test_ds.append(image)
        self.test_ds = np.array(test_ds)

    def select_subset(self, num_samples):
        indices = np.random.choice(self.train_ds.shape[0], num_samples, replace=False)
        self.train_ds = self.train_ds[indices]
        self.labels = self.labels[indices]


class GSVDataLoaderTF(GSVDataLoaderBase):
    def __init__(self, train_dir, test_dir, seed=42):
        super().__init__(train_dir, test_dir, seed)
        self.autotune = tf.data.experimental.AUTOTUNE

    def load_train(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_dir,
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=32,
            image_size=self.image_dims,
            seed=self.seed,
        )
        train_ds = train_ds.map(self.normalize_images)
        self.train_ds = train_ds.prefetch(self.autotune)

    def load_test(self, batch_size=32):
        test_ds = tf.data.Dataset.list_files(self.test_dir + "/*")
        test_ds = test_ds.map(self.process_image, num_parallel_calls=self.autotune)
        test_ds = test_ds.filter(lambda x: x is not None)
        self.test_ds = test_ds.batch(batch_size).prefetch(self.autotune)

    def select_subset(self, num_samples):
        self.train_ds = self.train_ds.shuffle(buffer_size=1000, seed=self.seed)
        self.train_ds = self.train_ds.take(num_samples)

    def process_image(self, file_path):
        try:
            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_dims)
            image /= 255.0  # Normalize
            return image
        except tf.errors.InvalidArgumentError:
            return None

    def normalize_images(self, x, y):
        return x / 255.0, y
