"""
preprocess.py
"""

from functools import partial

import numpy as np
import tensorflow as tf
from utils.preprocessing.hog import HOGExtractor
from utils.preprocessing.sift import SIFTExtractor


class PreprocessorSklearn:
    def __init__(self):
        self.sift_extractor = SIFTExtractor()
        self.hog_extractor = HOGExtractor()

    def flatten(self, images):
        normalized_images = images.astype(np.float32) / 255.0
        return normalized_images.reshape(images.shape[0], -1)

    def sift(self, images):
        features = []
        for image in images:
            if image is None or image.size == 0:
                features.append(np.zeros(128))
            elif len(image.shape) != 3 or image.shape[2] != 3:
                print(f"Unexpected image shape: {image.shape}")
                features.append(np.zeros(128))
            else:
                features.append(self.sift_extractor.extract(image))
        return np.array(features)

    def hog(self, images):
        features = []
        for image in images:
            if image is None or image.size == 0:
                features.append(np.zeros(324))
            elif len(image.shape) != 3 or image.shape[2] != 3:
                print(f"Unexpected image shape: {image.shape}")
                features.append(np.zeros(324))
            else:
                features.append(self.hog_extractor.extract(image))
        return np.array(features)


class PreprocessorTF:
    def __init__(self):
        self.sift_extractor = SIFTExtractor()
        self.hog_extractor = HOGExtractor()

    def flatten(self, dataset):
        def process_batch(images, labels):
            normalized_images = tf.cast(images, tf.float32) / 255.0
            flattened_images = tf.reshape(normalized_images, (tf.shape(images)[0], -1))
            return flattened_images, labels

        return dataset.map(process_batch)

    def extract_features(self, extractor, images):
        return np.array([extractor.extract(img) for img in images])

    def sift(self, dataset):
        extractor = partial(self.extract_features, self.sift_extractor)

        def extract_sift(images, labels=None):
            images = tf.image.convert_image_dtype(images, tf.uint8)
            sift_features = tf.numpy_function(
                extractor,
                [images],
                tf.float32,
            )
            if labels is None:
                return sift_features
            return sift_features, labels

        return dataset.map(extract_sift)

    def hog(self, dataset):
        extractor = partial(self.extract_features, self.hog_extractor)

        def extract_hog(images, labels=None):
            images = tf.image.convert_image_dtype(images, tf.uint8)
            hog_features = tf.numpy_function(
                extractor,
                [images],
                tf.float32,
            )
            if labels is None:
                return hog_features
            return hog_features, labels

        return dataset.map(extract_hog)
