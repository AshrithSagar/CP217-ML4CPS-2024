"""
preprocess.py
"""

import cv2
import numpy as np
import tensorflow as tf


class SIFTExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def extract(self, image):
        _, des = self.sift.detectAndCompute(image, None)
        return np.mean(des, axis=0) if des is not None else np.zeros(128)


class PreprocessorSklearn:
    def flatten(self, images):
        normalized_images = images.astype(np.float32) / 255.0
        return normalized_images.reshape(images.shape[0], -1)

    def sift(self, images):
        self.sift_extractor = SIFTExtractor()
        features = [self.sift_extractor.extract(image) for image in images]
        return np.array(features)


class PreprocessorTF:
    def flatten(self, dataset):
        def process_batch(images, labels):
            normalized_images = tf.cast(images, tf.float32) / 255.0
            flattened_images = tf.reshape(normalized_images, (tf.shape(images)[0], -1))
            return flattened_images, labels

        return dataset.map(process_batch)

    def sift(self, dataset):
        def extract_sift(images, labels):
            sift_features = tf.numpy_function(
                lambda imgs: np.array(
                    [self.sift_extractor.extract(img) for img in imgs]
                ),
                [images],
                tf.float32,
            )
            return sift_features, labels

        self.sift_extractor = SIFTExtractor()
        return dataset.map(extract_sift)
