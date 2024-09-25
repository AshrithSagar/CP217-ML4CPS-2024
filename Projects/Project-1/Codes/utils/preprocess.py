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
        if image is None or image.size == 0:
            return np.zeros(128)

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Ensure right depth (CV_8U)
        if gray_image.dtype != np.uint8:
            gray_image = (gray_image * 255).astype(np.uint8)

        _, des = self.sift.detectAndCompute(gray_image, None)
        return np.mean(des, axis=0) if des is not None else np.zeros(128)


class PreprocessorSklearn:
    def __init__(self):
        self.sift_extractor = SIFTExtractor()

    def flatten(self, images):
        normalized_images = images.astype(np.float32) / 255.0
        return normalized_images.reshape(images.shape[0], -1)

    def sift(self, images):
        features = [self.sift_extractor.extract(image) for image in images]
        return np.array(features)


class PreprocessorTF:
    def __init__(self):
        self.sift_extractor = SIFTExtractor()

    def flatten(self, dataset):
        def process_batch(images, labels):
            normalized_images = tf.cast(images, tf.float32) / 255.0
            flattened_images = tf.reshape(normalized_images, (tf.shape(images)[0], -1))
            return flattened_images, labels

        return dataset.map(process_batch)

    def extract_sift_features(self, images):
        return np.array([self.sift_extractor.extract(img) for img in images])

    def sift(self, dataset):
        def extract_sift(images, labels):
            images = tf.image.convert_image_dtype(images, tf.uint8)
            sift_features = tf.numpy_function(
                self.extract_sift_features,
                [images],
                tf.float32,
            )
            return sift_features, labels

        return dataset.map(extract_sift)
