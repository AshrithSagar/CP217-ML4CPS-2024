"""
preprocess.py
"""

import cv2
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.feature import hog


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


class HOGExtractor:
    def __init__(
        self,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
    ):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations

    def extract(self, image):
        if image is None or image.size == 0:
            return np.zeros(324)
        gray_image = rgb2gray(image)
        hog_features = hog(
            gray_image,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            orientations=self.orientations,
            block_norm="L2-Hys",
        )
        return hog_features


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

    def extract_sift_features(self, images):
        return np.array([self.sift_extractor.extract(img) for img in images])

    def sift(self, dataset):
        def extract_sift(images, labels=None):
            images = tf.image.convert_image_dtype(images, tf.uint8)
            sift_features = tf.numpy_function(
                self.extract_sift_features,
                [images],
                tf.float32,
            )
            if labels is None:
                return sift_features
            return sift_features, labels

        return dataset.map(extract_sift)

    def extract_hog_features(self, images):
        return np.array([self.hog_extractor.extract(img) for img in images])

    def hog(self, dataset):
        def extract_hog(images, labels=None):
            images = tf.image.convert_image_dtype(images, tf.uint8)
            hog_features = tf.numpy_function(
                self.extract_hog_features,
                [images],
                tf.float32,
            )
            if labels is None:
                return hog_features
            return hog_features, labels

        return dataset.map(extract_hog)
