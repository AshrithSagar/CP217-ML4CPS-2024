"""
preprocess.py
"""

import cv2
import numpy as np
import tensorflow as tf


def extract_sift_features(image):
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(image, None)
    return np.mean(des, axis=0) if des is not None else np.zeros(128)


def sift_preprocess_sklearn(images):
    features = [extract_sift_features(image) for image in images]
    return np.array(features)


def flatten_preprocess_sklearn(images):
    normalized_images = images.astype(np.float32) / 255.0
    return normalized_images.reshape(images.shape[0], -1)


def sift_preprocess_tf(images):
    sift_features = tf.map_fn(
        lambda x: tf.numpy_function(extract_sift_features, [x], tf.float32), images
    )
    return sift_features


def flatten_preprocess_tf(images):
    normalized_images = tf.cast(images, tf.float32) / 255.0
    return tf.reshape(normalized_images, (tf.shape(images)[0], -1))
