"""
preprocess.py
"""

import cv2
import numpy as np


def flatten_preprocess(images):
    num_samples = images.shape[0]
    image_size = np.prod(images.shape[1:])
    flattened_images = images.reshape(num_samples, image_size)
    normalized_images = flattened_images / 255.0
    return normalized_images


def sift_preprocess(images):
    sift = cv2.SIFT_create()
    sift_features = []
    for image in images:
        kp, des = sift.detectAndCompute(image, None)
        sift_features.append(des)
    return np.array(sift_features)
