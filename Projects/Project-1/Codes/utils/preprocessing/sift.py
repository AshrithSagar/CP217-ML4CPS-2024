import cv2
import numpy as np


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
