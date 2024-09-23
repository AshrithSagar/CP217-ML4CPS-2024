"""
preprocess.py
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, images):
        self.scaler.fit(images)

    def transform(self, images):
        return self.scaler.transform(images)

    def fit_transform(self, images):
        return self.scaler.fit_transform(images)
