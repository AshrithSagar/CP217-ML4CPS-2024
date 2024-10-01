import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog


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
