import cv2
import numpy as np


class images_to_matrix_class_for_two_d:
    """
    This class is used to convert a list of image names into a 2D matrix representation.
    It builds a matrix with each element representing a pixel value of an image.
    """

    def __init__(self, images_name, img_width, img_height):
        """
        Initializes the images_to_matrix_class_for_two_d object.

        Args:
            images_name (list): List of image names.
            img_width (int): Width of the images.
            img_height (int): Height of the images.
        """

        self.images_name = images_name
        self.img_width = img_width
        self.img_height = img_height
        self.img_size = img_width * img_height

    def get_matrix(self):
        """
        Converts the images to a 2D matrix representation.

        Returns:
            img_mat (numpy.ndarray): 2D matrix representation of the images.
        """

        img_mat = np.zeros(
            (len(self.images_name), self.img_height, self.img_width),
            dtype=np.uint8)

        i = 0
        for name in self.images_name:
            gray = cv2.imread(name, 0)  # Read the image from the file with the name 'name' as grayscale and store it in 'gray'
            gray = cv2.resize(gray, (self.img_height, self.img_width))  # Resize the image 'gray' to size (img_height, img_width)
            mat = np.asmatrix(gray)  # Convert the 'gray' image to a matrix 'mat'
            img_mat[i, :, :] = mat  # Assign the matrix 'mat' to the i-th row of the 2D matrix 'img_mat'
            i += 1

        print("Matrix Size:", img_mat.shape)
        return img_mat