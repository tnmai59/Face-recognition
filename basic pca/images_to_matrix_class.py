import cv2
import numpy as np

class images_to_matrix_class:
    """
    This class is used to convert a list of image names into the corresponding matrix representation.
    It builds a complete matrix with each column representing an image converted into a column vector.
    """

    def __init__(self, images_name, img_width, img_height):
        """
        Initializes the images_to_matrix_class object.

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
        Converts the images to a matrix representation.

        Returns:
            img_mat (numpy.ndarray): Matrix representation of the images.
        """

        col = len(self.images_name)  # Save the number of columns in the resulting matrix corresponding to the number of images
        img_mat = np.zeros((self.img_size, col))  # Create a matrix with size (img_size, col)

        i = 0
        for name in self.images_name:
            gray = cv2.imread(name, 0)  # Read the image from the file with the name 'name' as grayscale and store it in 'gray'
            gray = cv2.resize(gray, (self.img_height, self.img_width))  # Resize the image 'gray' to size (img_height, img_width)
            mat = np.asmatrix(gray)  # Convert the 'gray' image to a matrix 'mat'
            img_mat[:, i] = mat.ravel()  # Assign the flattened matrix 'mat' as a column vector to column 'i' of the matrix 'img_mat'
            i += 1

        return img_mat