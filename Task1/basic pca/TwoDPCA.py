import numpy as np 
import cv2
import scipy.linalg as s_linalg
class two_d_pca_class:

    def __init__(self, images, y, target_names):
        """
        Constructor for the 2D PCA class.

        Parameters:
        - images (numpy.ndarray): Input data matrix where each row represents an image.
        - y (numpy.ndarray): Class labels for each image.
        - target_names (list): Names corresponding to different classes.
        """
        self.images = np.asarray(images)
        self.y = y
        self.target_names = target_names

        # Finding means of image
        self.mean_face = np.mean(self.images, axis = 0)

        # Subtracting mean face from images
        self.images_mean_subtracted = self.images - self.mean_face

    # Function used for finding value of p for convering 95% of image information
    def give_p(self, d):
        """
        Calculate the number of principal components to retain to cover 95% of image information.

        Parameters:
        - d (numpy.ndarray): Array of eigenvalues.

        Returns:
        - p (int): Number of principal components to retain.
        """
        # Sum of all eigen values
        sum = np.sum(d)
        sum_95 = 0.95*sum
        temp = 0
        p = 0
        # Accumulate eigenvalues until the sum reaches 95%
        while temp < sum_95:
            temp += d[p]
            p += 1
        return p
    
    def reduce_dim(self):
        """
        Reduce the dimensionality of the input data using 2D Principal Component Analysis (PCA).

        Returns:
        - numpy.ndarray: Transformed data in the reduced dimension space.
        """
        no_of_images = self.images.shape[0]
        mat_height = self.images.shape[1]

        # Creating empty matrix for find covarience matrix
        g_t = np.zeros((mat_height, mat_height))

        for i in range(no_of_images):
            # Muliplying net subtracted image with its transpose and adding in gt
            temp = np.dot(self.images_mean_subtracted[i].T, self.images_mean_subtracted[i])
            g_t += temp

        # Dividing by the total number of images
        g_t /= no_of_images

        # Finding eigenvalues and eigenvectors
        d_mat, p_mat = np.linalg.eig(g_t)

        # Finding the first p important vectors
        p = self.give_p(d_mat)
        self.new_bases = p_mat[:, 0:p]

        # Finding new coordinates using dot product new bases
        self.new_coordinates = np.dot(self.images, self.new_bases)

        # Returning new coordinates matrix
        return self.new_coordinates
    
    def original_data(self, new_coordinates):
        """
        Reconstruct the original data from the reduced dimension space.

        Parameters:
        - new_coordinates (numpy.ndarray): Data in the reduced dimension space.

        Returns:
        - numpy.ndarray: Reconstructed data in the original dimension space.
        """
        return np.dot(new_coordinates, self.new_bases.T)
    
    def new_cord(self, name, img_height, img_width):
        """
        Calculate the reduced coordinates for a new image.

        Parameters:
        - name (str): File name of the new image.
        - img_height (int): Height of the new image.
        - img_width (int): Width of the new image.

        Returns:
        - numpy.ndarray: Reduced coordinates for the new image.
        """
        img = cv2.imread(name, 0)
        cv2.imshow("Recognize Image", img)
        cv2.waitKey(0)
        gray = cv2.resize(img, (img_height, img_width))
        return np.dot(gray, self.new_bases)
    
    def new_cord_for_image(self, image):
        """
        Calculate the reduced coordinates for a given image.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - numpy.ndarray: Reduced coordinates for the input image.
        """
        return np.dot(image, self.new_bases)
    
    def recognize_face(self, new_cord):
        """
        Recognize the face based on reduced coordinates.

        Parameters:
        - new_cord (numpy.ndarray): Reduced coordinates for the face to be recognized.

        Returns:
        - str: Name of the recognized person.
        """
        no_of_images = len(self.y)
        distances = []
        for i in range(no_of_images):
            temp_imgs = self.new_coordinates[i]
            dist = np.linalg.norm(new_cord - temp_imgs)
            distances += [dist]

        print("Distances", distances)
        min = np.argmin(distances)
        per = self.y[min]
        per_name = self.target_names[per]

        print("Person ", per, ": ", min, self.target_names[per], "Dist: ", distances[min])
        return per_name

            
    