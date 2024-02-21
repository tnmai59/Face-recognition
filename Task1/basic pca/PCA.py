import numpy as np 
import cv2
import scipy.linalg as s_linalg
class pca_class:

    def __init__(self, images, y, target_names, no_of_elements, quality_percent):       
        """
        Constructor for the PCA class.

        Parameters:
        - images (numpy.ndarray): Input data matrix where each column represents an image.
        - y (numpy.ndarray): Class labels for each image.
        - target_names (list): Names corresponding to different classes.
        - no_of_elements (list): Number of elements per class.
        - quality_percent (float): Percentage of total energy to retain during dimensionality reduction.
        """
        self.no_of_elements = no_of_elements        
        self.images = np.asarray(images)        
        self.y = y      
        self.target_names = target_names        
        mean = np.mean(self.images, 1)      
        self.mean_face = np.asmatrix(mean).T        
        self.images = self.images - self.mean_face      
        self.quality_percent = quality_percent     

    def give_p(self, d): 
        """
        Calculate the number of principal components to retain based on a quality percentage.

        Parameters:
        - d (numpy.ndarray): Array of singular values obtained from SVD.

        Returns:
        - p (int): Number of principal components to retain.
        """
        sum = np.sum(d)         
        sum_85 = self.quality_percent * sum/100         
        temp = 0
        p = 0
        # Accumulate singular values until the sum reaches the specified threshold
        while temp < sum_85:                
            temp += d[p]
            p += 1
        return p 
    
    def reduce_dim(self):
        """
        Reduce the dimensionality of the input data using PCA.

        Returns:
        - self.new_coordinates.T (numpy.ndarray): Transformed data in the reduced dimension space.
        """
        # Perform SVD on the input data
        p, d, q = s_linalg.svd(self.images, full_matrices = True) 

        # Convert the matrices to numpy matrices for further processing
        p_matrix = np.matrix(p) 
        d_diag = np.diag(d) 
        q_matrix = np.matrix(q) 

        # Determine the number of principal components to retain based on quality_percent
        p = self.give_p(d) 

        # Extract the first p_to_retain columns from the left singular vectors matrix (U)
        self.new_bases = p_matrix[:, 0:p] 

        # Project the original data onto the selected principal components
        self.new_coordinates = np.dot(self.new_bases.T, self.images) 

        return self.new_coordinates.T

    def original_data(self, new_coordinates):  
        """
        Reconstruct the original data from the reduced dimension space.

        Parameters:
        - new_coordinates (numpy.ndarray): Data in the reduced dimension space.

        Returns:
        - numpy.ndarray: Reconstructed data in the original dimension space.
        """     
        return self.mean_face + np.dot(self.new_bases, new_coordinates.T)  

    def show_eigen_face(self, height, width, min_pix_int, max_pix_int, eig_no): 
        """
        Display an eigenface.

        Parameters:
        - height (int): Height of the image.
        - width (int): Width of the image.
        - min_pix_int (int): Minimum pixel intensity for display.
        - max_pix_int (int): Maximum pixel intensity for display.
        - eig_no (int): Index of the eigenface to display.
        """    
        ev = self.new_bases[:, eig_no: eig_no + 1]      
        # min_orig = np.min(ev)   
        # max_orig = np.max(ev)   
        # ev = min_pix_int + (((max_pix_int - min_pix_int)/(max_orig - min_orig) * (ev - min_orig)) * ev)
        min_orig = np.min(ev)
        max_orig = np.max(ev)
        ev = min_pix_int + (((max_pix_int - min_pix_int)/(max_orig - min_orig)) * (ev - min_orig))  
        ev_re = np.reshape(ev, (height, width))  
        cv2.imshow("Eigen Face" + str(eig_no), cv2.resize(np.array(ev_re, dtype=np.uint8), (200,200)))
        cv2.waitKey(0)

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
        img = cv2.imread(name)
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img_height, img_width))
        img_vec = np.asmatrix(gray).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.y)) + img_vec)/(len(self.y) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    def new_cord_for_images(self, image):
        """
        Calculate the reduced coordinates for a given image.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - numpy.ndarray: Reduced coordinates for the input image.
        """
        img_vec = np.asmatrix(image).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.y)) + img_vec)/(len(self.y) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    
    def recognize_face(self, new_cord_pca, k=0):
        """
        Recognize the face based on reduced coordinates.

        Parameters:
        - new_cord_pca (numpy.ndarray): Reduced coordinates for the face to be recognized.
        - k (int): Index of the face to be recognized.

        Returns:
        - str: Name of the recognized person or 'Unknown'.
        """
        classes = len(self.no_of_elements)
        start = 0
        distances = []
        for i in range(classes):
            temp_imgs = self.new_coordinates[:, int(start): int(start + self.no_of_elements[i])]
            mean_temp = np.mean(temp_imgs, 1)
            start = start + self.no_of_elements[i]
            dist = np.linalg.norm(new_cord_pca - mean_temp)
            distances += [dist]
        min = np.argmin(distances)

        # Temp Threshold
        threshold = 100000
        if distances[min] < threshold:
            print("Person ", k, ": ", min, self.target_names[min])
            return self.target_names[min]
        else:
            print("Person ", k, ": ", min, "Unknown")
            return "Unknown"