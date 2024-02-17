import os

class dataset_class:
    """
    This class is used for processing an image dataset.
    """
    
    def __init__(self, required_no):
        """
        Initializes the dataset_class object.
        
        Args:
            required_no (int): The number of images required for each label.
        """
        
        # Dataset Name
        self.dir = "D:\Face recognition\pics"

        self.images_name_for_train = []  # List to store the names of images used for training
        self.target_name_as_array = []  # Array to store the target labels corresponding to the training images
        self.target_name_as_set = {}  # Dictionary to map label indices to label names
        self.y_for_train = []  # List to store the target labels corresponding to the training images
        self.no_of_elements_for_train = []  # List to store the count of processed images for each label in the training set
        
        self.images_name_for_test = []  # List to store the names of images used for testing
        self.y_for_test = []  # List to store the target labels corresponding to the testing images
        self.no_of_elements_for_test = []  # List to store the count of processed images for each label in the testing set
        
        per_no = 0
        
        # Loop through all files and directories in self.dir
        for name in os.listdir(self.dir):
            dir_path = os.path.join(self.dir, name)
            
            if os.path.isdir(dir_path):
                if len(os.listdir(dir_path)) >= required_no:
                    # Ensure that only directories with the required number of selected images will be used
                    
                    i = 0
                    # Count the number of processed images in the current directory
                    for img_name in os.listdir(dir_path):
                        img_path = os.path.join(dir_path, img_name)
                        
                        if i < required_no:
                            # Check if the number of selected images for person i is less than the required number of images required_no
                            self.images_name_for_train += [img_path]
                            self.y_for_train += [per_no]
                            
                            # Check if the number of images in the no_of_elements_for_train list is greater than the person's serial number (per_no)
                            if len(self.no_of_elements_for_train) > per_no:
                                # If yes, increase the number of images corresponding to the person's serial number in the list
                                # This means that at least one other person has been processed before
                                self.no_of_elements_for_train[per_no] += 1
                            else:
                                # If not, it means no one else has been processed before, so a new image will be added to the list with the initial value of 1
                                # Process the first image of a person
                                self.no_of_elements_for_train += [1]
                            
                            if i == 0:
                                # Check if this is the first image in a person's directory
                                self.target_name_as_array += [name]
                                # Add the person's name to the list (store the names of all people in their appearance order in the dataset)
                                self.target_name_as_set[per_no] = name
                                # Assign the person's name to per_no (person's serial number)
                            
                        else:
                            # If the number of processed images for person i is greater than or equal to required_no, add that image to the list and corresponding dictionary for the testing process
                            self.images_name_for_test += [img_path]
                            self.y_for_test += [per_no]
                            
                            # Check if the number of images in the no_of_elements_for_test list is greater than the person's serial number
                            if len(self.no_of_elements_for_test) > per_no:
                                # If yes, it means there is at least one other image of this person in the test set
                                self.no_of_elements_for_test[per_no] += 1
                            else:
                                # If not, it means there is no other image of this person in the test set, so a new image will be added to the list with the initial value of 1
                                # Process the first image of a person in the test set
                                self.no_of_elements_for_test += [1]
                        
                        i += 1
                    
                    per_no += 1
 