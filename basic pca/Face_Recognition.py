# Importing necessary libraries
import cv2
import time
import numpy as np

# Importing custom algorithms and feature extraction classes
from PCA import pca_class
from TwoDPCA import two_d_pca_class
from TwoD_Square_PCA import two_d_square_pca_class
from images_to_matrix_class import images_to_matrix_class
from images_matrix_for_2d_square_pca import  images_to_matrix_class_for_two_d
from dataset import dataset_class

# Algorithm type (pca, 2d-pca, 2d2-pca)
algo_type = "pca"


# Recognition type (0 for single image, 1 for video, 2 for group image)
reco_type = 0

# Number of images for training (Left will be used as testing image)
no_of_images_of_one_person = 8
dataset_obj = dataset_class(no_of_images_of_one_person) 


# Data for training
images_names = dataset_obj.images_name_for_train
y = dataset_obj.y_for_train 
no_of_elements = dataset_obj.no_of_elements_for_train 
target_names = dataset_obj.target_name_as_array 

# Data for testing
images_names_for_test = dataset_obj.images_name_for_test 
y_for_test = dataset_obj.y_for_test 
no_of_elements_for_test = dataset_obj.no_of_elements_for_test

# Timing the training process
training_start_time = time.process_time()
img_width, img_height = 50, 50
# Creating an instance of the images_to_matrix_class for feature extraction
if algo_type == "pca":
    i_t_m_c = images_to_matrix_class(images_names, img_width, img_height)
else:
    i_t_m_c = images_to_matrix_class_for_two_d(images_names, img_width, img_height)

# Extracting features from images
scaled_face = i_t_m_c.get_matrix()

# Displaying the original image
if algo_type == "pca":
    cv2.imshow("Original Image" , cv2.resize(np.array(np.reshape(scaled_face[:,1],[img_height, img_width]), dtype = np.uint8),(200, 200)))
    cv2.waitKey()
else:
    cv2.imshow("Original Image" , cv2.resize(scaled_face[0],(200, 200)))
    cv2.waitKey()

# Selecting the algorithm based on algo_type
if algo_type == "pca":
    my_algo = pca_class(scaled_face, y, target_names, no_of_elements, 90)
elif algo_type == "2d-pca":
    my_algo = two_d_pca_class(scaled_face, y, target_names)
else:
    my_algo = two_d_square_pca_class(scaled_face, y, target_names)

# Reducing dimensions using the selected algorithm
new_coordinates = my_algo.reduce_dim()

# Displaying eigenfaces for PCA algorithm
if algo_type == "pca":
    my_algo.show_eigen_face(img_width, img_height, 50, 150, 0)

# Displaying the reconstructed image after PCA
if algo_type == "pca":
    cv2.imshow("After PCA Image", cv2.resize(np.array(np.reshape(my_algo.original_data(new_coordinates[1, :]), [img_height, img_width]), dtype = np.uint8), (200, 200)))
    cv2.waitKey()
else:
    cv2.imshow("After PCA Image", cv2.resize(np.array(my_algo.original_data(new_coordinates[0]), dtype = np.uint8), (200, 200)))
    cv2.waitKey()

# Calculating training time
training_time = time.process_time() - training_start_time


# Face recognition for single image
if reco_type == 0:
    time_start = time.process_time()

    correct = 0
    wrong = 0
    i = 0
    net_time_of_reco = 0

    for img_path in images_names_for_test:

        time_start = time.process_time()
        find_name = my_algo.recognize_face(my_algo.new_cord(img_path, img_height, img_width))
        time_elapsed = (time.process_time() - time_start)
        net_time_of_reco += time_elapsed
        rec_y = y_for_test[i]
        rec_name = target_names[rec_y]
        if find_name is rec_name:
            correct += 1
            print("Correct", " Name:", find_name)
        else:
            wrong +=1
            print("Wrong:", " Real Name:", rec_name, "Rec Y:", rec_y, "Find Name:", find_name)
        i+=1

    # Displaying recognition results and statistics
    print("Correct", correct)
    print("Wrong", wrong)
    print("Total Test Images", i)
    print("Percent", correct/i*100)
    print("Total Person", len(target_names))
    print("Total Train Images", no_of_images_of_one_person * len(target_names))
    print("Total Time Taken for reco:", time_elapsed)
    print("Time Taken for one reco:", time_elapsed/i)
    print("Training Time", training_time)

# Face recognition for video
if reco_type == 1:
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=7)

        i = 0
        for(x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            scaled = cv2.resize(roi_gray, (img_height, img_width))
            rec_color = (255, 0, 0)
            rec_stroke = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), rec_color, rec_stroke)

            new_cord = my_algo.new_cord_for_image(scaled)
            name = my_algo.recognize_face(new_cord)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (255, 255, 255)
            font_stroke = 2
            cv2.putText(frame, name + str(i), (x, y), font, 1, font_color, font_stroke, cv2.LINE_AA)
            i += 1



        cv2.imshow('Colored Frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

# Face recognition for group image
if reco_type == 2:
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    dir = r'images/Group/'


    frame = cv2.imread(dir + "group_image.jpg")


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)


    i = 0

    for(x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        scaled = cv2.resize(roi_gray, (img_height, img_width))
        rec_color = (0, 255, 0)
        rec_stroke = 5
        cv2.rectangle(frame, (x, y), (x+w, y+h), rec_color, rec_stroke)

        new_cord = my_algo.new_cord_for_image(scaled)
        print("New Cord PCA"+str(i), new_cord)
        name = my_algo.recognize_face(new_cord)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 0, 0)
        font_stroke = 5
        cv2.putText(frame, name + str(i), (x, y), font, 8, font_color, font_stroke, cv2.LINE_AA)
        i += 1
        cv2.imshow('Face', scaled)
        cv2.waitKey()
        
    frame = cv2.resize(frame, (1080, 568))
    cv2.imshow('Colored Frame', frame)
    cv2.waitKey()