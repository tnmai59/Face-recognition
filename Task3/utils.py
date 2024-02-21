# Common
import os
import cv2 as cv
import numpy as np
from IPython.display import clear_output as cls
import tensorflow as tf
import re

# Data 
from tqdm import tqdm
from glob import glob

# Data Visuaalization
import plotly.express as px
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# Model
from tensorflow.keras.models import load_model

def get_train_test_split(rootpath, train_size=0.8):
    train_filepaths, test_filepaths = [], []
    dir_names = os.listdir(rootpath)
    for name in tqdm(dir_names, desc='Train test split', total=len(dir_names)):
        count = 0
        file_names = os.listdir(rootpath + name)
        for fn in file_names:
            file_path = rootpath + name + '/' + fn
            if count < len(file_names)*train_size:
                train_filepaths.append(file_path)
            else:
                test_filepaths.append(file_path)
            count += 1
    return train_filepaths, test_filepaths

def show_data(
    images: np.ndarray, 
    labels: np.ndarray,
    person_names: np.ndarray,
    GRID: tuple=(15,6),
    FIGSIZE: tuple=(25,50), 
    recog_fn = None,
    database = None
) -> None:
    
    """
    Function to plot a grid of images with their corresponding labels.

    Args:
        images (numpy.ndarray): Array of images to plot.
        labels (numpy.ndarray): Array of corresponding labels for each image.
        GRID (tuple, optional): Tuple with the number of rows and columns of the plot grid. Defaults to (15,6).
        FIGSIZE (tuple, optional): Tuple with the size of the plot figure. Defaults to (30,50).
        recog_fn (function, optional): Function to perform face recognition. Defaults to None.
        database (dictionary, optional): Dictionary with the encoding of the images for face recognition. Defaults to None.

    Returns:
        None
    """
    
    # Plotting Configuration
    plt.figure(figsize=FIGSIZE)
    n_rows, n_cols = GRID
    n_images = n_rows * n_cols
    
    # loop over the images and labels
    for index in range(n_images):
        
        # Select image in the corresponding label randomly
        image_index = np.random.randint(len(images))
        image, label = images[image_index], person_names[int(labels[image_index])]
        
        # Create a Subplot
        plt.subplot(n_rows, n_cols, index+1)
        
        # Plot Image
        plt.imshow(image)
        plt.axis('off')
        
        if recog_fn is None:
            # Plot title
            plt.title(label)
        else:
            recognized = recog_fn(image, database)
            plt.title(f"True:{label}\nPred:{recognized}")
    
    # Show final Plot
    plt.tight_layout()
    plt.show()

def load_image(image_path: str, IMG_W: int, IMG_H: int) -> np.ndarray:
    """Load and preprocess image.
    
    Args:
        image_path (str): Path to image file.
        IMG_W (int, optional): Width of image. Defaults to 160.
        IMG_H (int, optional): Height of image. Defaults to 160.
    
    Returns:
        np.ndarray: Preprocessed image.
    """
    
    # Load the image
    image = plt.imread(image_path)
    
    # Resize the image
    image = cv.resize(image, dsize=(IMG_W, IMG_H))
    
    # Convert image type and normalize pixel values
    image = image.astype(np.float32)
    
    return image

def image_to_embedding(image: np.ndarray, model) -> np.ndarray:
    """Generate face embedding for image.
    
    Args:
        image (np.ndarray): Image to generate encoding for.
        model : Pretrained face recognition model.
    
    Returns:
        np.ndarray: Face embedding for image.
    """
    
    # Obtain image encoding
    embedding = model.predict(image[np.newaxis,...])
    
    # Normalize bedding using L2 norm.
    embedding /= np.linalg.norm(embedding, ord=2)
    
    # Return embedding
    return embedding
    
def generate_avg_embedding(image_paths: list, model) -> np.ndarray:
    """Generate average face embedding for list of images.
    
    Args:
        image_paths (list): List of paths to image files.
        model : Pretrained face recognition model.
    
    Returns:
        np.ndarray: Average face embedding for images.
    """
    
    # Collect embeddings
    embeddings = np.empty(shape=(len(image_paths), 128))
    
    # Loop over images
    for index, image_path in enumerate(image_paths):
        
        # Load the image
        image = load_image(image_path)
        
        # Generate the embedding
        embedding = image_to_embedding(image, model)
        
        # Store the embedding
        embeddings[index] = embedding
        
    # Compute average embedding
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Clear Output
    cls()
    
    # Return average embedding
    return avg_embedding

def compare_embeddings(embedding_1: np.ndarray, embedding_2: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compares two embeddings and returns 1 if the distance between them is less than the threshold, else 0.

    Args:
    - embedding_1: A 128-dimensional embedding vector.
    - embedding_2: A 128-dimensional embedding vector.
    - threshold: A float value representing the maximum allowed distance between embeddings for them to be considered a match.

    Returns:
    - the distance between the embeddings if it is less than the threshold, else 0.
    """

    # Calculate the distance between the embeddings
    embedding_distance = embedding_1 - embedding_2

    # Calculate the L2 norm of the distance vector
    embedding_distance_norm = np.linalg.norm(embedding_distance)

    # Return the distance if it is less than the threshold, else 0
    return embedding_distance_norm if embedding_distance_norm < threshold else 0

def recognize_face(image: np.ndarray, database: dict, model, threshold: float = 1.0) -> str:
    """
    Given an image, recognize the person in the image using a pre-trained model and a database of known faces.
    
    Args:
        image (np.ndarray): The input image as a numpy array.
        database (dict): A dictionary containing the embeddings of known faces.
        threshold (float): The distance threshold below which two embeddings are considered a match.
        model (keras.Model): A pre-trained Keras model for extracting image embeddings.
        
    Returns:
        str: The name of the recognized person, or "No Match Found" if no match is found.
    """
    
    # Generate embedding for the new image
    image_emb = image_to_embedding(image, model)
    
    # Clear output
    cls()
    
    # Store distances
    distances = []
    names = []
    
    # Loop over database
    for name, embed in database.items():
        
        # Compare the embeddings
        dist = compare_embeddings(embed, image_emb, threshold=threshold)
        
        if dist > 0:
            # Append the score
            distances.append(dist)
            names.append(name)
    
    # Select the min distance
    if distances:
        min_dist = min(distances)
    
        return names[distances.index(min_dist)].title().strip()
    
    return "No Match Found"

def generating_labels(filepaths, pattern):
    labels = []
    for p in tqdm(filepaths, total=len(filepaths)):
        # label = re.findall(r'VN-celeb/(\d+)/', p)[0]
        label = re.findall(pattern, p)[0]
        labels.append(int(label))
    return labels



