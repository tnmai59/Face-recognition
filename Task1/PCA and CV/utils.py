from PIL import Image
import numpy as np
from numpy import asarray
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def create_database(folder):
    image_folders = []
    for f in os.listdir(folder):
        image_folders.append(f)

    X, y = [], []
    for f in tqdm(image_folders, total = len(image_folders)):
        # print(f)
        loc = folder + '/' + f
        folnum = int(f[1:])
        # folnum = int(f)
        for file in os.listdir(loc):
            file_loc = loc + '/' + file
            image = Image.open(file_loc)
            pixels = asarray(image)
            pixels = np.reshape(pixels,[1,pixels.shape[0]*pixels.shape[1]])

            if len(X) == 0:
                X = pixels
            else:
                X = np.concatenate((X,pixels),axis=0)
            y.append(folnum)
    return X, np.array(y)

def euclidean_distance(X_train, X_test, y_train):
    y_predict, match_index = [], []
    for i in tqdm(range(len(X_test))):
        min_ = np.argmin(np.sqrt(np.sum((X_train - X_test[i])**2,axis=1)))
        y_predict.append(y_train[min_])
        match_index.append(min_)
    return y_predict, match_index

def mahattan_distance(X_train, X_test, y_train):
    y_predict, match_index = [], []
    for i in range(len(X_test)):
        min_ = np.argmin(np.sum(np.abs(X_train - X_test[i]), axis = 1))
        y_predict.append(y_train[min_])
        match_index.append(min_)
    return y_predict, match_index

def minkowski_distance(X_train, X_test, y_train, p=3):
    y_predict, match_index = [], []
    for i in range(len(X_test)):
        distances = np.sum(np.abs(X_train - X_test[i])**p, axis=1)
        min_index = np.argmin(np.power(distances, 1/p))
        match_index.append(min_index)
        y_predict.append(y_train[min_index])
    return y_predict, match_index

def sse(X_train, X_test, y_train):
    y_predict, match_index = [], []
    for i in range(len(X_test)):
        d = np.sum((X_train-X_test[i])**2, axis=1)
        y_predict.append(y_train[np.argmin(d)])
        match_index.append(np.argmin(d))
    return y_predict, match_index

def ab_distance(X_train, X_test, y_train):
    y_predict, match_index = [], []
    for i in range(len(X_test)):
        d = -(np.sum(X_train*X_test[i],axis=1))/np.sqrt(np.sum(X_train**2,axis=1)*np.sum(X_test[i]**2))
        y_predict.append(y_train[np.argmin(d)])
        match_index.append(np.argmin(d))
    return y_predict, match_index

def ccb_distance(X_train, X_test, y_train):
    y_predict, match_index = [], []
    for i in range(len(X_test)):
        num = len(X_test)*np.sum(X_train*X_test[i], axis=1)-np.sum(X_train,axis=1)*np.sum(X_test[i])
        denom = np.sqrt((len(X_test)*np.sum(X_train**2,axis=1)-np.sum(X_train,axis=1)**2)*(len(X_test)*np.sum(X_test[i]**2)-np.sum(X_test[i])**2))
        d = -num/denom
        min_ = np.argmin(d)
        match_index.append(min_)
        y_predict.append(y_train[min_])
    return y_predict, match_index

def mahalanobis_distance(X_train, X_test, y_train):
    y_predict, match_index = [], []

    # Compute the covariance matrix for the training data
    covariance_matrix = np.cov(X_train, rowvar=False)

    # Compute the inverse of the covariance matrix
    try:
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError:
        # Handle a singular covariance matrix
        raise ValueError("Covariance matrix is singular; cannot compute Mahalanobis distance.")

    for i in range(len(X_test)):
        # Compute the Mahalanobis distance
        diff = X_train - X_test[i]
        d = np.sum(diff.dot(inv_covariance_matrix) * diff, axis=1)
        min_ = np.argmin(d)
        match_index.append(min_)
        # Assign the label of the nearest neighbor as the predicted label
        y_predict.append(y_train[min_])

    return y_predict, match_index

def chi2_distance(X_train, X_test, y_train):
    y_predict, match_index = [], []
    for i in range(len(X_test)):
        d = np.sum((X_train-X_test[i])**2/(X_train+X_test[i]+1e-10),axis=1)
        min_ = np.argmin(d)
        match_index.append(min_)
        y_predict.append(y_train[min_])
    return y_predict, match_index

def canbera_distance(X_train, X_test, y_train):
    y_predict, match_index = [], []
    for i in range(len(X_test)):
        d = np.sum(np.abs(X_train-X_test[i])/(np.abs(X_train)+np.abs(X_test[i])),axis=1)
        min_ = np.argmin(d)
        match_index.append(min_)
        y_predict.append(y_train[min_])
    return y_predict, match_index

def modified_mahattan_distance(X_train, X_test, y_train):
    y_predict, match_index = [], []
    for i in range(len(X_test)):
        d = np.sum(np.abs(X_train-X_test[i]),axis=1)/(np.sum(np.abs(X_train),axis=1)*np.sum(np.abs(X_test[i])))
        min_ = np.argmin(d)
        match_index.append(min_)
        y_predict.append(y_train[min_])
    return y_predict, match_index

def show_wrong_match(X_train, X_test, y_train, y_test, y_predict, match_index):
    original_shape = (112,92)
    wrong_list = []
    for i in range(len(y_predict)):
        if y_predict[i] != y_test[i]:
            wrong_list.append((i, match_index[i]))
    if len(wrong_list) == 0:
        print('All true')
        return
    fig, ax = plt.subplots(len(wrong_list),2,figsize=(10,10))
    if len(wrong_list) == 1:
        img_actual = np.reshape(X_test[wrong_list[0][0]], original_shape)
        ax[0].imshow(img_actual, cmap='gray')
        ax[0].set_title(f'Actual : {y_test[wrong_list[0][0]]}')
        ax[0].axis('off')

        # Display the predicted image
        img_predicted = np.reshape(X_train[wrong_list[0][1]], original_shape)
        ax[1].imshow(img_predicted, cmap='gray')
        ax[1].set_title(f'Predicted: {y_train[wrong_list[0][1]]}')
        ax[1].axis('off')
    else:
        for i in range(len(wrong_list)):
            # Display the actual image
            img_actual = np.reshape(X_test[wrong_list[i][0]], original_shape)
            ax[i, 0].imshow(img_actual, cmap='gray')
            ax[i, 0].set_title(f'Actual : {y_test[wrong_list[i][0]]}')
            ax[i, 0].axis('off')

            # Display the predicted image
            img_predicted = np.reshape(X_train[wrong_list[i][1]], original_shape)
            ax[i, 1].imshow(img_predicted, cmap='gray')
            ax[i, 1].set_title(f'Predicted: {y_train[wrong_list[i][1]]}')
            ax[i, 1].axis('off')
        plt.show()


def cross_validation_db(
        # X_origin,
          X, y, num_folds, func):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    avg = 0
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f'Fold: {fold}')
        X_train, X_test = X[train_index], X[test_index]
        # X_train_origin, X_test_origin = X_origin[train_index], X_origin[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_predict, match_index = func(X_train, X_test, y_train)
        avg += accuracy_score(y_test, y_predict)
        # show_wrong_match(X_train_origin, X_test_origin, y_train, y_test, y_predict, match_index)
    return avg/num_folds

def cross_validation_ml(X, y, num_folds, models):
    for name, model in models:
        kfold=KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        cv_scores=cross_val_score(model, X, y, cv=kfold)
        print("{} mean cross validations score:{:.5f}".format(name, cv_scores.mean()))