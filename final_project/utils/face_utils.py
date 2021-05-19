#!/usr/bin/env python

"""
Utility functions for retrieiving real and fake faces and training a classifyer to distinguish them.  

"""

# import dependencies

# standard library
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


import cv2

# detect faces
import mtcnn
from mtcnn.mtcnn import MTCNN


# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC



from PIL import Image
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import expand_dims



# define functions

def get_label(folder):
    """
    Retrieve the y_label from folder name. 
    """
    folder = f"{folder.stem}"
    x = folder.index("_")
    y_lab = folder[x+1:x+5]
    return y_lab


def retrieve_face(result_list, pixels, required_size):
    """
    1. Retrieve face using the face coordinates detected with MTCNN
    2. Resize face array to required input size of the Facenet model
    """
    # else extract the bounding box of the face
    x1, y1, width, height = result_list[0]['box']
    # make sure not indices are negative
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def load_all_faces(input_path, required_size = (160, 160)):
    """
    1. Load images
    2. Retrieve faces and resize them so they can go into the Facenet model
    3. Convert face arrays to RGB color
    4. Save face arrays and y_labels
    """
    # initiate face detector
    detector = MTCNN()
    # create empty lists
    array_list = []
    y_labels = []
    for folder in Path(input_path).glob("*"):
        for each_file in Path(folder).glob("*.jpg"):
            # load image
            image = Image.open(str(each_file))
            pixels = np.asarray(image)
            # detect faces in the image
            results = detector.detect_faces(pixels)
            if len(results)==0:
                pass
            else:
                # retrieve face array 
                face_array = retrieve_face(results, pixels, required_size)
                # convert from rgb to vgr
                image_rgb = face_array[:, :, [2, 1, 0]]
                # save face array and y-label 
                array_list.append(image_rgb)
                y_labels.append(get_label(folder = folder))
    return array_list, y_labels



def plot_images(path, output_class, label):
    """
    Load nine images from each class and plot in 3x3 grid. 
    """
    # define path to images
    image_dir = os.path.join(path, output_class)
    # initiate counter
    k = 0
    # create figure
    fig, ax = plt.subplots(3,3, figsize=(10,10))
    fig.suptitle("Example plot: " + label + " faces")
    for j in range(3):
        for i in range(3):
            # load image
            img = load_img(os.path.join(image_dir, os.listdir(os.path.join(image_dir))[k]))          
            ax[j,i].imshow(img)
            # remove numbers from axes
            ax[j,i].axis('off')
            k +=1
    fig.tight_layout()
    plt.suptitle("Example plot: " + label + " faces")
    plt.savefig(os.path.join("..","output", f"example_images_{label}.png"))
    
    
    
    
def single_embedding(model, face_pixels):
    """
    Get face embedding for a single face using facenet model. 
    """ 
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples) 
    return yhat[0]



def embed_faces(embedding_model, faces_list):
    """
    1. Get face embeddings for all detected faces saved in list of faces. 
    2. Normalize values with l2 normalization and return list of face embeddings (arrays). 
    """
    # define normalizer 
    input_normalizer = Normalizer(norm='l2')
    
    # create list to store face embeddings 
    faces_embedding = list()
    
    for face in faces_list:
        embedding = single_embedding(embedding_model, face)
        faces_embedding.append(embedding)
        # normalize the retrieved faces 
        final_embeddings = input_normalizer.transform(np.asarray(faces_embedding))
    return final_embeddings


def encode_labels(label_list):
    """
    Encode y_labels so they can be fed to the SVM model. 
    """
    # define label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)
    # encode labels
    labels_transformed = label_encoder.transform(label_list)
    return labels_transformed


# ALTERNATIVE MODEL TRAINING FUNCTION
def train_SVM_model(trainX, trainY, testX, testY, class_names): 
    """
    1. Train and evaluate SVM model - use data augmentation to increase data sample. 
    2. Save classification report.
    """
    # define model
    base_model = SVC()
    # define parameters
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.5, 1, 2]}
    # initiate gridsearch
    model = GridSearchCV(base_model, parameters)
    
    # train model
    fitted = model.fit(trainX, trainY)
    
    # print best parameters
    print(fitted.best_estimator_)
    
    
    # make predictions
    predictions = model.predict(testX) 
    # create clf report 
    clf_report = classification_report(testY,
                            predictions,
                            target_names = class_names)
    
    # create df for storing metrics
    df = pd.DataFrame.from_dict(classification_report(testY,predictions,
                                                      target_names = class_names, 
                                                      output_dict = True)).round(decimals=2)
        
    # save classification report    
    df.to_csv(os.path.join("..","output", "SVM_clf_report_gridsearch.csv"), index = True)
    return clf_report




