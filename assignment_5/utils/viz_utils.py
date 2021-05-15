#!/usr/bin/env python

"""
Utility functions for convolutional neural network analysis of classical paintings. 

"""

# import dependencies

# data tools
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sklearn tools
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# parsing arguments
import argparse

# tf tools
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from pathlib import Path

# image processing 
import cv2



# define functions 
def load_data(input_path, width, height):
    """
    Load all train or validation data, resize images and make into arrays and normalize. 
    Return training images in a list + a list with their corresponding labels. 
    
    """
    array_list = []
    y_labels = []
    for folder in Path(input_path).glob("*"):
        y_lab = f"{folder.stem}"
        for each_file in Path(folder).glob("*"):
            image = cv2.imread(str(each_file))
            # resize image
            resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
            # append resized image and rescale
            array_list.append(np.array(resized.astype("float")/255.))
            y_labels.append(y_lab)
    return array_list, y_labels


def one_hot(ylabs):
    """
    Take a list of labels (str) and make it into onehot encoding.
    """
    # get values as array
    values = np.array(ylabs)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    ylabs_onehot = onehot_encoder.fit_transform(integer_encoded)
    return ylabs_onehot


def train_network(trainX, trainY, testX, testY, batch_size, epochs, width, height): 
    """
    Train convolutional neural network and save training history. 
    """
    # initialise model
    model = Sequential()

    # define CONV => RELU layer
    model.add(Conv2D(32, (3, 3), # depth of 32 of this conv layer and 3x3 kernels
                     padding="same", # use 0's around the outer edges of the image to pad
                     input_shape=(width, height, 3))) # the shape of the input data (32x32 pixels and 3 color channels)
    model.add(Activation("relu")) # adding a relu activation layer 

    # softmax classifier
    model.add(Flatten()) # flatten the architecture into a fully connected layer 
    model.add(Dense(trainY.shape[1])) # number of output classes
    model.add(Activation("softmax"))
    
    # define what the model optimises by 
    opt = SGD(lr =.01)
    
    # compile the model (define the loss function it should use to compile/train the model)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    # fit model and save training history
    history = model.fit(trainX, trainY, 
              validation_data=(testX, testY), 
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    
    return model 


def model_structure_viz(model):
    """
    Save visualization of model architecture.
    """
    impath = os.path.join("..","output","model_architecture.png")
    tensorflow.keras.utils.plot_model(model, to_file = impath, show_shapes = True)
    

def plot_train_hist(model, epochs):
    """
    Plot network training history. 
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), model.history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), model.history.history["val_loss"], label="val_loss", linestyle=":")
    plt.plot(np.arange(0, epochs), model.history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), model.history.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join("..","output", "training_history.png"))



def evaluate_model(model, testX, testY, batch_size, class_names):
    """
    Evaluate model performance and save classification report.
    """
    # make predictions
    predictions = model.predict(testX, batch_size = batch_size)
    # create clf report 
    clf_report = classification_report(testY.argmax(axis = 1),
                            predictions.argmax(axis = 1),
                            target_names = class_names)
    
    # create df for storing metrics
    df = pd.DataFrame( classification_report(testY.argmax(axis = 1),predictions.argmax(axis = 1),target_names = class_names,output_dict = True)).transpose().round(decimals=2)
        
    # save classification report    
    df.to_csv(os.path.join("..","output", "classification_report.csv"), index = True)
    
    return clf_report










