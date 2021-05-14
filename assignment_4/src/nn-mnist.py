#!/usr/bin/env python

"""
Predicting digits in the mnist dataset using feedforward neural network. 
Parameters:
    infile: str <input-file-name>
    outfile: str <output-file-name>
    split: float <train-test-split>
    epochs: int <number-of-epochs-network>

Usage:
    nn-mnist.py --epochs <number-of-epochs-network>
Example:
    $ python nn-mnist.py --epochs 10
"""

# import dependencies 
import sys,os
import numpy as np
sys.path.append(os.path.join("..", "..", ".."))
import argparse
import pandas as pd

import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.neuralnetwork import NeuralNetwork
#from utils.neuralnetwork import plot_train_hist
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# define main function
def main():
    print("\nInitialising analysis...")
    
    # initialise argumentparser
    ap = argparse.ArgumentParser()
    
    # define arguments
    ap.add_argument("-i", 
                    "--infile", 
                    required=False, 
                    type = str, 
                    help="Input filename",
                    default="mnist_784")
    ap.add_argument("-s", 
                    "--split", 
                    required=False,
                    type=float,
                    help="Train/test split", 
                    default=0.2)
    ap.add_argument("-o", 
                    "--outfile", 
                    required=False, 
                    type = str,
                    help="Output csv filename", 
                    default = "metrics_nn.csv")
    ap.add_argument("-e", 
                    "--epochs", 
                    required=False, 
                    type = int, 
                    help="Number of epochs",
                    default = 1000)



    # parse arguments to args
    args = vars(ap.parse_args())
    
    # fetch args
    input_name = args["infile"]
    split_value = args["split"]
    n_epochs = args["epochs"]

    
    print("\nFetching data...")
    
    # fetch data 
    digits = datasets.load_digits()
    
    # convert to floats
    data = digits.data.astype("float")
    
    # perform min-max regularization
    data = (data - data.min())/(data.max() - data.min())
    
    # create train/test split 
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                        digits.target,
                                                        test_size = args["split"])
    
    # scaling the input features
    X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min())
    X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min())
    
    # convert labels from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    
    print("\nTraining network...")

    # train neural network
    nn = NeuralNetwork([X_train.shape[1], 20, 15, 10]) # no. of input features, hiddenlayer1, hiddenlayer2, no. of output features
    nn.fit(X_train, y_train, epochs = n_epochs)
    
    # save fitted model 
    model_fit = nn.fit(X_train, y_train, epochs = n_epochs)
    
    # calculate predictions for the test set and print in terminal
    predictions = nn.predict(X_test_scaled)
    predictions = predictions.argmax(axis=1)
    print("\nCalculated performance metrics: ")
    print(metrics.classification_report(y_test.argmax(axis=1), predictions))
    
    # create df for storing metrics
    df = pd.DataFrame(metrics.classification_report(y_test.argmax(axis=1), 
                                                    predictions, 
                                                    output_dict=True)).transpose().round(decimals=2)

    # save classification report    
    df.to_csv(os.path.join("..", "out", args["outfile"]), index = True)

    
# define behaviour from command line 
if __name__=="__main__":
    main()
     
 
    
    
    
    
    
    
