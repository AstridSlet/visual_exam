#!/usr/bin/env python

"""
Predicting digits in the mnist dataset using a logistic regression classifier. 
Parameters:
    infile: str <input-file-name>
    split: float <train-test-split>
    epochs: int <number-of-epochs-network>

Usage:
    lr-mnist.py --split <train-test-split>
Example:
    $ python lr-mnist.py --split 0.2
"""

# import dependencies 
import sys,os
import numpy as np
import argparse
sys.path.append(os.path.join(".."))
import pandas as pd

# import utility function
import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
                    type = float,
                    help="Train/test split", 
                    default=0.2)
    ap.add_argument("-o", 
                    "--outfile", 
                    required=False,
                    type = str,
                    help="Output csv filename", 
                    default = "metrics_logreg.csv")


    # parse arguments to args
    args = vars(ap.parse_args())
    
    print("\nFetching data...")
    
    # fetch data 
    X, y = fetch_openml(args["infile"], return_X_y = True)
    
    # make into numpy array 
    X = np.array(X)
    y = np.array(y)
    
    # create train/test split NB you need this 0.2 to be parsed as an arg!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args["split"])
    
    # min-max scaling the input features
    X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min())
    X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min())
    
    print("\nTraining model...")

    # train logostic regression model 
    clf = LogisticRegression(penalty='none', 
                         tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train)
    
    
    # calculate predictions for the test set 
    y_pred = clf.predict(X_test_scaled)
    
    
    print("\nCalculated performance metrics: ")
    # calculate metrics and print in terminal 
    cm = metrics.classification_report(y_test, y_pred)
    print(cm)
    
    # create df for storing metrics
    df = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict=True)).transpose().round(decimals=2)

    
    # save classification report    
    df.to_csv(os.path.join("..", "out", args["outfile"]), index= True)

    
# define behaviour from command line 
if __name__=="__main__":
    main()
    
 
    
    
    
    
    
    
