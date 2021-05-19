#!/usr/bin/env python

"""
Distinguishing fake and real faces using the Multi-Task Cascaded Convolutional Neural Network (MTCNN) (face detection), Googles  FaceNet model (face embeddings) and Sklearn (classification).
Parameters:
    infile: str <path-to-training-images>
    testsplit: float <train-test-split>

Usage:
    fake-faces.py --infile: str <path-to-training-images>
Example:
    $ python fake-faces.py --infile "my_data_folder"
"""


# import dependencies 

# data tools
import os, sys
sys.path.append(os.path.join(".."))

# sklearn tools
from sklearn.model_selection import train_test_split
from keras.models import load_model

# parsing arguments
import argparse

# utility functions 
from utils.face_utils import get_label
from utils.face_utils import retrieve_face
from utils.face_utils import load_all_faces
from utils.face_utils import plot_images
from utils.face_utils import single_embedding
from utils.face_utils import embed_faces
from utils.face_utils import train_SVM_model
from utils.face_utils import encode_labels



# define main function
def main():
    print("\n[INFO] Initialising analysis...")
    
    # initialise argumentparser
    ap = argparse.ArgumentParser()
    
    # define arguments
    ap.add_argument("-i", 
                    "--infile", 
                    required=False,
                    type = str,
                    help="Input path image data in quotations",  
                    default = os.path.join("data", "real_and_fake_face"))
    ap.add_argument("-t", 
                    "--testsplit", 
                    type = float, 
                    required=False, 
                    help="Train/test split",
                    default=0.2)


    # parse arguments to args
    args = vars(ap.parse_args())
    
    
    # define input paths to train and validation data 
    inpath = os.path.join("..", args["infile"])
    print(inpath)

    print("\n[INFO] Detecting faces...")
    
    # retreive faces from images + save y_labels (fake/real) 
    X, Y = load_all_faces(input_path = inpath)
    
    # get list of unique labels for classification report
    label_names = set(Y)
    print(f"Loaded {len(X)} images with the labels {label_names}")
    
    # make train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= args["testsplit"], random_state=42)
       
    print("\n[INFO] Creating data visualization for each label...") 
    # create visualizations
    plot_images(inpath, 'training_real', label = "Real")
    plot_images(inpath, 'training_fake', label = "Fake")
    
    print("\n[INFO] Embedding faces...")
    # load embedding model from folder
    embedding_model = load_model(os.path.join("..", "model","facenet_keras.h5"))
    
    # embed faces using loaded model and normalize pixel values
    trainX = embed_faces(embedding_model, X_train)
    testX = embed_faces(embedding_model, X_test)

    # encode labels
    trainY = encode_labels(y_train)
    testY = encode_labels(y_test)
    
    print("\n[INFO] Train SVM model...") 
    # create classification report 
    clf_report = train_SVM_model(trainX, trainY, testX, testY, class_names = label_names)
    
    print("\n Classification report:")
    print(clf_report)
    
    print("\n [INFO] ALL DONE! ") 
    
# define behaviour from command line 
if __name__=="__main__":
    main()
    
 
    
    
    
    
    
    
