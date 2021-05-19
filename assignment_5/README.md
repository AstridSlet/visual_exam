# Assignment 5: CNNs on cultural image data

## Project description 
This assignment consists in building a classifier that can predict artists from paintings. This could e.g., be utilized if a new, never-before-seen painting is found, which is claimed to be of the artist Renoir. An accurate predictive model could assist art historians and archivists in determining the real artist. To solve this task a deep learning model that can classify paintings by their respective artists should be build. 

Requirements
* Save your script as cnn-artists.py
* You should save visualizations showing loss/accuracy of the model during training; you should also a save the output from the classification report.

## Methods
The first step in this process was to make the data into a format that could be fed into the CNN model. All of the images are of different shapes and sizes, and the script therefore first resizes the images to have them be a uniform (smaller) shape (32 x 32 pixels as default). 

Additionally, it was necessary to make the the images into an array to be able to use them in the model and to extract 'labels' from filenames for use in the classification report. 

The model architecture of the convolutional network can be seen in the output folder (output/model_architecture.png). The network has a single convolutional layer (which has a set of learnable filters/kernels) and a relu activation layer, a flattening layer. Additionally, it has a fully connected layer and softmax layer for the final classification, as the model is used for multiclass classification. The model optimizes using categorical crossentropy, which should be used when there are two or more label classes (source).


## Usage

For this assignment the following command line script was created:
* cnn-artists.py
    * arguments:
        *  "--infile1", required=False, help="Input path, training data", type = str, default="training"
        * "--infile2", required=False, help="Input path, training data", type = str, default="validation"
        * "--width", required=False, help="Width on resized images", type=int, default=32
        *  "--height", required=False, help="Height on resized images", type=int, default = 32
        * "--epochs", required=False, help="Train/test split", type=int, default=15
        *  "--batchsize", required=False, help="Batchsize in model training", type=int, default = 32


The full data set, which includes paintings from 10 different impressionist artists, can be found here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data 
In order to run the script, the data set has to be downloaded and placed in the - currently empty – “data”-folder of this repository. You can then unzip the file in terminal with:

```
$ cd visual_exam/assignment_5/data
$ unzip name-of-file.zip
```


This will unpack the training and validation images in the data folder. If you have successfully cloned this repository, downloaded the data to the data-folder, and created the virtual environment visual_venv you can run the script for this project from command line with:


```
$ cd visual_exam
$ source visual_venv/bin/activate
$ cd assignment_5
$ python cnn-artists.py
```

When running the script a classification report will be printet to the command line with F1-scores for each of the ten output classes (Hassam, Gaugin, VanGogh, Monet, Passaro, Renoir, Degas, Cezanne, Matisse and Sargent) and saved as a csv in the output folder. Additionally the script will produce an image of the model architecture and the training history, which is also saved in the output folder. 

## Discussion of results
Sklearn’s function classification_report() returns the macro average F1 score for each class (averaging the unweighted mean per label). The F1-score can be interpreted as an average of precision and recall. 

* Recall expresses the proportion of positive samples (true positives) that are correctly classified as positive out of all the positive samples there are in total (true positives + false negatives), e.g. out of all the paintings by Monet there are, how many of these are classified as Monet? 
* Precision expresses the number of samples that were classified as positive that are indeed positive, e.g. out of the group of images that are predicted to be by Monet (true positives + false positives), how many paintings are indeed by Monet (true positives).

When looking at the F1-scores for the different classes it seems that the model performs somewhat dissimilar when predicting the different classes with F1-scores down to 0.28 (Monet) and 0.29 (Hassam) and up to 0.44 (Degas). Possibly, there are some more distinct patterns in the Degas’ paintings, making it easier for the model to correctly these paintings. 

For the worst classes (Monet and Hassam) the errors that the model make are slightly different. For the Monet-output class that model has higher precision than recall, and for the Hassam-output group recall is higher than precision. This means that the model is more likely to predict images as Hassam (higher recall) but is less precise in its predictions as compared to when the model has to predict the Monet output-class.    

When looking at the training history (output/training_history.png) you can see that the validation loss becomes higher than the training loss already after 4 epochs suggesting that the model overfits the training data rather quickly, while it still fails to reach high F1-scores for the output classes.  
