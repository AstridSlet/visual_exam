# Assignment 5: CNNs on cultural image data

## Project description 
This assignment consists in building a classifier that can predict artists from paintings. This could e.g., be utilized if a new, never-before-seen painting is found, which is claimed to be of the artist Renoir. An accurate predictive model could assist art historians and archivists in determining the real artist. To solve this task a deep learning model that can classify paintings by their respective artists should be build. 

Requirements
* Save your script as cnn-artists.py
* You should save visualizations showing loss/accuracy of the model during training; you should also a save the output from the classification report.

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
$ source lang_venv/bin/activate
$ cd assignment_5
$ python cnn-artists.py
```

When running the script a classification report will be printet to the command line with F1-scores for each of the ten output classes (Hassam, Gaugin, VanGogh, Monet, Passaro, Renoir, Degas, Cezanne, Matisse and Sargent) and saved as a csv in the output folder. Additionally the script will produce an image of the model architecture and the training history, which is also saved in the output folder. 

