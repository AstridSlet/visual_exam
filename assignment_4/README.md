# Assignment 4: Classification benchmarks using Logistic Regression and a Neural Network

## Project description 
This assignment consists in creating two command-line tools which can be used to perform a simple classification task on the MNIST data set and print the output to the terminal. These scripts can thus be used to provide easy-to-understand benchmark scores for evaluating these models.

You should create two Python scripts. One takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. The other should take the full MNIST dataset, train a neural network classifier, and print the evaluation metrics to the terminal.

Bonus Challenge
* Have the scripts save the classifier reports in a folder called out, as well as printing them to screen. And the user should be able to define the file name as a command line argument.

## Methods
The project contains two scripts placed in the src folder:
•    A script that loads an image data set and trains a simple logistic regression classifier using Sklearn to distinguish between 10 digits.
•    A script that loads an image data set and trains a feed forward neural network with two hidden layers using Sklearn to distinguish between 10 digits. 

Both scripts make use of a series of functions provided in a separate script (utils/classifier_utils.py). Additionally, the neural network makes use of a python class (utils/neuralnetwork.py) developed in class. 



## Usage

For assignment two command line scripts were created:
* lr-mnist.py (logistic regression)
    * arguments:
        * "--infile", required=False, help="Input filename", type = str, default="mnist_784"
        * "--split", required=False, help="Train/test split", type=float, default=0.2
        * "--outfile", required=False, help="Output csv filename", default = "metrics_logreg.csv"
* nn-mnist.py (neural network)
    * arguments:
        * "--infile", required=False, help="Input filename", type = str, default="mnist_784"
        * "--split", required=False, help="Train/test split", type=float, default=0.2
        * "--outfile", required=False, help="Output csv filename", default = "metrics_nn.csv"
        *  "--epochs", required=False, help="Number of epochs", type = int, default = 1000

If you have successfully cloned this repository and created the virtual environment visual_venv you can run the scripts from command line with:

```
$ cd visual_exam
$ source visual_venv/bin/activate
$ cd assignment_4
$ python lr-mnist.py
$ python nn-mnist.py
```

All arguments described above are not required. If nothing else is specified, the code will run with the default settings:
* For the logistic regression: The script will run using the sklearn data set mnist_784, the train/test split will be 0.2 and the output name will be metrics_logreg.csv (the metrics are printed to command line and the file saved in the “out” folder). 
* For the neural network: The script will run using the sklearn dataset 'digits', the train/test split will be 0.2, the output name will be metrics_nn.csv (the metrics are printed to command line and the file saved in the “out” folder) and the number of epochs will be 1000. The neural network as a default has two hidden layers. During training the training loss of the model is printed for every 100 epochs. 

## Discussion of results
When using Sklearn’s function classification_report() it returns the macro average F1 score for each class (averaging the unweighted mean per label) (Pedregosa et al., 2011). When viewing the output of the simple logistic regression, the network actually reaches very high F1-scores for all ten output classes with F1-values between 0.88 (for the digits 8 and 9) and up to 0.97 (for the digit 0) and a macro average of 0.92. The slightly lower F1-scores for the digits 8 and 9 could be due to these digits having some similarity in how they are shaped making it harder for the model to distinguish and correctly classify them. 

When training the neural network, the accuracies even reach 1 for some classes, which is almost unrealistically high. When viewing the training loss printed to command line reveals that the training loss is reduced to almost 0 meaning that this model might perform worse when tested on out-of-sample images due to overfitting on the training data. 

#### References:
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M.,
Prettenhofer, P., Weiss, R., & Dubourg, V. (2011). Scikit-learn: Machine learning in
Python. the Journal of machine Learning research, 12, 2825–2830.

