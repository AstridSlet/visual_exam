# Assignment 4: Classification benchmarks using Logistic Regression and a Neural Network

## Project description 
This assignment consists in creating two command-line tools which can be used to perform a simple classification task on the MNIST data set and print the output to the terminal. These scripts can thus be used to provide easy-to-understand benchmark scores for evaluating these models.

You should create two Python scripts. One takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. The other should take the full MNIST dataset, train a neural network classifier, and print the evaluation metrics to the terminal.

Bonus Challenge
* Have the scripts save the classifier reports in a folder called out, as well as printing them to screen. And the user should be able to define the file name as a command line argument.


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
$ source lang_venv/bin/activate
$ cd assignment_4
$ python lr-mnist.py
$ python nn-mnist.py
```

