# Final project: Detecting fake faces

## Project description 
For this project I chose to work on creating a script that could distinguish fake faces from real faces, inspired by an interests in the phenomenon of deepfakes and how to detect them. For computational reasons I have chosen to use a smaller data set from Kaggle, which includes 2041 images in total (960 fake, 1081 real). The fake faces have been made by adding subparts from different faces (eyes, nose, mouth, or whole face) using photoshop. When running the script, sample plots of both classes (Fake and Real), are produced in the output folder to illustrate how the classes differ. 

## Methods
Though the images in the chosen data set are relatively centered on the faces, I chose to use the Multi-Task Cascaded Convolutional Neural Network (MTCNN) which is a state-of-the-art deep learning model for face detection (Zhang et al., 2019). The algorithm generates a bounding box (defined with coordinates) when it detects a face. These coordinates can then be used to create an image array with just the contents of the bounding box. In other words, you can use this algorithm to ‘zoom’ in on just the face. In this way, when you train a classifier to distinguish between subclasses in your data, you are preventing the model from learning from background features of the image. A possible extension of this project would be to use a data set which included full-figure images of people instead of the sample data set provided here, which is made possible by including this step with the MTCNN model.   

After extracting the faces, I used Googles pretrained FaceNet model (Schroff et al., 2015), which is placed in the ‘model’ folder of this repository. The FaceNet model is a convolutional neural network, which has been trained on the MS-Celeb-1M dataset that consists of 10 million face images from the internet. The model maps high-dimensional data (image arrays) into low-dimensional representations of 128 vectors commonly referred to as embeddings (ibid). In other words, the model takes an image and outputs 128 most important features. When the model is fit these vectors are constructed in a way so the images of different classes in the data set are separated as much as possible. Thus, when feeding the model this data set the model will try to separate the fake and real images as much as possible. 

The output of the FaceNet model is afterwards fed to a SVM classifier using Sklearn (Pedregosa et al., 2011), where the kernel choice (linear vs RBF kernel) and C-parameter is optimized using Sklearn’s GridSearch function.  



## Usage

For this assignment the following command line script was created:
* fake-faces.py
    * arguments:
        *  "--infile", required=False, type = str, help="Input path image data in quotations", default = os.path.join("data", "real_and_fake_face")
        *  "--testsplit", type = float, required=False, help="Train/test split", default=0.2


The main script make use of a series of utility functions provided in the script face_utils.py, which can be found in the utils folder. In order to run the main script, the data set has to be downloaded and placed in the - currently empty – “data”-folder of this repository. The full data set with real and fake faces can be found here: https://www.kaggle.com/ciplab/real-and-fake-face-detection 
After placing the data in the data folder you can unzip the file in terminal with:

```
$ cd visual_exam/final_project/data
$ unzip name-of-file.zip
```

Additionally, the pretrained model, which is stored in the ‘model’ folder has to be unzipped with:

```
$ cd visual_exam/final_project/model
$ unzip name-of-file.zip
```

If you have successfully cloned this repository, downloaded the data in the data folder, unzipped the pretrained FaceNet model and created the virtual environment visual_venv you can run the main script from command line with:

```
$ cd visual_exam
$ source lang_venv/bin/activate
$ cd final_project/src
$ python fake-faces.py
```

## Discussion of results
The output of the SVM model yielded a F1 score of 0.55 for ‘real’ faces and 0.61 for ‘fake’ faces, suggesting that the model was better at recognizing the fake faces. Both precision and recall is higher for the ‘fake’ faces output class, meaning that the model both detects more fake faces and is more precise (fewer false positives) than when detecting the ‘real’ faces. 

If an algorithm of this kind were to be utilized in the real world one could argue, that it was better to have a model that prioritized the recognition of fake faces, so it would be able to make an ‘alarm call’ if presented with images that could be suspicious. Thus, one could argue that recall should be prioritized over precision for the ‘fake’ faces class when optimizing such a model. For this model however, precision was higher (0.63) than precision (0.59). 

Overall, the results of this model are however not very impressive with a macro F1 score of 0.58 which is just above chance. A possible explanation is that the nature of this classification problem is slightly deviant of what the FaceNet model is trying to do. As described, the model is mapping high-dimensional data (image arrays) into low-dimensional representations (embeddings) of 128 vectors. The optimization goal of the model is that these vectors should be different for faces from different classes and similar for faces from the same class. It can be hard to interpret what these vectors mean, as we are simply asking the model to construct them in a way that solves the job and not in a way that makes sense in the real world. It has been suggested that something like distance between eyes could hidden in the numbers of an embedding vector. If you were using the model to distinguish different persons from each other, you would expect this distance to be the same for the same person and thus such a vector would be very beneficial in distinguishing classes. 

But in this particular classification problem it is naturally no the same person that is always the ‘fake’ person. Instead, if the algorithm was indeed looking at distances between eyes, it would have to come up with a vector representation of ‘unreal/fake’ distances between eyes vs. ‘normal/real’ distances between eyes. It is easy to imagine that a model would need more than 2000 images to do detect generalizable patterns of ‘abnormal/fake’ face structures. It would therefore be interesting to try out the face detection pipeline presented in this project on a substantially larger image corpus. 


