
# Assignment 2: Simple image search

## Project description 

Download the Oxford-17 flowers image data set, available at this link:
https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

Choose one image in your data that you want to be the 'target image'. Write a Python script which does the following:

* Use the cv2.compareHist() function to compare the 3D color histogram for your target image to each of the other images in the corpus one-by-one.
* In particular, use chi-square distance method, like we used in class. Round this number to 2 decimal places.
* Save the results from this comparison as a single .csv file, showing the distance between your target image and each of the other images. The .csv file should show the filename for every image in your data except the target and the distance metric between that image and your target. 
* Print the filename of the image which is 'closest' to your target image



## Usage
For this assignment the following command line script was created:
* image_search.py
    * arguments:
        *  "--inpath", required = False, type = str, help = "Path to input images", default = "data"
        *  "--outpath", required = False, type = str, help = "Output path, for csv file with chi^2 values", default = "chi_comparison.csv"
        *   "--target_image", required = False, type = str, help = "Target image file name", default = "image_0002.jpg"

In order to run the script, you need to unzip the data folder (data.zip) or download the data set from the link in the project description and place the images in a folder named “data”. You can unzip the data folder with:

```
$ cd visual_exam/assignment_2
$ unzip data.zip
```

If you have successfully cloned this repository, unzipped/downloaded the data and created the virtual environment visual_venv you can run the script from command line with:


```
$ cd visual_exam
$ source lang_venv/bin/activate
$ cd assignment_2
$ python image_search.py
```
When the script is run all chi values are saved along with the corresponding file name in a csv file. The file name of the image with the lowest chi-square distance is printed to the terminal along with the chi-square value between that image and the target image. 
