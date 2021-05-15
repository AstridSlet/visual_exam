#!/usr/bin/env python

"""
For all images in data folder compute color histogram and compare histogram to target image histogram.
Parameters:
    inpath: str <path-to-images>
    outpath: str <path-for-output-file>
    target_image: str <name-of-target-img>
Usage:
    image_search.py --target_image <name-of-target-img>
Example:
    $ python image_search.py --target_image image_0002.jpg 
"""

# import dependencies
import os, sys
from pathlib import Path 
import cv2 
import pandas as pd 
import argparse


# define functions 
def hist_norm_function(image_path):
    '''
    Function for creating and normalizing color histogram for 3D color images.
    '''
    # load image
    image = cv2.imread(image_path)
    histogram = cv2.calcHist([image], [0,1, 2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    hist_norm = cv2.normalize(histogram, histogram, 0,255, cv2.NORM_MINMAX)
    return hist_norm


def calc_chi_values(images_path, target_path):
    '''
    Loop over images, compute histograms and compare histogram to target image.
    Return: 
        * list of file names
        * list of chi^2 values for each comparison.
    '''
    # create histogram for target image
    target_histogram = hist_norm_function(str(target_path))
    # define empty lists to store info
    filename_list = []
    chi_list = []
    for each_file in Path(images_path).glob("*.jpg"):
    # skip target image
        if os.path.basename(each_file) == os.path.basename(target_path):
            pass
    # loading rest of images
        else:
            # get filename
            file_name = os.path.basename(each_file)
            # save filename
            filename_list.append(file_name)
            # compute histogram for each image
            histogram = hist_norm_function(str(each_file))
            # compute chi value
            chi_list.append(round(cv2.compareHist(target_histogram, histogram, cv2.HISTCMP_CHISQR), 2))
    return filename_list, chi_list


# define main function 
def main():
    # initialise argumentparser
    ap = argparse.ArgumentParser()
    
    # define parameters
    ap.add_argument("-i",
                    "--inpath",
                    required = False,
                    type     = str,
                    help = "Path to input images",
                    default = "data")
    ap.add_argument("-o",
                    "--outpath",
                    required = False,
                    type = str,
                    help = "Output path, for csv file with chi^2 values",
                    default = "chi_comparison.csv")
    ap.add_argument("-t",
                    "--target_image",
                    required = False,
                    type = str,
                    help = "Target image file name",
                    default = "image_0002.jpg")
    
    # parse arguments to args
    args = vars(ap.parse_args())
    
    
    # define path to input images
    images_path = os.path.join("..", args["inpath"])

    # define target image path
    target_path = os.path.join("data", args["target_image"])
    print(f"The target image path is {target_path}")
    # get filenames and chi^2 values for all images
    filename_list, chi_list = calc_chi_values(images_path, target_path)
   
    
    
    # create df and sort by chi value
    df = pd.DataFrame(zip(filename_list, chi_list), 
               columns =["filename", "distance"])
    df = df.sort_values(by = "distance")
    
    # define output filepath
    outpath = os.path.join("..","output", args["outpath"])
    
    # save df 
    df.to_csv(outpath, index=False)
    
    # print minimum value to command line
    return(print(f"The file with the lowest distance to the target image is {df.filename.iloc[0]} with a distance of {df.distance.iloc[0]}"))




                                                  
# Define behaviour when called from command line
if __name__=="__main__":
    main()
