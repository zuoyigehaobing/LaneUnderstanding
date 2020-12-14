"""
Data preparation folder
- data loading
- data augmentation
- splitting training and testing sets
- create batches to be fed to the training process
"""
import os
import cv2
import numpy as np
import pandas as pd


# load the dataset
# crop the image or limit the size
# construct masks from the ground truth



class Dataset:

    def __init__(self, class_num=2):

        # typically two, one for the lane another for the background
        self.class_num = class_num




if __name__ == "__main__":

    pic_path = r"../data/debug/train/0001TP_009210.png"
    mask_path = pic_path.replace(".png", "_L.png").replace(r"/train/", r"/train_labels/")

    img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Debug")
