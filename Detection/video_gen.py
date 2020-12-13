# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob

from google.colab import drive
drive.mount('/content/drive')

imgdir = "data/camvid_res/test_res"
    # "/content/drive/Shared drives/EECS504/OverlayResults/Kitti_itr1/testing_seg_det/"
filenames = [img for img in glob.glob(imgdir + "*.png")]
filenames.sort()

img_array = []
for filename in filenames:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

# kitti test 3 fps
# camvid val 15 fps, test/train 5 fps
fps = 3
outdir = "TARGET_DIR"
out = cv2.VideoWriter(outdir + 'kitti_itr1_train_'+str(fps)+'fps.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()