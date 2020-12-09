"""
General Tasks:
    1. load images/masks from given folder
    2. resize the image/masks
    3. convert into numpy array or tensor

Logs:
    - [Shan 08,Dec] Initialization
"""
import os
import cv2
import numpy as np
import torch

# Global Variables
IM_WIDTH, IM_HEIGHT, IM_CHANNELS, NUM_CLASS = 224, 224, 3, 2
LANE_COLOR = np.array([128, 64, 128])
ARCWAY_COLOR = np.array([192, 0, 128])


def load_data(img_dir, mask_dir=None):
    """
    Output format: <batch, channels, h, w>

    Note: no normalization here
    :param img_dir: the input dir
    :param mask_dir: the mask dir
    :return: a tensor to be fed into training
    """

    if mask_dir is None:
        mask_dir = img_dir + "_labels"

    # collect image files
    img_files = []
    for item in os.listdir(img_dir):
        if ".png" in item:
            img_files.append(item)

    # initialize the return array
    imgs = np.zeros((len(img_files), IM_HEIGHT, IM_WIDTH, IM_CHANNELS))
    masks = np.zeros((len(img_files), NUM_CLASS, IM_HEIGHT, IM_WIDTH))

    # loop over image files, resize accordingly
    counter = 0
    for item in img_files:

        # load an image
        img = cv2.imread(os.path.join(img_dir, item), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))    # REVERSE ORDER
        imgs[counter, :] = img

        # load a mask
        mask = cv2.imread(os.path.join(mask_dir,
                                       item.replace(".png", "_L.png")),
                          cv2.IMREAD_COLOR)

        # combine and highlight the foreground regions
        mask[(np.any(mask != LANE_COLOR, axis=2)) & (np.any(mask != ARCWAY_COLOR, axis=2))] = np.array([0, 0, 0])
        mask[np.any(mask != 0, axis=2)] = 255
        mask = mask[:, :, 0]

        masks[counter, 0, :] = 1



        counter += 1

    rval = torch.from_numpy(np.moveaxis(imgs, 3, 1))
    return rval



if __name__ == "__main__":

    X = load_data(r"../raw_data/toy_dataset")
    print(X.shape)
