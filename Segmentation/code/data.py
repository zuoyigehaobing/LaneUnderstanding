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
from torch.utils.data import Dataset

# Global Variables
IM_WIDTH, IM_HEIGHT, IM_CHANNELS, NUM_CLASS = 224, 224, 3, 2
LANE_COLOR = np.array([128, 64, 128])
ARCWAY_COLOR = np.array([192, 0, 128])



class CamVidDataset(Dataset):
    def __init__(self, imgs, masks):
        self.X =imgs
        self.y = masks
    def __len__(self):
        return len(self.X[:, 0, 0, 0])

    def __getitem__(self, index):
        data = {'X': torch.FloatTensor(self.X[index, :, :, :]), 'y': torch.LongTensor(self.y[index, :, :, :])}
        return data

def load_data(img_dir, mask_dir=None, im_height=224, im_width=224):
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
    imgs = np.zeros((len(img_files), im_height, im_width, IM_CHANNELS))
    masks = np.zeros((len(img_files), NUM_CLASS, im_height, im_width))

    # loop over image files, resize accordingly
    counter = 0
    for item in img_files:

        # load an image
        img = cv2.imread(os.path.join(img_dir, item), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (im_width, im_height))    # REVERSE ORDER
        imgs[counter, :] = img

        # load a mask

        mask = cv2.imread(os.path.join(mask_dir,
                                       item.replace(".png", "_L.png")),
                          cv2.IMREAD_COLOR)

        # combine and highlight the foreground regions
        mask[(np.any(mask != LANE_COLOR, axis=2)) & (np.any(mask != ARCWAY_COLOR, axis=2))] = np.array([0, 0, 0])
        mask[np.any(mask != 0, axis=2)] = 1
        mask = cv2.resize(mask, (im_width, im_height))
        mask = mask[:, :, 0]
        masks[counter, 0, :] = 1 - mask
        masks[counter, 1, :] = mask

        # increase the counter
        counter += 1


    # change to torch tensor, move the channel forward to the 2nd axis
    imgs = torch.from_numpy(np.moveaxis(imgs, 3, 1))
    masks = torch.from_numpy(masks)

    # cast as float32
    return imgs.type(torch.float32), masks.type(torch.long)


def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    X, y = load_data(r"../raw_data/toy_dataset", im_height=360, im_width=480)
    dataset = CamVidDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print("passed")
