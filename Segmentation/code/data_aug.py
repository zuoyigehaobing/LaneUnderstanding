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
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm


# Global Variables
IM_WIDTH, IM_HEIGHT, IM_CHANNELS, NUM_CLASS = 224, 224, 3, 2
LANE_COLOR = np.array([128, 64, 128])
ARCWAY_COLOR = np.array([192, 0, 128])

# Aug
ia.seed(1)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        # horizontal flip, 0.3 chance
        iaa.Fliplr(0.3),

        # random crop and center the image
        iaa.Crop(percent=(0, 0.15)),

        # GaussianBlur/Average Blur/ MedianBlur
        iaa.Sometimes(
            0.5,
            iaa.OneOf([
                iaa.GaussianBlur((0, .8)),
                iaa.AverageBlur(k=(1, 3)),
                iaa.MedianBlur(k=(1, 3)),
            ]),

        ),


        # Strengthen or weaken the contrast in each image.
        iaa.Sometimes(
            0.9,
            iaa.LinearContrast((0.6, 2.0)),
        ),


        # 0.1 channel chance to apply random noise
        iaa.Sometimes(
            0.5,
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.1),
        ),

        # intensity change
        iaa.Multiply((0.8, 1.2), per_channel=0.3),

        # Random coarse dropout to simulate shadow
        iaa.Sometimes(
            0.2,
            iaa.CoarseDropout(
                (0.01, 0.1), size_percent=(0.02, 0.05),
                per_channel=0.2
            ),
        )

    ],
    # do all of the above augmentations in random order
    random_order=True
)



class CamVidDataset(Dataset):
    def __init__(self, imgs, masks):
        self.X =imgs
        self.y = masks
    def __len__(self):
        return len(self.X[:, 0, 0, 0])

    def __getitem__(self, index):
        data = {'X': torch.FloatTensor(self.X[index, :, :, :]), 'y': torch.LongTensor(self.y[index, :, :])}
        return data


def load_data(img_dir, mask_dir=None, im_height=224, im_width=224, aug_copies=0):
    """
    Output format: <batch, channels, h, w>

    Note: no normalization here
    :param img_dir: the input dir
    :param mask_dir: the mask dir
    :return: a tensor to be fed into training
    """

    class_pixels = [1, 1]
    if mask_dir is None:
        mask_dir = img_dir + "_labels"

    # collect image files
    img_files = []
    for item in os.listdir(img_dir):
        if ".png" in item:
            img_files.append(item)

    # initialize the return array
    imgs = np.zeros((len(img_files) * (aug_copies + 1), im_height, im_width, IM_CHANNELS))
    masks = np.zeros((len(img_files) * (aug_copies + 1), im_height, im_width))

    # loop over image files, resize accordingly
    counter = 0
    for idx in tqdm(range(len(img_files))):

        item = img_files[idx]

        # load an image
        img = cv2.imread(os.path.join(img_dir, item), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (im_width, im_height))    # REVERSE ORDER
        imgs[counter, :] = img

        # load a mask if exists else ignore
        if os.path.exists(mask_dir):
            mask = cv2.imread(os.path.join(mask_dir,
                                           item.replace(".png", "_L.png")),
                              cv2.IMREAD_COLOR)

            # combine and highlight the foreground regions
            mask[(np.any(mask != LANE_COLOR, axis=2)) & (np.any(mask != ARCWAY_COLOR, axis=2))] = np.array([0, 0, 0])
            mask[np.any(mask != 0, axis=2)] = 1
            mask = cv2.resize(mask, (im_width, im_height))
            mask = mask[:, :, 0]
            masks[counter, :, :] = mask

            # update class pixel
            class_pixels[1] += mask.sum()
            class_pixels[0] += (1 - mask).sum()

        # get access to the mask
        mask = masks[counter, :, :]

        # increase the counter
        counter += 1

        imshow(img)
        # imshow(mask)

        # augmentation starts
        for k in range(aug_copies):
            segmap = SegmentationMapsOnImage(mask.astype(np.int32), shape=img.shape)
            images_aug_i, segmaps_aug_i = seq(image=img, segmentation_maps=segmap)
            imgs[counter, :] = images_aug_i
            masks[counter, :] = segmaps_aug_i.get_arr()
            counter += 1

            imshow(imgs[counter-1, :].astype(np.uint8))
            # imshow(masks[counter-1, :].astype(np.float))

    # change to torch tensor, move the channel forward to the 2nd axis
    imgs = torch.from_numpy(np.moveaxis(imgs, 3, 1))
    masks = torch.from_numpy(masks)

    # show suggested weights
    print("Bkg : Lane = {} : {}".format(class_pixels[0], class_pixels[1]))

    # cast as float32
    return imgs.type(torch.float32), masks.type(torch.long)


def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    X, y = load_data(r"../raw_data/toy_dataset", im_height=360, im_width=480, aug_copies=5)
    print(X.shape, y.shape)


    # for i in range(X.shape[0]):
    #     imshow(np.moveaxis(X[i, :].numpy().astype(np.uint8), 0, -1))
    #     imshow(y[i, :].numpy().astype(np.float))
    dataset = CamVidDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print("passed")
