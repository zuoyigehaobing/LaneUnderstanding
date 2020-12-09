"""
Main file for training Yolo model on Pascal VOC dataset
"""
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model_resnet import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
from imgaug import augmenters as iaa
import imgaug
import numpy as np


seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0.0005
EPOCHS = 35
NUM_WORKERS = 16
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/VOC/images"
LABEL_DIR = "data/VOC/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

augmentation = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Fliplr(1.0)),
    iaa.Sometimes(0.5, iaa.Resize((0.8, 1.2))),
    iaa.Sometimes(0.5, iaa.blur.AverageBlur(k=(5,5))),
    iaa.Sometimes(0.5, iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)),
    iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})),
])

# augmentation = None

transform = Compose([
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])



def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "data/VOC/train.csv",
        transform=transform,
        augmentation=augmentation,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    train_dataset.__getitem__(1)

    test_dataset = VOCDataset(
        "data/VOC/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    bestmap = -1
    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
        #
        #    import sys
        #    sys.exit()
        lr = LEARNING_RATE
        if epoch == 15:
            lr=1e-5
        if epoch == 25:
            lr=5e-6

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        if mean_avg_prec > bestmap:
            bestmap = mean_avg_prec
            checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(5)

        print(f"epoch {epoch} Train mAP: {mean_avg_prec}")
        print(f"bestmap {bestmap}")

        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()
