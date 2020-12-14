"""
Evaluation and visualization of the model
"""
import torch
from torch.utils.data import DataLoader
from train import DEVICE
import cv2
import numpy as np
import os

def avg_pixelwise_accuracy(model, dataset):
    if not dataset:
        return -1

    # change to eval mode
    model.eval()

    # get the dataloader using large batch since only forward pass required
    loader = DataLoader(dataset=dataset,
                        batch_size=8,
                        )

    correct, total = 0, 0
    for i, batch in enumerate(loader):

        # mount to GPU if available [Need to fix y]
        imgs, labels = batch['X'].to(DEVICE), batch['y'].to(DEVICE)

        # argmax X along dim 1
        out = model(imgs)
        predicted = torch.argmax(out, dim=1)
        total += labels.nelement()
        correct += predicted.eq(labels).sum().item()

    return correct / total


def write_to_dir(model, load_dir, save_dir):
    # change to eval mode
    model.eval()

    # create output dir
    os.makedirs(save_dir, exist_ok=True)

    # loop over the given directory
    for item in sorted(os.listdir(load_dir)):
        if '.png' in item:
            img = cv2.imread(os.path.join(load_dir, item))
            img = cv2.resize(img, (480, 320))
            img = np.moveaxis(img, -1, 0)
            img = torch.from_numpy(img).type(torch.float32)
            out = model(img.unsqueeze(0).to(DEVICE))
            predicted = torch.argmax(out, dim=1)
            predicted = predicted.cpu().data.numpy()[0, :]

            img = torch.movedim(img, 0, -1).type(torch.uint8).numpy().astype(
                np.float)
            img[:, :, 1] += (predicted * 255 * 0.5).astype(np.uint8)
            img = np.clip(img, 0, 255).astype(np.uint8)
            # img[:, :, 0] += (predicted * 255 * 0.7).astype(np.uint8)

            cv2.imwrite(os.path.join(save_dir, item), img)
