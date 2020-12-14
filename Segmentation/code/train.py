"""
Training
"""
from matplotlib import pyplot as plt
import torch
from torchvision import models
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
from model import SegNet
import data
from utils import *
from eval import *

camvid_path = r'/content/gdrive/MyDrive/CamVid'
# for reproduction
torch.manual_seed(123)

# Pre config
IN_CHANNELS = 3
NUM_CLASS = 2

# training parameters
NUM_EPOCHS = 200
LEARNING_RATE = 0.1   # will decrease with epoch growing larger
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
LABEL_WEIGHTS = torch.FloatTensor([0.5, 2])
WEIGHT_DECAY = 0.0005
NUM_WORKERS = 0
PIN_MEMORY = False


# load the dataset
X_train, y_train = data.load_data(camvid_path+"/train", im_height=360, im_width=480, aug_copies=5)
print("Loaded {} training samples".format(len(X_train)))
X_valid, y_valid = data.load_data(camvid_path+"/val", im_height=360, im_width=480)
print("Loaded {} validation samples".format(len(X_valid)))
X_test, y_test = data.load_data(camvid_path+"/test", im_height=360, im_width=480)
print("Loaded {} testing samples".format(len(X_test)))

# Observe an image from KiTTi
kitti_observe = cv2.imread("/content/gdrive/MyDrive/KiTTi_dataset/testing_crop/um_000052.png")
kitti_observe_tensor = torch.from_numpy(np.moveaxis(kitti_observe, -1, 0))


def train(model, optimizer, loss_fn, X_train, y_train, X_valid=None,
          y_valid=None):
    # get the dataloader
    train_dataset = data.CamVidDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=False,
                              shuffle=True,
                              drop_last=True,
                              worker_init_fn=None
                              )

    valid_dataset = data.CamVidDataset(X_valid,
                                       y_valid) if X_valid is not None else None

    # for plotting and logging
    epoch_lst, iters, losses, train_acc_lst, val_acc_lst, iter_counter = [], [], [], [], [], 0

    # for checkpoints
    checkpoint_path_template = "Epoch{}_loss{:.4f}_trainacc{:.3f}_valacc{:.3f}.pth"
    acc_recoder, save_gap = 96, 0.2

    # start training
    for epoch in range(NUM_EPOCHS):

        # visualize an segmentation every 10 epoch
        if epoch % 4 == 0:
            show_all(model, X_valid[0, :], y_valid[0, :])
            show_all(model, kitti_observe_tensor.float(),
                     kitti_observe_tensor.float()[0, :])

        # Learning Rate Decay [Optional]
        if (epoch + 1) % 15 == 0:
            optimizer.param_groups[0]['lr'] /= 3
            print(
                "<=============== Learning Rate {} -> {}===== ==========>".format(
                    optimizer.param_groups[0]['lr'] * 3,
                    optimizer.param_groups[0]['lr']))

        epoch_loss = 0
        t_start = time.time()
        for i, batch in enumerate(train_loader):

            # mount to GPU if available [Need to fix y]
            imgs, labels = batch['X'].to(DEVICE), batch['y'][:, :, :].to(DEVICE)

            # change the mode to training mode and step training
            model.train()
            out = model(imgs)
            loss = loss_fn(out,
                           labels)  # note: soft-max should not be used here since it's included in nn.CrossEntropyLoss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save the current training information
            losses.append(loss)
            epoch_loss += loss
            iters.append(iter_counter)
            iter_counter += 1

        delta = time.time() - t_start

        # train_acc = avg_pixelwise_accuracy(model, train_dataset)
        train_acc = avg_pixelwise_accuracy(model, train_dataset) * 100
        valid_acc = avg_pixelwise_accuracy(model, valid_dataset) * 100
        print(
            "Epoch #{}\tLoss: {:.6f}\tTrain Acc: {:.3f}%\tValid Acc: {:.3f}%\tTime: {:.2f}s".format(
                epoch + 1, epoch_loss, train_acc, valid_acc, delta))

        train_acc_lst.append(train_acc)
        val_acc_lst.append(valid_acc)
        epoch_lst.append(epoch)

        # Save checkpoints
        if valid_acc > acc_recoder + save_gap:
            acc_recoder = valid_acc
            checkpoint_name = checkpoint_path_template.format(epoch + 1,
                                                              epoch_loss,
                                                              train_acc,
                                                              valid_acc)
            save_checkpoints(model, checkpoint_name, epoch + 1, optimizer,
                             epoch_loss, train_acc, valid_acc)

    # Plot the curve
    plt.title("Learning Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Learning Curve")
    plt.plot(epoch_lst, train_acc_lst, label="Train")
    plt.plot(epoch_lst, val_acc_lst, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Pixelwise Accuracy")
    plt.legend(loc='best')
    plt.show()


## train from scratch
torch.cuda.empty_cache()
TRANSFER_LEARNING = False
model = SegNet(3,2, transfer_learning=TRANSFER_LEARNING).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss(weight=LABEL_WEIGHTS).to(DEVICE)
train(model, optimizer, loss, X_train, y_train, X_valid, y_valid)


## train from vgg weights
torch.cuda.empty_cache()
TRANSFER_LEARNING = True
model_vgginit = SegNet(3, 2, transfer_learning=TRANSFER_LEARNING).to(DEVICE)
optimizer = torch.optim.Adam(model_vgginit.parameters(), lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss(weight=LABEL_WEIGHTS).to(DEVICE)
torch.cuda.empty_cache()
train(model_vgginit, optimizer, loss, X_train, y_train, X_valid, y_valid)

## load a model
model_name = r'../Epoch49_loss2.1120_trainacc97.727_valacc97.859.pth' # scratch
model_load = load_checkpoints(model_name)
show_pred_mask(model_load, X_train[0, :])
model = model_load


# report the accuracy on test dataset
torch.cuda.empty_cache()
model_selected = model
test_dataset = data.CamVidDataset(X_test, y_test)
test_accuracy = avg_pixelwise_accuracy(model_selected, test_dataset)
print("Selected model's pixelwise accuracy on test dataset : {:.5f}%".format(test_accuracy * 100))

# visualize some outputs
model_selected = model
show_all(model_selected, X_test[2, :], y_test[2, :], cmap='jet')
show_all(model_selected, X_test[11, :], y_test[11, :], cmap='jet')
show_all(model_selected, X_test[-2, :], y_test[-2, :], cmap='jet')
show_all(model_selected, X_test[43, :], y_test[43, :], cmap='jet')
show_all(model_selected, X_train[2, :], y_train[2, :], cmap='jet')
show_all(model_selected, X_train[11, :], y_train[11, :], cmap='jet')
show_all(model_selected, X_train[-2, :], y_train[-2, :], cmap='jet')
show_all(model_selected, X_train[43, :], y_train[43, :], cmap='jet')

# demo on a kitti image
model.eval()
kitti_img = cv2.imread("/content/gdrive/MyDrive/Segmentation/KiTTi/um_000007.png")
kitti_tensor = torch.from_numpy(np.moveaxis(kitti_img, -1, 0))
show_img(kitti_tensor)
show_pred_mask(model, kitti_tensor.float())
show_all(model, kitti_tensor.float(), kitti_tensor.float()[0, :])

# write results to disk
model_load = model
load_dir = r'/content/gdrive/MyDrive/KiTTi_dataset/testing_crop'
save_dir = r'/content/gdrive/MyDrive/KiTTi_dataset/testing_crop_segnet_new'
write_to_dir(model_load, load_dir, save_dir)

load_dir = r'/content/gdrive/MyDrive/KiTTi_dataset/training_crop'
save_dir = r'/content/gdrive/MyDrive/KiTTi_dataset/training_crop_segnet_new'
write_to_dir(model_load, load_dir, save_dir)

load_dir = camvid_path+"/train"
save_dir = camvid_path+"/train_segnet_new"
write_to_dir(model_load, load_dir, save_dir)

load_dir = camvid_path+"/test"
save_dir = camvid_path+"/test_segnet_new"
write_to_dir(model_load, load_dir, save_dir)

load_dir = camvid_path+"/val"
save_dir = camvid_path+"/val_segnet_new"
write_to_dir(model_load, load_dir, save_dir)
