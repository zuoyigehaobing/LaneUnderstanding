import os
import torch
import numpy as np
from model import SegNet
from train import TRANSFER_LEARNING, IN_CHANNELS, NUM_CLASS, DEVICE
from matplotlib import pyplot as plt


def save_checkpoints(model, filename, epoch, optimizer, loss, train_acc, valid_acc):

    dirname = "/content/gdrive/MyDrive/Segmentation/check_points/large_aug"
    os.makedirs(dirname, exist_ok=True)

    save_path = os.path.join(dirname, filename)
    print(save_path)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'transfer_learning': TRANSFER_LEARNING
            }, save_path)


def load_checkpoints(filename, transfer_learning=True):

    dirname = "/content/gdrive/MyDrive/Segmentation/check_points/large_aug"
    file_path = os.path.join(dirname, filename)
    checkpoint = torch.load(file_path)
    if 'transfer_learning' in checkpoint:
        transfer_learning = checkpoint['transfer_learning']
    model = SegNet(IN_CHANNELS, NUM_CLASS, transfer_learning=transfer_learning)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Model Loaded: Epoch#{}\tLoss:{:.4f}\tTrainAcc:{:.4f}\tValidAcc:{:.4f}".format(
      checkpoint['epoch'],
      checkpoint['loss'],
      checkpoint['train_acc'],
      checkpoint['valid_acc'],
    ))
    return model.to(DEVICE)


def show_img(img_tensor):
    if len(img_tensor.shape) == 4:
        img = img_tensor.squeeze()
    else:
        img = img_tensor
    img = torch.movedim(img, 0, -1).type(torch.uint8).numpy()[:, :, ::-1]
    plt.imshow(img)
    plt.show()


def show_mask(mask_tensor):
    # shape height x width, single channel
    plt.imshow(mask_tensor)
    plt.show()


def show_pred_mask(model, img_tensor):
    if len(img_tensor.shape) == 4:
        source_img = img_tensor
    else:
        source_img = img_tensor.unsqueeze(0).to(DEVICE)

    model.eval()
    out = model(source_img)
    predicted = torch.argmax(out, dim=1)
    predicted = predicted.cpu().data.numpy()[0, :]
    plt.imshow(predicted)
    plt.show()


def show_all(model, img_tensor, mask_tensor, cmap='jet'):
    # regulate the image tensor
    if len(img_tensor.shape) == 4:
        img = img_tensor.squeeze()
        source_img = img_tensor
    else:
        img = img_tensor
        source_img = img_tensor.unsqueeze(0).to(DEVICE)

    # Raw image
    img = torch.movedim(img, 0, -1).type(torch.uint8).numpy()[:, :, ::-1]

    # Ground Truth mask
    mask = mask_tensor

    # predicted mask
    model.eval()
    out = model(source_img)
    predicted = torch.argmax(out, dim=1)
    predicted = predicted.cpu().data.numpy()[0, :]

    fig, axs = plt.subplots(1, 5, figsize=(20, 20), constrained_layout=True)

    axs[0].imshow(img)
    axs[0].set_title("Image")

    axs[1].imshow(mask)
    axs[1].set_title("Ground Truth")

    axs[2].imshow(predicted)
    axs[2].set_title("SegNet Outcome")

    img[:, :, 1] += (predicted * 255 * 0.15).astype(np.uint8)
    img[:, :, 0] += (predicted * 255 * 0.3).astype(np.uint8)
    axs[3].imshow(img, vmin=0, vmax=255)
    # axs[3].imshow(predicted, cmap=cmap, alpha=0.3)
    axs[3].set_title("SegNet Overlay")

    axs[4].imshow((predicted - mask.numpy()) ** 2)
    axs[4].set_title("Error")

    plt.show()

