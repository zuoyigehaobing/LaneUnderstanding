import os
import torch
from model import SegNet


def load_checkpoints(file_path, transfer_learning=True):

    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    if 'transfer_learning' in checkpoint:
        transfer_learning = checkpoint['transfer_learning']
    model = SegNet(3, 2, transfer_learning=transfer_learning)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Model Loaded: Epoch#{}\tLoss:{:.4f}\t"
          "TrainAcc:{:.4f}\tValidAcc:{:.4f}".format(checkpoint['epoch'],
                                                    checkpoint['loss'],
                                                    checkpoint['train_acc'],
                                                    checkpoint['valid_acc'],
                                                    ))

    return model
