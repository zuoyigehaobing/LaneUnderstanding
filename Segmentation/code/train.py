"""
To be filled by Songlin, Shan

Required files before implementation"
    - data.py
    - model.py
    - loss.py (* better to have)
"""


import argparse
from model import SegNet
from data import load_data, CamVidDataset 
import os
import time
import torch
import torch.nn.functional as F 
import numpy as np 
from torch.utils.data import DataLoader


# Constants
NUM_EPOCHS = 20
LEARNING_RATE = 0.05
BATCH_SIZE = 5

#GPU (argument) 
parser = argparse.ArgumentParser()
args = parser.parse_args()


# Argument 

def train(X, y):
        
    losses = []
    
    dataset = CamVidDataset(X, y)
    # base on GPU, add num_workers 
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # GPU parameters 
    CUDA = False
    GPU_ID = None

    if CUDA: 
        model = SegNet(3,2).cuda(GPU_ID)
        weights = torch.FloatTensor([0.1, 0.5]).cuda(GPU_ID)
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weights).cuda(GPU_ID)
        
    else: 
        model = SegNet(3, 2)
        weights = torch.FloatTensor([0.1, 0.5])
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)
    
            
    # Optimizer (ADAM or SGD)  
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)
    
     
    print("Start Training") 
    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()
        

        for batch in loader: 
 
            if CUDA: 
                X_batch = X_batch = torch.autograd.Variable(batch['X']).cuda(GPU_ID) 
                y_batch = torch.autograd.Variable(batch['y']).cuda(GPU_ID)
            else: 
                X_batch = torch.autograd.Variable(batch['X'])
                y_batch = torch.autograd.Variable(batch['y'])

            # softmax output
            prediction = model(X_batch)
            output = F.softmax(prediction, dim=1)
            label = torch.argmax(y_batch, dim=1) 

            
            optimizer.zero_grad()
            loss = cross_entropy(output, label)
            loss.backward()
            optimizer.step()

#             print(loss.float()) 
            loss_f += loss.float()

        delta = time.time() - t_start
        
        losses.append(loss_f)
        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))
        

if __name__ == "__main__":

    X, y = load_data(r"../raw_data/toy_dataset")
    train(X[0:10, :, :, :], y[0:10, :, :, :])
