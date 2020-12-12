import torchvision
import torch
import torch.nn as nn


class Yolov1(nn.Module):
    def __init__(self, **kwargs):
        super(Yolov1, self).__init__()
        self.darknet = torchvision.models.resnet152(pretrained=False)
        features = list(self.darknet.fc.children())[:-1]
        self.darknet.fc = nn.Sequential(*features)
        self.fcs = nn.Sequential(nn.Linear(2048, 7 * 7 * (20 + 2 * 5)))

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
