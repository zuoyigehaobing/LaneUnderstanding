import os
from os.path import isfile, join
from os import listdir
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_resnet import Yolov1


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
folder_path = "data/images/"
transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    check_point = torch.load('resnet.tar')
    model.load_state_dict(check_point["state_dict"])
    model.eval()

    image_paths = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for image_path in image_paths:
        image = Image.open(folder_path + image_path).convert('RGB')
        image = transforms(image)
        image = image.unsqueeze(0).to(DEVICE)
        # last dimension 0-20 class probabilities, 20-25 [confidence1,x1,y1,w1,h1], 25-30...
        out = model(image).reshape(7,7,30)
        # convert w.r.t per cell to w.r.t image
        c1 = out[..., 20]
        c2 = out[..., 25]
        b1 = out[..., 21:25]
        b2 = out[..., 26:30]
        scores = torch.cat((c1.unsqueeze(0),c2.unsqueeze(0)),dim=0)
        best = b1*(1-scores.argmax(0).unsqueeze(-1)) + b2*scores.argmax(0).unsqueeze(-1)
        idx = torch.arange(7).repeat(7,1).unsqueeze(-1).to(DEVICE)
        x = 1/7*(best[...,:1]+idx)
        y = 1/7*(best[...,1:2]+idx.permute(1,0,2))
        wh = 1/7*best[...,2:4]
        converted = torch.cat((x,y,wh),dim=-1)
        bclass = out[..., :20].argmax(-1).unsqueeze(-1)
        bconf = torch.max(c1,c2).unsqueeze(-1)
        # 7x7x6, [...,0:4]->xywh [...,4]class [...,5]confidence
        converted_preds = torch.cat((converted, bclass, bconf), dim=-1)
        # perform non maximum supression
        pass
        # draw bounding box over input image
        image = Image.open(folder_path + image_path).convert('RGB')
        pass


if __name__ == "__main__":
    main()
