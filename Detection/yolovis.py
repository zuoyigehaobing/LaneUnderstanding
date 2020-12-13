import os
from os.path import isfile, join
from os import listdir
import numpy as np
import cv2
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_resnet import Yolov1
from matplotlib import pyplot as plt
from voc_classes import classes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

eval_path = "data/KiTTi_dataset/training_crop/" # "data/CamVid/train/" #
mask_path = "data/KiTTi_dataset/training_crop_segnet/"
result_path = "data/kitti_res/training_res/" # "data/camvid_res/train_res/" # "data/kitti_res/training_res/"
conf_thresh = 0.37
iou_thresh = 0.4
transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min = x1 - w1/2
    x1_max = x1 + w1/2
    y1_min = y1 - h1/2
    y1_max = y1 + h1/2
    x2_min = x2 - w2/2
    x2_max = x2 + w2/2
    y2_min = y2 - h2/2
    y2_max = y2 + h2/2

    intersec = 0
    if x1 > x2 and x1_min < x2_max:
        if y1 > y2 and y1_min < y2_max:
            intersec = (x2_max - x1_min) * (y2_max - y1_min)
        if y2 > y1 and y2_min < y1_max:
            intersec = (x2_max - x1_min) * (y1_max - y2_min)
    if x2 > x1 and x2_min < x1_max:
        if y1 > y2 and y1_min < y2_max:
            intersec = (x2_max - x1_min) * (y2_max - y1_min)
        if y2 > y1 and y2_min < y1_max:
            intersec = (x2_max - x1_min) * (y1_max - y2_min)

    union = w1*h1 + w2*h2 - intersec
    return intersec / union


def non_max_suppress(bboxes, bclassids, confidences, conf_thresh, iou_thresh):
    bboxesL = np.reshape(bboxes.detach().numpy(), (49, 4))
    bclassL = np.reshape(bclassids.detach().numpy(), (49, 1))
    confsL = np.reshape(confidences.detach().numpy(), (49, 1))
    catL = np.concatenate((bboxesL, bclassL, confsL), axis=1)
    # print(catL.shape)  # 49*6

    catL = catL[catL[:, 5] > conf_thresh]
    finL = np.array([])
    first = 1
    while len(catL) > 0:
        maxid = np.argmax(catL[:, 5])
        maxval = np.max(catL[:, 5])
        pick = catL[maxid]
        if first == 1:
            finL = np.array([pick])
            first = 0
        else:
            finL = np.append(finL, [pick], axis=0)
        catL = np.delete(catL, (maxid), axis=0)

        for i in range(len(catL)):
            score = IOU(pick[0:4], catL[i, 0:4])
            if score >= iou_thresh:
                catL[i, 5] = -1

        catL = catL[catL[:, 5] > conf_thresh]

    print("final len: ", len(finL))
    if len(finL) == 0:
        return [], [], []

    bboxesF, bclassF, confsF = np.split(finL, [4, 5], axis=1)
    # print(catL)
    # print(bboxesF.shape, bclassF.shape, confsF.shape)
    return bboxesF, bclassF, confsF


def showBox(img_path, imagename, bboxes, Confidences, classids, pix_width):
    # bboxes:[[x, y, w, h], ...]
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    image = cv2.imread(img_path)
    ih, iw, c = image.shape

    for i in range(len(bboxes)):
        x, y, w, h = bboxes[i]

        x_min = int(round(x*iw - w*iw/2))
        x_max = int(round(x*iw + w*iw/2))
        y_min = int(round(y*ih - h*ih/2))
        y_max = int(round(y*ih + h*ih/2))

        if x_min == x_max or y_min == y_max:
            continue

        if int(classids[i]) not in [1, 5, 6, 13, 14, 18]:
            print("skip, classid ", classids[i], classes[int(classids[i])])
            continue

        # color based on class name
        random.seed(classes[int(classids[i])])
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)

        # print("params: ", (x_min,y_min), (x_max,y_max), (b,g,r), pix_width)
        image = cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (b,g,r), pix_width)

        label = classes[int(classids[i])] #  + ", Confidence: " + str(Confidences[i])
        tw, th = cv2.getTextSize(label, font, font_scale, 1)[0]
        # text offset in textbox
        xo = 10
        yo = 6

        x_max_tbox = x_min + tw + xo*2
        y_max_tbox = y_min - th - yo*2
        on_top = 1
        left_to_right = 1

        # Assuming no class label would be longer than half width of image
        if x_max_tbox >= iw:
            left_to_right = -1
        if y_max_tbox <= 0:
            on_top = -1

        tbox_co = [(x_min - pix_width, y_min), (x_max_tbox - pix_width, y_max_tbox)]
        if left_to_right > 0:
            if on_top < 0:
                tbox_co[1] = (x_max_tbox - pix_width, y_min + th + yo*2)
        else:
            if on_top > 0:
                tbox_co[0] = (x_max - tw - xo*2 + pix_width, y_min)
                tbox_co[1] = (x_max + pix_width, y_max_tbox)
            else:
                tbox_co[0] = (x_max - tw - xo*2 + pix_width, y_min - th - yo*2)
                tbox_co[1] = (x_max + pix_width, y_min)

        image = cv2.rectangle(image, tbox_co[0], tbox_co[1], (b,g,r), cv2.FILLED)
        image = cv2.putText(image, label, (tbox_co[0][0] + xo, tbox_co[0][1] - yo),
                font, font_scale, (0,0,0), 1)

    # plt.imshow(image)
    # plt.show()
    cv2.imwrite(result_path + "res" + str(conf_thresh) + "_" + imagename, image)


def main():
    if torch.cuda.is_available():
        model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to("cuda")
        check_point = torch.load('resnet.tar')

    else:
        device = torch.device('cpu')
        model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
        check_point = torch.load('resnet.tar', map_location=device)

    model.load_state_dict(check_point["state_dict"])
    model.eval()

    workload = len(listdir(eval_path))
    itr = 0

    image_paths = [f for f in listdir(eval_path) if isfile(join(eval_path, f))]
    for image_path in image_paths:
        itr += 1
        print(eval_path + image_path, "\t", round(itr/workload*100, 2), "%")
        image = Image.open(eval_path + image_path).convert('RGB')
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
        # converted_preds = torch.cat((converted, bclass, bconf), dim=-1)

        # perform non maximum supression
        bboxes, bclassids, confs = non_max_suppress(converted, bclass, bconf, conf_thresh, iou_thresh)

        # draw bounding box over input image
        # image = Image.open(folder_path + image_path).convert('RGB')
        showBox(mask_path + image_path, image_path, bboxes, confs, bclassids, 3)


if __name__ == "__main__":
    main()
