import cv2
import data
import numpy as np
import torch
import utils
from datetime import datetime

SELECTED_MODEL = r'../check_points/Epoch49_loss2.1120_trainacc97.727_valacc97.859.pth'
IMGDATA_DIR = r'../CamVid/test'
SAVE_PATH = r'../videos/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.mp4'

# load the dataset
# X_demo, y_demo = data.load_data(IMGDATA_DIR)

# load the model
# BATCH_SIZE = 8
# DEVICE = "cpu"
# model = utils.load_checkpoints(SELECTED_MODEL)


print("debug")

vidcap = cv2.VideoCapture('../videos/0006R0.MXF.mp4')

# writing video
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(SAVE_PATH, fourcc, 5.0, (224, 224))

counter = 0
while counter < X_demo.shape[0]:

    frame = X_demo[counter, :]

    # shift the axis
    frame = np.moveaxis(frame.numpy(), 0, -1).astype(np.uint8)

    # write to the video
    # out.write(frame)

    cv2.imshow('Frame', frame)

    # Press S on keyboard
    # to stop the process
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

    counter += 1


out.release()

# Closes all the frames
cv2.destroyAllWindows()


