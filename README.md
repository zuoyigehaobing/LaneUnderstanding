# Lane Understanding and object detection

## TODO

### General tasks

- Zihan, Shan -> Data Augmentation.  [Offline: Rotation, Translation,  distortion] [Online: Noise/Blur, Intensity, local distortion] [image size : x, y as variables]  (pending)

### Detection Module [Stage 1]

- Image size —> 224 x 224 [may change]  
- Zihan -> Provide YOLO Input format [(label, x, y, h , w), (label, x, y, h , w), ]  (pending)
- Shan -> Data preprocessing without augmentation  (pending)
- Zuoyi, Zihan ->  Evaluation: Accuracy, FPR, Localization Error, Bounding box overlay.  (pending)


### Segmentation Module [Stage 1]

- Image Size: 224 x 224 [may change]
- Shan -> Data preprocessing [Pytorch -> Keras]  (pending)
- -> Data augmentation [same as YOLO]  (pending)
- -> Model definition, Loss, training, training plotting(*)  (pending)
		Songlin -> Loss [Pytorch, Keras]  (pending)
		Shan -> Model definition  (pending)
		Shan, Songling -> training  (pending)
- Songlin, Shan -> Evaluation metrics and visualization  (pending)
