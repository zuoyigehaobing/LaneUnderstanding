# Face_Detection

To achieve better understanding of autonomous vehicle’s surroundings, we proposed a new pipeline by combining custom variants of SegNet for road segmentation and YOLOv1 for traffic object detection. By utilizing transfer learning and novel image augmentation that is not mentioned in the original configuration, our model obtained better performance on chosen datasets. We were able to achieve good results on both in-domain and out-domain traffic datasets captured in real life.

<img src="https://github.com/zuoyigehaobing/LaneUnderstanding/blob/main/demo/pipeline.png" width="600" height="200">

## Writeup

### For our proposal see [HERE](https://github.com/zuoyigehaobing/LaneUnderstanding/blob/main/doc/EECS504%20Project%20Proposal.pdf)

### For our project report see [HERE](https://github.com/zuoyigehaobing/LaneUnderstanding/blob/main/doc/Project%20Report.pdf)


## Model checkpoints

- SegNet: https://drive.google.com/drive/u/0/folders/1M_jH04mBN9ZuXHPCkZbCAFIRHUbnOhO7

- YOLOv1: https://drive.google.com/file/d/1SDcfLAAhujRvOCjaW9-325Ol7Lp9Vttu/view?usp=sharing

## Visual results

### On Camvid valid dataset:

<img src="https://github.com/zuoyigehaobing/LaneUnderstanding/blob/main/demo/camvid_itr2_val_15fps.gif" width="600">


### On Camvid training dataset (Note that we usd this dataset in our training):

<img src="https://github.com/zuoyigehaobing/LaneUnderstanding/blob/main/demo/camvid_itr2_train_5fps.gif" width="600">


### On KITTI 1:

<img src="https://github.com/zuoyigehaobing/LaneUnderstanding/blob/main/demo/kitti_itr2_test_3fps.gif" width="600">


### On KITTI 2:

<img src="https://github.com/zuoyigehaobing/LaneUnderstanding/blob/main/demo/kitti_itr2_train_5fps.gif" width="600">


### Contact

{Bingzhao Shan, Songlin Liu, Zihan Wang, Zuoyi Li} @ Umich


#### Some intermediate SegNet result can be found [here](https://github.com/zuoyigehaobing/LaneUnderstanding/blob/main/Segmentation/code/SegNet_on_colab.pdf)
