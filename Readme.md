# AAIS Pipe Counting

## Description
    This code is the implementation of image processing and deep learning model to count the number of pipes in Industrial site from either RGB image or camera

## Requirements
* opencv-python
* opencv-contrib-python
* numpy
* pillow
* pytorch


## Classical image processing approach
### a) Detect and Count the pipes using Blob Detector
```
cd task1_pipe_counting \
python byBlobDetector.py
```
Ref: https://learnopencv.com/blob-detection-using-opencv-python-c/
### b) Detect and Count the pipes using masking color thresholding and Hough circle transform


    "labellingTools.py" can be used for pipes detection and labelling purposes.The file is saved to xml as the same  "labelImg" 
    

## Deep learning approach
### a) Ad-hoc object detection model
   explore yolo4, yolo5 as the backbones


### References
* Paper Ref
``` 
@inproceedings{m_Ranjan-etal-CVPR21, 
author = {Viresh Ranjan and Udbhav Sharma and Thu Nguyen and Minh Hoai},
title = {Learning To Count Everything},
year = {2021},
booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
}

@INPROCEEDINGS{cholakkal_sun2019object,
    author = {Cholakkal, Hisham and Sun, Guolei and Khan, Fahad Shahbaz and Shao, Ling},
    title = {Object Counting and Instance Segmentation with Image-level Supervision},
    booktitle = {CVPR},
    year = {2019}
}

@article{cholakkal_sun2020towards,
  title={Towards Partial Supervision for Generic Object Counting in Natural Scenes},
  author={Cholakkal, Hisham and Sun, Guolei and Khan, Salman and Khan, Fahad Shahbaz and Shao, Ling and Gool, Luc Van},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2020}
}

```
* Application demo Ref
``` 
demo: https://www.youtube.com/watch?v=4ONcSj6-S8k
```

### Tools

* Image Labeling Software
```
https://github.com/heartexlabs/labelImg
```
* circles counter (can be used for pipes counter) by hough circle transform algorithm
```
https://github.com/pwwiur/hough-counter
```

# ToDO

Since classical image processing approach yield unstatisfactory result, A tailored yolo model will be developed

### Todo

- [ ] Take pipes images samples  
- [ ] label using labelImg softwares
- [ ] Develop model based on yolov4 or yolov5 

### Done ???

- [x] Basic approach
- [x] Literature Review
