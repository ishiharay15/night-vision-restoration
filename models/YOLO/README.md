<div align="center">

# YOLO Implementation on Enhanced Images

<div align="left">

## 🔎 Running YOLO
First download the weights file (same as LabA1) in the yolo_files folder by using the following command:
```
wget https://pjreddie.com/media/files/yolov3.weights
```
To test YOLO on output images run:
```
python yolo_img_detector.py --input /[location of output images folder]

#Example:
python yolo_img_detector.py --input ../NeRCo/results/LOL/test_latest/images/
```
All input images will be pulled from the input folder and will be output into [Model]/out_imgs.