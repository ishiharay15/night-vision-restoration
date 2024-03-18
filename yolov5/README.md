<div align="center">

# YOLOv5 Implementation on Enhanced Images

<div align="left">

## 🔎 Setup:
Install all dependencies:
```
pip install -r requirements.txt  # install dependencies
```
Then, download the YOLOv5s weights file and add to yolo_files folder by using the following command:
```
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```
To test YOLO on output images run:
```
python detect.py --weights yolov5s.pt --source 0                                                # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/LNwODJXcvt4'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

#Example:
--weights yolov5s.pt --source ../NeRCo/results/LOL/test_latest/images/
```
For YOLO with detection summary:
```
python detect_with_summary.py --weights yolov5s.pt --source ../
```
All input images will be pulled from the input folder and will be output into [Model]/out_imgs.

For full documentation and usage, check out the [YOLOv5 repository](https://github.com/ultralytics/yolov5?tab=readme-ov-file) and [original documentation](https://docs.ultralytics.com/yolov5/quickstart_tutorial/#inference-with-pytorch-hub).