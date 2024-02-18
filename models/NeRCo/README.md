<div align="center">

# Cloned Implicit Neural Representation for Cooperative Low-light Image Enhancement

Cloned from: [**Implicit Neural Representation for Cooperative Low-light Image Enhancement**](https://github.com/Ysz2022/NeRCo)

Please reference the above repository for any issues with this implementation.

## Prerequisites
- Linux or macOS
- Python 3.8
- NVIDIA GPU + CUDA CuDNN

## 🔑 Setup
Type the command:
```
pip install -r requirements.txt
```

## 🧩 Download
You need **create** a directory `./saves/[YOUR-MODEL]` (e.g., `./saves/LSRW`). \
The baseline for all of our models is to test on LOL.
Please download the pre-trained model for LOL and save in `./saves/LOL`. \
- [**NeRCo trained on LOL**](https://drive.google.com/file/d/1uL4u1iXN2xoVr4Owr5uZgYY3k03nvJZ3/view?usp=sharing)


## 🚀 Quick Run
- Create directories `./dataset/testA` and `./dataset/testB`. Put your test images in both `./dataset/testA` and `./dataset/testB`. I found that images and image names must match in both testA and testB to run correctly.

- Test the model with the pre-trained weights:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot ./dataset --name LOL --preprocess=none
```
- The test results will be saved to a directory here: `./results/LOL/test_latest/images`, and will also be displayed in a html file here: `./results/LOL/test_latest/index.html`.

## 🤖 Training
Please reference main repository above for training.

## 🔎 YOLO
First download the weights file (same as LabA1) in the yolo_files folder by using the following command:
```
wget https://pjreddie.com/media/files/yolov3.weights
```
To test YOLO on output images run:
```
python yolo_img_detector.py --model LOL
```
All input images will be pulled from the  `./results/LOL/test_latest/images` folder and will be output into out_imgs/LOL.

If you find this code useful for your research, please use the following BibTeX entry.

