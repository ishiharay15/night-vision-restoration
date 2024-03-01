<div align="center">

# Cloned Low-Light Image Enhancement with Multi-stage Residue Quantization and Brightness-aware Attention

Cloned from: [**Low-Light Image Enhancement with Multi-stage Residue Quantization and Brightness-aware Attention**](https://github.com/LiuYunlong99/RQ-LLIE)

Please reference the above repository for any issues with this implementation.

<div align="left">

## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN
- PyTorch 1.8.0.

## 🔑 Setup
1. Clone the repository and set up environment:
```
git clone https://github.com/LiuYunlong99/RQ-LLIE.git
conda create --name [RQ or RQ-LLIE] python=3.9.18
conda activate RQ
```

## 🧩 Download
Create directory `./dataset`.\
The baseline for all of our models is to test on LOLv1, but model can be tested on both LOLv1 and LOLv2 (real and synthetic).\
Download the following datasets:\
LOLv1: [[Google Drive]](https://drive.google.com/file/d/1XqnxVcvTxr11qSOy4_wVhEjIMdAGx88t/view?usp=drive_link)

LOLv2: [[Google Drive]](https://drive.google.com/file/d/1iYvbYTNnFGU3tKhNuS8MNE2eXSXkSvz6/view?usp=drive_link)

Create directory `./pretrained_models`.\
Please download the folder of pre-trained model for LOLv1, LOLv2 real and synthetic and save in `./dataset`.\
[**RQ-LLIE trained on LOL**](https://drive.google.com/drive/folders/1mFBjwejx1qlvILfiyzl1MQb4RjKAqyhx?usp=drive_link)

## 📋 Options
You must specify path for datasets for **dataroot_GT** and **dataroot_LQ** in the following files found in `./options/[test or train]`:\
- LOLv1.yml
- LOLv2_real.yml
- LOLv2_synthetic.yml
**Example:**
```
dataroot_GT: dataset/LOLv1/eval15/high
dataroot_LQ: dataset/LOLv1/eval15/low
```
In `options.py` specify CUDA device. Default is set to:
```
CUDA_VISIBLE_DEVICES=0
```

## 🚀 Quick Run
- Test with the pre-trained model:
```
# LOLv1
python test_LOLv1_v2_real.py -opt options/test/LOLv1.yml

# LOLv2-Real
python test_LOLv1_v2_real.py -opt options/test/LOLv2_real.yml

# LOLv2-Synthetic
python test_LOLv2_synthetic.py -opt options/test/LOLv2_synthetic.yml
```
- The test results will be saved to a directory here: `./results/[LOLv--]/images`.

## 🤖 Training
Please reference main repository above for training.

## 🔎 YOLO
For YOLO implementation of enhancement results, please check the README in YOLO.
