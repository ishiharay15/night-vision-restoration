<div align="center">

# Cloned Retinexformer

Cloned from: [**Retinexformer**](https://github.com/caiyuanhao1998/Retinexformer)

Please reference the above repository for any issues with this implementation.

<div align="left">


## 🔑 Setup
Create Retinexformer environment:
```
conda create -n Retinexformer python=3.7
conda activate Retinexformer
```
Install Dependencies:
```conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

## 🧩 Downloads
The baseline for all of our models is to test on LOL.
Please download the [LOLv1 dataset](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H) and place in project directory.
Then, download the pre-trained model for LOL and save in `./pretrained_weights`. 
- [**NeRCo trained on LOL**](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV)

## 🚀 Quick Run

Test the model with the pre-trained weights:
```
# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

```
The test results will be saved to the `results` directory.

For video testing, save test videos to `./Enhancement/Video/Input`.
Then test with:

```
python3 Enhancement/video_test.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1 --input_vid [inputvid.mp4]
```

## 🤖 Training
Please reference main repository above for training.
