[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/low-light-image-enhancement-with-normalizing/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=low-light-image-enhancement-with-normalizing)

# Cloned Low-Light Image Enhancement with Normalizing Flow
### [Paper](https://arxiv.org/pdf/2109.05923.pdf) | [Project Page](https://wyf0912.github.io/LLFlow/)

**Low-Light Image Enhancement with Normalizing Flow**
<br>_Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-pui Chau, Alex C. Kot_<br>
In AAAI'2022

## Get Started
### Dependencies and Installation
- Python 3.8
- Pytorch 1.9

1. Clone Repo

2. Create Conda Environment
```
conda create --name LLFlow python=3.8
conda activate LLFlow
```
3. Install Dependencies
```
cd LLFlow
pip install -r ./code/requirements.txt
```
### Dataset
You can refer to the following links to download the datasets
[LOL](https://daooshee.github.io/BMVC2018website/), and
[LOL-v2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light). 
### Pretrained Model
I provide the pre-trained models with the following settings:
- A light weight model with promising performance trained on LOL [[Google drive](https://drive.google.com/file/d/1tukKu2KBZ_ohlQiLG4EKnrn1CAt_2F6G/view?usp=sharing)] with training config file `./confs/LOL_smallNet.yml`
- A standard-sized model trained on LOL [[Google drive](https://drive.google.com/file/d/1t3kASTRXbnEnCZ0EcIvGhMHYkoJ8E2C4/view?usp=sharing)] with training config file `./confs/LOL-pc.yml`.
- A standard-sized model trained on LOL-v2 [[Google drive](https://drive.google.com/file/d/1n7XwIlNr1lUxgZ9qlmFXCwzMTWSStQIW/view?usp=sharing)] with training config file `./confs/LOLv2-pc.yml`.

### Test
You can check the training log to obtain the performance of the model. You can also directly test the performance of the pre-trained model as follows

1. Modify the paths to dataset and pre-trained mode. You need to modify the following path in the config files in `./confs`
```python
#### Test Settings
dataroot_unpaired # needed for testing with unpaired data
dataroot_GT # needed for testing with paired data
dataroot_LR # needed for testing with paired data
model_path
```
2. Test the model

To test the model with paired data and obtain the evaluation results, e.g., PSNR, SSIM, and LPIPS. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and model path ```model_path``` in the config file. Then run
```bash
python test.py --opt your_config_path
# You need to specify an appropriate config file since it stores the config of the model, e.g., the number of layers.
```

To test the model with unpaired data, you need to specify the unpaired data path ```dataroot_unpaired```, and model path ```model_path``` in the config file. Then run
```bash
python test_unpaired.py --opt your_config_path -n results_folder_name
# You need to specify an appropriate config file since it stores the config of the model, e.g., the number of layers.
```
You can check the output in `../results`.
## 🤖Train
please refer to the original linked github to train the model. 

## 🔎Yolo
to implement YOLO of the results please check the YOLO README
