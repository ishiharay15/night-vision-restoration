# importing required libraries
from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import time

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse



parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument(
    '--opt', type=str, default='Options/RetinexFormer_SDSD_indoor.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='SDSD_indoor', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--input_vid', default='./Enhancement/Video/pixel3.mp4',
                    type=str, help='Path of input video')

args = parser.parse_args()

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

# 加载模型
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 生成输出结果的文件
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
# stx()
os.makedirs(result_dir, exist_ok=True)

psnr = []
ssim = []

video_path = os.path.join('Enhancement/Input/', args.input_vid)
input_dir, input_filename = os.path.split(args.input_vid)
input_filename_without_ext = os.path.splitext(input_filename)[0]
output_path = os.path.join('Enhancement/Results/', input_dir, input_filename_without_ext + '_results.mp4')
#output_path = './Enhancement/Results/nex1_result.mp4'


# opening video capture stream
vcap = cv2.VideoCapture(video_path)
fps = vcap.get(cv2.CAP_PROP_FPS)
vwidth = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
vheight = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
print(vwidth, vheight, length)
#vout = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (vwidth, vheight))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vout = cv2.VideoWriter(output_path, fourcc, fps, (vwidth, vheight))

# processing frames in input stream
num_frames_processed = 0 
fps_arr = []
start = time.time()


while vcap.isOpened():
    ret, frame = vcap.read()
    if num_frames_processed % 10 == 0:
        print(num_frames_processed)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #img = np.float32(frame/255)
    start_time = time.time()
    with torch.inference_mode():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(frame) / 255

        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 4
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * \
            factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model_restoration(input_)
        
        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        cast_restored = np.uint8(restored * 255)
        
    num_frames_processed += 1    # displaying frame 
    vout.write(cast_restored)
    #fps_arr.append(time.time - start_time)
    #cv2.imshow('frame' , cast_restored)
#print( "FPS: ", sum(fps_arr) / num_frames_processed)



# printing time elapsed and fps 
end = time.time()
elapsed = end-start
fps = num_frames_processed/elapsed 
print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))# releasing input stream , closing all windows 
vcap.release()
cv2.destroyAllWindows()