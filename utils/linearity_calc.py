import os
import json
import imageio as iio
from tqdm import tqdm
import imageio
import cv2
import time
import numpy as np
import torch
import gdown

from utils.linearity.inference_model import IQAModel

def Fix(s):
    return (4 - len(s)) * '0' + s

k = None
b = None

def get_linearity_model(device):
    global k, b
    device = torch.device(device)

    if not os.path.exists('utils/linearity/models/p1q2.pth'):
        if not os.path.exists('utils/linearity/models/'):
            os.makedirs('utils/linearity/models/')
        url = 'https://drive.google.com/uc?id=1HFyhei4D5Qd-PU3eubFt7lDXC6hzLg-5'
        output = 'utils/linearity/models/p1q2.pth'
        gdown.download(url, output)
    if not os.path.exists('utils/linearity/models/p1q2.pth'):
        print("Weights were not downloaded")
        raise ValueError

    model = IQAModel().to(device)
    checkpoint = torch.load('utils/linearity/models/p1q2.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    k = checkpoint['k']
    b = checkpoint['b']
    return model

def calc_linearity_metric(model, device, frames_cnt, dist_frames_path = None, gt_frames_path = None):
    global k, b

    score = 0.0
    full_time = 0.0

    device = torch.device(device)
    
    model.eval()

    for i in tqdm(range(frames_cnt)):
        dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
            
        try:
            start_time = time.time()
            
            im = to_tensor(iio.imread(small_dist_image)).to(device)
            im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            with torch.no_grad():
                y = model(im.unsqueeze(0))
            score += y[-1].item() * k[-1] + b[-1]
            
            start_time = time.time() - start_time
            full_time += start_time
        except:
            start_time = time.time()
            
            im = to_tensor(iio.imread(dist_image)).to(device)
            im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            with torch.no_grad():
                y = model(im.unsqueeze(0))
            score += y[-1].item() * k[-1] + b[-1]
            
            start_time = time.time() - start_time
            full_time += start_time
        
    score /= frames_cnt
    full_time /= frames_cnt
        
    return score, full_time
