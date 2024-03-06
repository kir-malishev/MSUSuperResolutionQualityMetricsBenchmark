import os
import json
import imageio as iio
from tqdm import tqdm
import imageio
import cv2
import time
import numpy as np
import torch
from utils.nima.inference_model import InferenceModel

def check_yuv(path):
    return path[-3 : len(path)] == 'yuv'
def Fix(s):
    return (4 - len(s)) * '0' + s

def get_nima_model(device):
    return InferenceModel(device)

def calc_nima_metric(model, frames_cnt, dist_frames_path = None, gt_frames_path = None):
    score = 0.0
    full_time = 0.0

    for i in tqdm(range(frames_cnt)):
        dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
            
        try:
            start_time = time.time()
            score += model.predict(Image.open(small_dist_image))[0]
            start_time = time.time() - start_time
            full_time += start_time
        except:
            start_time = time.time()
            score += model.predict(Image.open(dist_image))[0]
            start_time = time.time() - start_time
            full_time += start_time
        
    score /= frames_cnt
    full_time /= frames_cnt
        
    return score, full_time
