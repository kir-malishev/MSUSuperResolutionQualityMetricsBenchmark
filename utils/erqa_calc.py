import os
import json
import imageio as iio
from tqdm import tqdm
import imageio
import cv2
import time
import numpy as np
import torch
import erqa
import erqa2.erqa as erqa2

def Fix(s):
    return (4 - len(s)) * '0' + s

def get_erqa_model(version='1.0'):
    if version == '1.0':
        return erqa.ERQA()
    elif version == '1.1':
        return erqa.ERQA(version = '1.1')
    else:
        return erqa2.ERQA()

def calc_erqa_metric(model, frames_cnt, dist_frames_path = None, gt_frames_path = None):
    score = 0.0
    full_time = 0.0

    for i in tqdm(range(frames_cnt)):
        dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
        
        gt_image = os.path.join(gt_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_gt_image = os.path.join(gt_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')

        try:
            start_time = time.time()
            score += model(iio.imread(small_dist_image), iio.imread(small_gt_image))
            start_time = time.time() - start_time
            full_time += start_time
        except:
            start_time = time.time()
            score += model(iio.imread(dist_image), iio.imread(gt_image))
            start_time = time.time() - start_time
            full_time += start_time
        
    score /= frames_cnt
    full_time /= frames_cnt
        
    return score, full_time
