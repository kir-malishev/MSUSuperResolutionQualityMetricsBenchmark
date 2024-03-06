import os
import json
import imageio as iio
from tqdm import tqdm
import imageio
import cv2
import time
import numpy as np
import torch
from PIL import Image
from utils.vsfa.inference_model import VSFA, get_features
from torchvision import transforms, models

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def check_yuv(path):
    return path[-3 : len(path)] == 'yuv'
def Fix(s):
    return (4 - len(s)) * '0' + s

def get_vsfa_model(device):
    device = torch.device(device)
    
    model = VSFA()
    model.load_state_dict(torch.load('utils/vsfa/models/VSFA.pt'))
    model.to(device)

    return model

def calc_vsfa_metric(model, device, frames_cnt, dist_frames_path = None, gt_frames_path = None):
    global transform
    score = 0.0
    full_time = 0.0

    device = torch.device(device)

    video_length = frames_cnt
    video_height, video_width, video_channel = iio.imread(os.path.join(dist_frames_path, 'frame0001.bmp')).shape
    small_video_height, small_video_width, _ = iio.imread(os.path.join(dist_frames_path, 'smallframe0001.bmp')).shape

    transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
    small_transformed_video = torch.zeros([video_length, video_channel, small_video_height, small_video_width])

    for frame_idx in range(video_length):
        dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(frame_idx + 1)) + '.bmp')
        small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(frame_idx + 1)) + '.bmp')

        frame = Image.open(dist_image)
        small_frame = Image.open(small_dist_image)

        frame = transform(frame)
        small_frame = transform(small_frame)

        transformed_video[frame_idx] = frame
        small_transformed_video[frame_idx] = small_frame
    
    try:
        full_time = time.time()

        features = get_features(small_transformed_video, frame_batch_size=1, device=device)
        features = torch.unsqueeze(features, 0)  # batch size 1

        with torch.no_grad():
            input_length = features.shape[1] * torch.ones(1, 1)
            outputs = model(features, input_length)
            y_pred = outputs[0][0].to('cpu').numpy()

        full_time = time.time() - full_time
    except:
        full_time = time.time()

        features = get_features(transformed_video, frame_batch_size=1, device=device)
        features = torch.unsqueeze(features, 0)  # batch size 1

        with torch.no_grad():
            input_length = features.shape[1] * torch.ones(1, 1)
            outputs = model(features, input_length)
            y_pred = outputs[0][0].to('cpu').numpy()

        full_time = time.time() - full_time

    score = y_pred
    full_time /= frames_cnt
        
    return score, full_time
