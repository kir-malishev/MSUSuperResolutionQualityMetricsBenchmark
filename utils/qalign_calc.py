import os
import json
from moviepy.editor import *
import imageio as iio
import numpy as np
from tqdm import tqdm
import torch
import time
from PIL import Image
from transformers import AutoModelForCausalLM

def check_yuv(path):
    return path[-3 : len(path)] == 'yuv'
def Fix(s):
    return (4 - len(s)) * '0' + s

def load_video(video_file):
    from decord import VideoReader
    vr = VideoReader(video_file)

    # Get video frame rate
    fps = vr.get_avg_fps()

    # Calculate frame indices for 1fps
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]

def get_qalign_model(metric_name, device):
    return AutoModelForCausalLM.from_pretrained("q-future/one-align", trust_remote_code=True, torch_dtype=torch.float16, device_map=device if device else 'cpu')
        
def calc_qalign_metric(model, metric_type, frames_cnt, dist_frames_path = None, gt_frames_path = None):
    if metric_type == 'vqa':
        video_arr = [[]]
        small_video_arr = [[]]
        
        for i in tqdm(range(frames_cnt)):
            dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
            small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
            
            small_video_arr[0].append(Image.fromarray(np.array(iio.imread(small_dist_image))))
            video_arr[0].append(Image.fromarray(np.array(iio.imread(dist_image))))
        
        score = 0.0
        
        try:
            full_time = time.time()
            score = model.score(small_video_arr, task_='quality', input_='video')
        except:
            full_time = time.time()
            score = model.score(video_arr, task_='quality', input_='video')
        
        full_time = (time.time() - full_time) / frames_cnt
        
        return score[0].item(), full_time
    else:
        score = 0.0
        full_time = 0.0
        
        task = 'quality' if metric_type == 'iqa' else 'aesthetics'
        
        for i in tqdm(range(frames_cnt)):
            dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
            small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
            
            try:
                start_time = time.time()
                score += model.score([Image.open(small_dist_image)], task_=task, input_='image')
                start_time = time.time() - start_time
                full_time += start_time
            except:
                start_time = time.time()
                score += model.score([Image.open(dist_image)], task_=task, input_='image')
                start_time = time.time() - start_time
                full_time += start_time
        
        score /= frames_cnt
        full_time /= frames_cnt
        
        return score[0].item(), full_time
