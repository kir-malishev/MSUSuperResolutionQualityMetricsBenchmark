import os
import json
import imageio as iio
from tqdm import tqdm
import imageio
import cv2
import time
import numpy as np
import torch
import sys
from torch.autograd import Variable                                                                                                                     
sys.path.append('utils/pieapp/model/')
from utils.pieapp.model.PieAPPv0pt1_PT import PieAPP
sys.path.append('utils/pieapp/utils')
from utils.pieapp.utils.image_utils import *


def check_yuv(path):
    return path[-3 : len(path)] == 'yuv'
def Fix(s):
    return (4 - len(s)) * '0' + s

patch_size = 64
batch_size = 1
num_patches_per_dim = 10
stride_val = 6

def get_pieapp_model(device):
    global patch_size, batch_size, num_patches_per_dim, stride_val

    if not os.path.isfile('utils/pieapp/weights/PieAPPv0.1.pth'):
        print("downloading dataset")
        os.chdir('utils/pieapp')
        os.system("bash scripts/download_PieAPPv0.1_PT_weights.sh")
        if not os.path.isfile('weights/PieAPPv0.1.pth'):
            print("PieAPPv0.1.pth not downloaded")
            sys.exit()    
            os.system("bash scripts/download_PieAPPv0.1_PT_weights.sh")
        if not os.path.isfile('weights/PieAPPv0.1.pth'):
            print("PieAPPv0.1.pth not downloaded")
            sys.exit()
        os.chdir(os.pardir)
        os.chdir(os.pardir)
    
    PieAPP_net = PieAPP(batch_size,num_patches_per_dim)
    PieAPP_net.load_state_dict(torch.load('utils/pieapp/weights/PieAPPv0.1.pth'))
    
    if 'cuda' in device:
        PieAPP_net.cuda()
        os.environ['CUDA_VISIBLE_DEVICES'] = device[4:]
    return PieAPP_net

def calc_pieapp_metric(PieAPP_net, device, frames_cnt, dist_frames_path = None, gt_frames_path = None):
    global patch_size, batch_size, num_patches_per_dim, stride_val

    score = 0.0
    full_time = 0.0

    for i in tqdm(range(frames_cnt)):
        dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
        
        gt_image = os.path.join(gt_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_gt_image = os.path.join(gt_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
        
        try:
            start = time.time()

            imagesA = np.expand_dims(cv2.imread(small_dist_image),axis =0).astype('float32')
            imagesRef = np.expand_dims(cv2.imread(small_gt_image),axis =0).astype('float32')
            _,rows,cols,ch = imagesRef.shape
            
            y_loc = np.concatenate((np.arange(0, rows - patch_size, stride_val),np.array([rows - patch_size])), axis=0)
            num_y = len(y_loc)
            x_loc = np.concatenate((np.arange(0, cols - patch_size, stride_val),np.array([cols - patch_size])), axis=0)
            num_x = len(x_loc)
            
            for x_iter in range(0, -(-num_x//num_patches)):
                for y_iter in range(0, -(-num_y//num_patches)):
                    # compute the size of the subimage
                    if (num_patches_per_dim*(x_iter + 1) >= num_x):
                        size_slice_cols = cols - x_loc[num_patches_per_dim*x_iter]
                    else:
                        size_slice_cols = x_loc[num_patches_per_dim*(x_iter + 1)] - x_loc[num_patches_per_dim*x_iter] + patch_size - stride_val
                    if (num_patches_per_dim*(y_iter + 1) >= num_y):
                        size_slice_rows = rows - y_loc[num_patches_per_dim*y_iter]
                    else:
                        size_slice_rows = y_loc[num_patches_per_dim*(y_iter + 1)] - y_loc[num_patches_per_dim*y_iter] + patch_size - stride_val
                    # obtain the subimage and samples patches
                    A_sub_im = imagesA[:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols,:]
                    ref_sub_im = imagesRef[:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols,:]
                    A_patches, ref_patches = sample_patches(A_sub_im, ref_sub_im, patch_size=64, strideval=stride_val, random_selection=False, uniform_grid_mode = 'strided')
                    num_patches_curr = A_patches.shape[0]/batch_size

                    PieAPP_net.num_patches = num_patches_curr

                    # initialize variable to be  fed to PieAPP_net
                    A_patches_var = Variable(torch.from_numpy(np.transpose(A_patches,(0,3,1,2))), requires_grad=False)
                    ref_patches_var = Variable(torch.from_numpy(np.transpose(ref_patches,(0,3,1,2))), requires_grad=False)
                    if 'cuda' in device:
                        A_patches_var = A_patches_var.cuda()
                        ref_patches_var = ref_patches_var.cuda()

                    # forward pass
                    _, PieAPP_patchwise_errors, PieAPP_patchwise_weights = PieAPP_net.compute_score(A_patches_var.float(), ref_patches_var.float())
                    curr_err = PieAPP_patchwise_errors.cpu().data.numpy()
                    curr_weights = PieAPP_patchwise_weights.cpu().data.numpy()
                    score_accum += np.sum(np.multiply(curr_err, curr_weights))
                    weight_accum += np.sum(curr_weights)
                    
            score += score_accum/weight_accum
            start = time.time() - start
        except:
            imagesA = np.expand_dims(cv2.imread(dist_image),axis =0).astype('float32')
            imagesRef = np.expand_dims(cv2.imread(gt_image),axis =0).astype('float32')
            _,rows,cols,ch = imagesRef.shape

            y_loc = np.concatenate((np.arange(0, rows - patch_size, stride_val),np.array([rows - patch_size])), axis=0)
            num_y = len(y_loc)
            x_loc = np.concatenate((np.arange(0, cols - patch_size, stride_val),np.array([cols - patch_size])), axis=0)
            num_x = len(x_loc)

            for x_iter in range(0, -(-num_x//num_patches)):
                for y_iter in range(0, -(-num_y//num_patches)):
                    # compute the size of the subimage
                    if (num_patches_per_dim*(x_iter + 1) >= num_x):
                        size_slice_cols = cols - x_loc[num_patches_per_dim*x_iter]
                    else:
                        size_slice_cols = x_loc[num_patches_per_dim*(x_iter + 1)] - x_loc[num_patches_per_dim*x_iter] + patch_size - stride_val
                    if (num_patches_per_dim*(y_iter + 1) >= num_y):
                        size_slice_rows = rows - y_loc[num_patches_per_dim*y_iter]
                    else:
                        size_slice_rows = y_loc[num_patches_per_dim*(y_iter + 1)] - y_loc[num_patches_per_dim*y_iter] + patch_size - stride_val
                    # obtain the subimage and samples patches
                    A_sub_im = imagesA[:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols,:]
                    ref_sub_im = imagesRef[:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols,:]
                    A_patches, ref_patches = sample_patches(A_sub_im, ref_sub_im, patch_size=64, strideval=stride_val, random_selection=False, uniform_grid_mode = 'strided')
                    num_patches_curr = A_patches.shape[0]/batch_size

                    PieAPP_net.num_patches = num_patches_curr

                    # initialize variable to be  fed to PieAPP_net
                    A_patches_var = Variable(torch.from_numpy(np.transpose(A_patches,(0,3,1,2))), requires_grad=False)
                    ref_patches_var = Variable(torch.from_numpy(np.transpose(ref_patches,(0,3,1,2))), requires_grad=False)
                    if 'cuda' in device:
                        A_patches_var = A_patches_var.cuda()
                        ref_patches_var = ref_patches_var.cuda()

                    # forward pass
                    _, PieAPP_patchwise_errors, PieAPP_patchwise_weights = PieAPP_net.compute_score(A_patches_var.float(), ref_patches_var.float())
                    curr_err = PieAPP_patchwise_errors.cpu().data.numpy()
                    curr_weights = PieAPP_patchwise_weights.cpu().data.numpy()
                    score_accum += np.sum(np.multiply(curr_err, curr_weights))
                    weight_accum += np.sum(curr_weights)

            score += score_accum/weight_accum
            start = time.time() - start
        
        full_time += start
    
    score /= frames_cnt
    full_time /= frames_cnt
        
    return score, full_time
