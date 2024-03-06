import torch
import kornia as K
import numpy as np
from functools import partial


def make_slice(img, left, right, axis):
    sl = [slice(None)] * img.ndim
    sl[axis] = slice(left, right)

    return img[tuple(sl)]


def shift1d(img, gt, shift=1, axis=0):
    if shift > 0:
        x1, x2 = shift, img.shape[axis]
        x3, x4 = 0, -shift  # gt
    elif shift == 0:
        x1, x2, x3, x4 = 0, img.shape[axis], 0, img.shape[axis]
    else:
        x1, x2 = 0, shift
        x3, x4 = -shift, img.shape[axis]

    img = make_slice(img, x1, x2, axis=axis)
    gt = make_slice(gt, x3, x4, axis=axis)

    return img, gt


def shift2d(img, gt, a=1, b=1):
    img, gt = shift1d(img, gt, a, axis=0)
    img, gt = shift1d(img, gt, b, axis=1)

    return img, gt
    

class ERQA:
    def __init__(self, shift_compensation=False, penalize_wider_edges=None, global_compensation=False, version='1.1', threshold1=100/255, threshold2=200/255, stride=None):
        """
        shift_compensation - if one-pixel shifts of edges are compensated
        """
        # Set global defaults
        self.global_compensation = global_compensation
        self.shift_compensation = shift_compensation

        # Set version defaults
        if version == '1.0':
            self.penalize_wider_edges = False
        elif version == '1.1':
            self.penalize_wider_edges = True
        else:
            raise ValueError('There is no version {} for ERQA'.format(version))

        # Override version defaults
        if penalize_wider_edges is not None:
            self.penalize_wider_edges = penalize_wider_edges

        # Set detector
        
        self.edge_detector = partial(K.filters.canny, low_threshold=threshold1, high_threshold=threshold2)

    def __call__(self, img, gt, return_maps=False):
        assert gt.shape == img.shape
        #assert gt.shape[2] == 4, 'Compared images should be in NxCxHxW format'

        if self.global_compensation:
            img, gt = self._global_compensation(img, gt)
            
        edge = self.edge_detector(img)[1]
        gt_edge = self.edge_detector(gt)[1]
        true_positive, false_negative = self.match_edges(edge, gt_edge)

        f1 = self.f1_matches(edge, true_positive, false_negative)

        false_positive = edge - true_positive
        
        if return_maps:
            return f1, true_positive, false_positive, false_negative
        else:
            return f1

    def _global_compensation(self, img, gt_img, window_range=3, metric='mse'):
        window = range(-window_range, window_range + 1)

        if metric == 'mse':
            def metric(x, y):
                return torch.mean((x.to(dtype=torch.float) - y.to(dtype=torch.float)) ** 2)
        else:
            raise ValueError('Unsupported metric "{}" for global compensation'.format(metric))

        shifts = {}
        for i in window:
            for j in window:
                shifted_img, cropped_gt_img = shift2d(img, gt_img, i, j)

                metric_value = metric(shifted_img, cropped_gt_img)
                shifts[(i, j)] = metric_value

        (i, j), _ = min(shifts.items(), key=lambda x: x[1])

        return shift2d(img, gt_img, i, j)

    def match_edges(self, edge, gt_edge):
        assert gt_edge.shape == edge.shape

        true_positive = torch.zeros_like(edge)
        false_negative = torch.clone(gt_edge)
        # Count true positive
        if self.shift_compensation:
            window_range = 1
        else:
            window_range = 0

        window = sorted(range(-window_range, window_range + 1), key=abs)  # Place zero at first place

        for i in window:
            for j in window:
                gt_ = torch.roll(false_negative, i, dims=1)
                gt_ = torch.roll(gt_, j, dims=0)

                ad = edge * gt_ * torch.logical_not(true_positive)

                true_positive = torch.logical_or(true_positive, ad)
                if self.penalize_wider_edges:
                    # Unmark already used edges
                    ad = torch.roll(ad, -j, dims=1)
                    ad = torch.roll(ad, -i, dims=0)
                    false_negative = torch.logical_and(false_negative, torch.logical_not(ad))

        if not self.penalize_wider_edges:
            false_negative = gt_edge * torch.logical_not(true_positive)

        assert not torch.logical_and(true_positive, false_negative).any()

        return true_positive.to(dtype=torch.uint8), false_negative.to(dtype=torch.uint8)

    def f1_matches(self, edge, true_positive, false_negative):
        tp = torch.sum(true_positive, dim=(1, 2, 3))
        fp = torch.sum(edge, dim=(1, 2, 3)) - tp
        fn = torch.sum(false_negative, dim=(1, 2, 3))
        
        prec = torch.div(tp,  (tp + fp))
        recall = torch.div(tp, (tp + fn))
        f1 = torch.div(2 * prec * recall, (prec + recall))
        f1[tp + fp == 0] = 0
        f1[tp == 0] = 0
        f1[tp + fn == 0] = 1
        return f1
