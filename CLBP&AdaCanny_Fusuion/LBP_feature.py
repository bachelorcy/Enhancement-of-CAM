from skimage.feature import local_binary_pattern
import os
import cv2
import numpy as np
from datetime import datetime

def generate_lbp_features(image, lbp_params):
    radius = lbp_params['radius']
    n_points = lbp_params['n_points']
    method = lbp_params['method']
    lbp_image = local_binary_pattern(image, n_points, radius, method).astype(np.uint8)
    return lbp_image


def fuse_rgb_lbp(rgb_image, lbp_image, fusion_weights):
    # 确保两个图像的尺寸和通道数相同
    if rgb_image.shape != lbp_image.shape:
        raise ValueError("RGB and LBP images must have the same dimensions and channels.")

    fused = cv2.addWeighted(rgb_image, 1 - fusion_weights[0], lbp_image, fusion_weights[0], 0)
    return fused