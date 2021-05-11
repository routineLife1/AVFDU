from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os

im0 = cv2.imread('D:/im-backup/000000367.png')
im1 = cv2.imread('D:/im-backup/000000368.png')

def ssim(i0,i1):
    return compare_ssim(i0,i1,multichannel=True)*100

print(ssim(im0,im1))