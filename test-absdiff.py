from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os

im0 = cv2.imread('D:/im-backup/000000468.png')
im1 = cv2.imread('D:/im-backup/000000469.png')
im2 = cv2.imread('D:/im-backup/000000470.png')
im3 = cv2.imread('D:/im-backup/000000471.png')

def diff(i0,i1):
    return cv2.absdiff(i0,i1).mean()

print(diff(im0,im1))
print(diff(im1,im2))
print(diff(im2,im3))