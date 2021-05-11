from tqdm import tqdm
import cv2
import numpy as np
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os
warnings.filterwarnings("ignore")

i0=cv2.imread("000001027.png")
i0 = cv2.resize(i0,(256,256))
i1=cv2.imread("000001028.png")
i1 = cv2.resize(i1,(256,256))
i2=cv2.imread("000001029.png")
i2 = cv2.resize(i2,(256,256))
i3=cv2.imread("000001030.png")
i3 = cv2.resize(i3,(256,256))

def ssim(i0,i1):
    return compare_ssim(i0,i1,multichannel=True)*100

m = ssim(i1,i2)
l = ssim(i0,i1)
r = ssim(i2,i3)
print(l,m,r)