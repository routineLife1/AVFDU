import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import warnings
warnings.filterwarnings("ignore")

double_vec_min_diff = 0.05

i0 = cv2.imread("test4.png")
i0 = cv2.resize(i0,(256,256))
i1 = cv2.imread("test5.png")
i1 = cv2.resize(i1,(256,256))
i2 = cv2.imread("test6.png")
i2 = cv2.resize(i2,(256,256))
i3 = cv2.imread("test7.png")
i3 = cv2.resize(i3,(256,256))

print(compare_ssim(i0,i1,multichannel=True))
print(compare_ssim(i1,i2,multichannel=True))
print(compare_ssim(i2,i3,multichannel=True))