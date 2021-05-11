import cv2
import numpy as np
from skimage import measure
import warnings
warnings.filterwarnings("ignore")

i0=cv2.imread("i0.jpg")
#i0 = cv2.resize(i0,(256,256))
i0_=i0.mean(2).astype("uint8")
i1=cv2.imread("i1.jpg")
#i1 = cv2.resize(i1,(256,256))
i1_=i1.mean(2).astype("uint8")
i2=cv2.imread("i2.jpg")
#i2 = cv2.resize(i2,(256,256))
i2_=i2.mean(2).astype("uint8")
i0_=cv2.medianBlur(i0_,3)
i1_=cv2.medianBlur(i1_,3)
i2_=cv2.medianBlur(i2_,3)
mask=np.stack([i0_,i1_,i2_]).var(0)
# print(var.shape)
# print(var.max())
# print(var.min())
# print(var.mean())


max_value = np.max(mask)
#print(max_value)
ret, mask = cv2.threshold(mask, 0.005*max_value, max_value, cv2.THRESH_BINARY)
#cv2.imwrite("mask1.png",mask)

from skimage import measure

def mask_denoise(mask):
    labels = measure.label(mask, connectivity=2)  # 8连通区域标记
    # 筛选连通区域大于阈值的
    properties = measure.regionprops(labels)
    return np.in1d(labels, [0]).reshape(labels.shape)
    # valid_label = set()
    # for prop in properties:
    #     print(prop.area)
    #     if prop.area > 337651:
    #         print(prop.label)
    #         valid_label.add(prop.label)
    # return np.in1d(labels, list(valid_label)).reshape(labels.shape)

mask=mask_denoise(mask)
#cv2.imwrite("mask2.png",mask.astype("uint8")*255)

i0[mask<0.5]=255
i1[mask<0.5]=255
i2[mask<0.5]=255
x1 = measure.compare_ssim(i0,i1,multichannel=True)
x2 = measure.compare_ssim(i1,i2,multichannel=True)
x3 = measure.compare_ssim(i0,i2,multichannel=True)
print(x1,x2,x3)
cv2.imwrite('01.png',i0)
cv2.imwrite('02.png',i1)
cv2.imwrite('03.png',i2)
#print(abs(x1-x2))