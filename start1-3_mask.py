#去除一拍三

from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os
from skimage import measure
warnings.filterwarnings("ignore")

min_vec = 2 
path = 'D:/im' #图片路径

LabData = [os.path.join(path,f) for f in os.listdir(path)]
frames = [cv2.resize(cv2.imread(f),(256,256)) for f in LabData]

def mask_denoise(mask):
    labels = measure.label(mask, connectivity=2)  # 8连通区域标记
    properties = measure.regionprops(labels)
    return np.in1d(labels, [0]).reshape(labels.shape)

def ssim(i0,i1):
    return compare_ssim(i0,i1,multichannel=True)*100

#使用五帧
def compare(i0,i1,i2,i3,i4):
    i1_=i1.mean(2).astype("uint8")
    i2_=i2.mean(2).astype("uint8")
    i3_=i3.mean(2).astype("uint8")
    i1_=cv2.medianBlur(i1_,3)
    i2_=cv2.medianBlur(i2_,3)
    i3_=cv2.medianBlur(i3_,3)
    #中间三帧来做mask
    mask=np.stack([i1_,i2_,i3_]).var(0)
    max_value = np.max(mask)
    ret, mask = cv2.threshold(mask, 0.005*max_value, max_value, cv2.THRESH_BINARY)
    mask=mask_denoise(mask)
    i0[mask<0.5]=255 
    i1[mask<0.5]=255
    i2[mask<0.5]=255
    i3[mask<0.5]=255
    i4[mask<0.5]=255
    #返回处理后的5帧
    return [i0,i1,i2,i3,i4]

duplicate = []
i0 = frames[0]
pbar = tqdm(total=len(frames))
for i in range(1,len(LabData)-3):
    i1 = frames[i]
    i2 = frames[i+1]
    i3 = frames[i+2]
    i4 = frames[i+3]
    c = compare(i0,i1,i2,i3,i4)
    # (ssim(i1,i2) + ssim(i2,i3)) / 2
    m = (ssim(c[1],c[2]) + ssim(c[2],c[3])) / 2
    # ssim(i0,i1)
    l = ssim(c[0],c[1])
    # ssim(i3,i4)
    r = ssim(c[3],c[4])
    # 估计三帧ssim值(i1,i2,i3) - ssim(i0,i1) > min_vec AND 估计三帧ssim值 - ssim(i3,i4) > min_vec
    if m - l > min_vec and m - r > min_vec:
        duplicate.append([LabData[i-1],LabData[i],LabData[i+1],LabData[i+2],LabData[i+3]])
    pbar.update(1)
    i0 = i1
for x in duplicate:
    try:
        os.remove(x[1]) #这里选择移除前两帧，具体可以自己选择
        os.remove(x[2])
    except:
        print('pass')
