#去除一拍二

from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os
from skimage import measure
warnings.filterwarnings("ignore")


ssim_value = 99.5
path = 'D:/im' #图片路径

print('loading data to ram...') #将数据载入到内存中，加速运算
LabData = [os.path.join(path,f) for f in os.listdir(path)] #记录文件名用
frames = [cv2.resize(cv2.imread(f),(256,256)) for f in LabData]
#frames = [cv2.Canny(cv2.resize(cv2.imread(f,0),(256,256)),100,200) for f in LabData]

def ssim(i0,i1):
    return compare_ssim(i0,i1,multichannel=True)*100

duplicate = [] #用于存放表示一拍二的多组四帧列表
I0 = frames[0] 
pbar = tqdm(total=len(frames))
for i in range(1,len(LabData)):
    I1 = frames[i]
    if ssim(I0,I1) > ssim_value:
        duplicate.append(LabData[i])
    pbar.update(1)
    I0 = I1
for x in duplicate:
    os.remove(x)