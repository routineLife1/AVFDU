# 去重
from tqdm import tqdm
import cv2
import numpy as np
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os
warnings.filterwarnings("ignore")


path = 'D:/im' #图片路径


print('loading data to ram...') #将数据载入到内存中，加速运算
LabData = [os.path.join(path,f) for f in os.listdir(path)] #记录文件名用
frames = [cv2.Canny(cv2.resize(cv2.imread(f,0),(256,256)),100,200) for f in LabData]

def diff(i0,i1):
    return cv2.absdiff(i0,i1).mean()

pbar = tqdm(total=len(frames))
duplicate = []
lf = frames[0]
for i in range(1,len(frames)):
    f = frames[i]
    if diff(lf,f) == 0:
        duplicate.append(i)
    lf = f
    pbar.update(1)
for x in duplicate:
    try:
        del frames[x]
        os.remove(LabData[x])
        del LabData[x]
    except:
        print('pass at {}'.format(x))

#去除一拍二

duplicate = [] #用于存放表示一拍二的多组四帧列表
I0 = frames[0]
pbar = tqdm(total=len(frames))
i = 1
while i < len(LabData)-2:
    #i0,i1,i2,i3为输入帧
    I1 = frames[i]
    I2 = frames[i+1]
    I3 = frames[i+2]
    #   i0,i1  i1,i2   i2,i3   分别对比的到diff值，i1,i2最为一个整体
    x1 = diff(I0,I1)
    x2 = diff(I1,I2)
    x3 = diff(I2,I3)
    #   左侧diff - 中间值(i1,i2) > 最小运动幅度     中间值 - 右侧diff > 最小运动幅度
    if x2 < x1 and x2 < x3:
        duplicate.append([i-1,i,i+1,i+2])
        I0 = I3
        i += 2
        pbar.update(2)
    else:
        I0 = I1
    pbar.update(1)
    i += 1 
for x in duplicate:
    try:
        del frames[x[1]]
        os.remove(LabData[x[1]]) #i0,i1,i2,i3 这里移除的是i1帧 （也可以选择i2帧）
        del LabData[x[1]]
    except:
        print('pass at {}'.format(x[1]))

#去除一拍三

#使用五帧
duplicate = []
i0 = frames[0]
pbar = tqdm(total=len(frames))
i= 1
while i < len(LabData)-3:
    i1 = frames[i]
    i2 = frames[i+1]
    i3 = frames[i+2]
    i4 = frames[i+3]
    m = (diff(i1,i2) + diff(i2,i3)) / 2
    l = diff(i0,i1)
    r = diff(i3,i4)
    # 估计三帧diff值(i1,i2,i3) - diff(i0,i1) > min_vec AND 估计三帧diff值 - diff(i3,i4) > min_vec
    if m - l < 0 and m - r < 0:
        duplicate.append([i-1,i,i+1,i+2,i+3])
        I0 = I3
        i += 3
        pbar.update(3)
    else:
        i0 = i1
    pbar.update(1)
    i += 1
for x in duplicate:
    try:
        os.remove(LabData[x[1]]) #这里选择移除旁边两帧，具体可以自己选择
        os.remove(LabData[x[3]])
    except:
        print('pass at {} {}'.format(x[1],x[3]))


