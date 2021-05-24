# 去重
from tqdm import tqdm
import cv2
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")


path = './im' #图片路径
side_vec = 2 #双边最小运动幅度


print('loading data to ram...') #将数据载入到内存中，加速运算
LabData = [os.path.join(path,f) for f in os.listdir(path)] #记录文件名用
frames = [cv2.resize(cv2.imread(f),(256,256)) for f in LabData]

def diff(i0,i1):
    return cv2.absdiff(i0,i1).mean()

pbar = tqdm(total=len(frames))
duplicate = []
lf = frames[0]
for i in range(1,len(frames)):
    f = frames[i]
    # 两两对比，diff值<0.01，辨别为重复帧
    if diff(lf,f) < 0.01:
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

duplicate_12 = [] #用于存放表示一拍二的多组四帧列表
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
    #   中间值 - 左侧diff (i1,i2) > 最小运动幅度     右侧diff - 中间值 > 最小运动幅度
    if x1 - x2 > side_vec and x3 - x2 > side_vec:
        #duplicate_12.append([i-1,i,i+1,i+2])
        duplicate_12.append(i-1)
    I0 = I1
    pbar.update(1)
    i += 1

#去除一拍三
frames_edge = [cv2.Canny(f,100,200) for f in frames]
frames = frames_edge
duplicate = []
i0 = frames[0]
pbar = tqdm(total=len(frames))
i = 1
while i < len(LabData)-3:
    i1 = frames[i]
    i2 = frames[i+1]
    i3 = frames[i+2]
    i4 = frames[i+3]
    l = diff(i0,i1)
    m = (diff(i1,i2) + diff(i2,i3)) / 2
    r = diff(i3,i4)
    # diff(i0,i1) - 估计三帧diff值(i1,i2,i3) > side_vec AND diff(i3,i4) - 估计三帧diff值 > min_vec
    if l - m > side_vec and r - m > side_vec:
        #duplicate.append([i-1,i,i+1,i+2,i+3])
        duplicate.append(i-1)
    i0 = i1
    pbar.update(1)
    i += 1

delgen = []
for x in duplicate:
    if x in duplicate_12 or x+1 in duplicate_12 or x - 1 in duplicate_12:
        delgen.append(x)
for x in delgen:
    duplicate.remove(x)
for x in duplicate_12:
    try:
        os.remove(LabData[x+1])
    except:
        print('pass at {}'.format(x+1))
for x in duplicate:
    try:
        os.remove(LabData[x+1])
        os.remove(LabData[x+3])
    except:
        print('pass at {} {}'.format(x+1,x+3))