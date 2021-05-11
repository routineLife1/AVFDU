#去除一拍三

from tqdm import tqdm
import cv2
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

path = 'D:/im-1-3' #图片路径

LabData = [os.path.join(path,f) for f in os.listdir(path)]
frames = [cv2.Canny(cv2.resize(cv2.imread(f,0),(256,256)),100,200) for f in LabData]

def diff(i0,i1):
    return cv2.absdiff(i0,i1).mean()

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
        duplicate.append([LabData[i-1],LabData[i],LabData[i+1],LabData[i+2],LabData[i+3]])
        I0 = I3
        i += 3
        pbar.update(3)
    else:
        i0 = i1
    pbar.update(1)
    i += 1
for x in duplicate:
    try:
        os.remove(x[1]) #这里选择移除前两帧，具体可以自己选择
        os.remove(x[2])
    except:
        print('pass')
