#去除一拍三

from tqdm import tqdm
import cv2
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

path = './im' #图片路径
log = open('1-3.txt','w')
side_vec = 2

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
    l = diff(i0,i1)
    m = (diff(i1,i2) + diff(i2,i3)) / 2
    r = diff(i3,i4)
    # 估计三帧diff值(i1,i2,i3) - diff(i0,i1) > min_vec AND 估计三帧diff值 - diff(i3,i4) > min_vec
    if l - m > side_vec and r - m > side_vec:
        #duplicate.append([LabData[i-1],LabData[i],LabData[i+1],LabData[i+2],LabData[i+3]])
        #duplicate.append([LabData[i-1],l,m,r])
        duplicate.append(i)
    i0 = i1
    pbar.update(1)
    i += 1
for x in duplicate:
    print(x,file=log)
