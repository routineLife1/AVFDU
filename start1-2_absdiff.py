#去除一拍二

from tqdm import tqdm
import cv2
import numpy as np
import warnings
import os
from skimage import measure
warnings.filterwarnings("ignore")

path = './im' #图片路径
log = open('1-2.txt','w')
side_vec = 2

print('loading data to ram...') #将数据载入到内存中，加速运算
LabData = [os.path.join(path,f) for f in os.listdir(path)] #记录文件名用
frames = [cv2.resize(cv2.imread(f,0),(256,256)) for f in LabData]

def diff(i0,i1):
    return cv2.absdiff(i0,i1).mean()

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
    if x1 - x2 > side_vec and x3 - x2 > side_vec:
        #duplicate.append([LabData[i-1],LabData[i],LabData[i+1],LabData[i+2]]) # i-1,i,i+1,i+2分别为i0,i1,i2,i3
        #duplicate.append([LabData[i-1],x1,x2,x3]) # i-1,i,i+1,i+2分别为i0,i1,i2,i3
        duplicate.append(i)
    I0 = I1
    pbar.update(1)
    i += 1 
for x in duplicate:
    print(x,file=log)
    #os.remove(x[1])