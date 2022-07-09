from tqdm import tqdm
import cv2
import warnings
import os
import numpy as np
warnings.filterwarnings("ignore")

path = r'E:\Work\experiment\AVFDU\input' # 图片路径
img_load_size = [256,256]
flow_scale_size = [32,32] # 光流计算缩放大小
max_epoch = 3 # 一直去除到一拍N，N为max_epoch（不建议超过3）

def diff(i0,i1):
    return cv2.absdiff(i0,i1).mean()

# 计算光流距离
def calc_flow_distance(i0,i1):
    prev_gray = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
        flow=None, pyr_scale=0.5, levels=1,iterations=20,
        winsize=20, poly_n=5, poly_sigma=1.1, flags=0)
    x = flow[:, :, 0]
    y = flow[:, :, 1]
    return np.linalg.norm(x)+np.linalg.norm(y)

# 预测光流距离系数
def predict_scale(i0,i1):
    w,h,_ = i0.shape
    diff = cv2.Canny(cv2.absdiff(i0,i1),100,200)
    mask = np.where(diff!=0)
    try:
        xmin = min(list(mask)[0])
    except Exception:
        xmin = 0
    try:
        xmax = max(list(mask)[0]) + 1
    except Exception:
        xmax = w
    try:
        ymin = min(list(mask)[1])
    except Exception:
        ymin = 0
    try:
        ymax = max(list(mask)[1]) + 1
    except Exception:
        ymax = h
    W = xmax - xmin
    H = ymax - ymin
    S0 = w * h
    S1 = W * H
    return -2 * (S1 / S0) + 3

# 亮度均衡化
def histeq(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    histeq = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    histeq[..., 0] = cv2.equalizeHist(histeq[..., 0])
    return cv2.cvtColor(histeq, cv2.COLOR_YUV2BGR)

import itertools
LabData = [os.path.join(path,f) for f in os.listdir(path)] #记录文件名用
frames = [histeq(cv2.resize(cv2.imread(f),(img_load_size[0],img_load_size[1]))) for f in LabData]

print('destatic abs frames and spilt scene...')
pbar = tqdm(total=len(frames))
delgen = []
lf = frames[0]
for i in range(1,len(frames)):
    f = frames[i]
    d = diff(lf,f)
    if d == 0:
        delgen.append(i)
    lf = f
    pbar.update(1)
tmp0 = LabData.copy()
tmp1 = frames.copy()
for x in delgen:
    os.remove(LabData[x])
    try:
        del tmp0[x]
        del tmp1[x]
    except Exception:
        print('err at {}',x )
LabData = tmp0
frames = tmp1

print('build one beta x frame list...')
opt = [] # 已经被标记   ，识别的帧
I0 = frames[0] # 第一帧
pbar = tqdm(total=(max_epoch-1) * len(LabData)) # 总轮数 * 数据长度
for queue_size, _ in enumerate(range(1,max_epoch), start=4):
    Icount = queue_size - 1 # 输入帧数
    Current = [] # 该轮被标记的帧   
    i = 1
    while (i < len(LabData) - Icount):
        c = [frames[p+i] for p in range(queue_size)] # 读取queue_size帧图像
        first_frame = c[0]
        last_frame = c[-1]

        count = 0
        for step in range(1,queue_size - 2):
            pos = 1
            while (pos + step <= queue_size - 2):
                m0 = c[pos]
                m1 = c[pos+step]

                # 对图象进行缩放
                width = flow_scale_size[0]
                height = flow_scale_size[1]
                first_frame = cv2.resize(first_frame,(width,height))
                last_frame = cv2.resize(last_frame,(width,height))
                m0 = cv2.resize(m0,(width,height))
                m1 = cv2.resize(m1,(width,height))

                # 计算光流距离
                value_scale = predict_scale(m0,m1)
                d0 = calc_flow_distance(first_frame,m0)
                d1 = calc_flow_distance(m0,m1) * value_scale
                d2 = calc_flow_distance(m1,last_frame)
                if d1 < d0 and d1 < d2:
                    count += 1
                pos += 1

        if count == (queue_size * (queue_size - 5) + 6) / 2:
            Current.append(i) # 加入标记序号
            i += queue_size - 3
            pbar.update(queue_size - 3)
        i += 1
        pbar.update(1)
    opted = len(opt) # 记录opt长度
    opt.extend(t + x + 1 for x, t in itertools.product(Current, range(queue_size - 3)))

    pbar.update(1) # 完成一轮

print('concat result...')
delgen=sorted(set(opt)) # 需要删除的帧

for d in delgen:
    try:
        os.remove(LabData[d])
    except Exception:
        print('pass')