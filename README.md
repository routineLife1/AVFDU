# Anime video frame deduplication 

# requirement(需求):

tqdm

python-opencv

numpy

skimage

warnings

os

# usage(用法)

python xxxx.py

# 参数说明(one_line.py):

path -> path to video frames(视频帧路径)

max_ssim -> deduplicate thresold using ssim(去除重复帧ssim阈值)

min_vec -> deduplicate one beta x using ssim(x = 2,3)(去除一拍二一拍三ssim阈值)

针对动漫一拍二，一拍三，以及重复帧的去除算法

static.py用于去除重复帧，start1-2.py用于去除一拍二，start1-3.py用于去除一拍三

推荐使用顺序: static.py -> start1-2.py -> start1-3.py 或者直接使用one_line.py

static.py to deduplicate static frames

start1-3.py is a demo of one beta three video frames deduplication

start1-2.py is a demo of one beta two video frames deduplication

Recommended order of using: static.py -> start1-2.py -> start1-3.py or you can direct run the one_line.py
