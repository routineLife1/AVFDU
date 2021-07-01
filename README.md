# 动漫一拍二，一拍三识别算法

absdiff.py和canny_absdiff.py属于早期方法

flow.py在无遮挡的条件下，人物一拍N部分识别准确度较高

# 环境需求:

python 3.7

tqdm

python-opencv

numpy

skimage

warnings

# 算法概述:

吞入2+2帧，设为i0,i1,i2,i3 通过实验发现，一拍N画面一般符合规律——"中间两帧的帧差小于左右两侧" 及 diff(i0,i1) > diff(i1,i2) and diff(i2,i3) > diff(i1,i2)（识别一拍二）

一拍三则可看成多个一拍二进行识别

注: 帧差可以使用光流距离来代替,详细见flow.py

# 测试结果

