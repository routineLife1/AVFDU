import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread("000001031.png",0)
edges = cv2.Canny(im,100,200)
cv2.imwrite('test7.png',255-edges)