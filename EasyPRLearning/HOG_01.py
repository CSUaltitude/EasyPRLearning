#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:02:28 2018

@author: zhouwei
svm + hog Plate Recognize 
"""

import cv2 as cv
import numpy as np
import os 
import math 
#import matplotlib.pyplot as plt 
'''
from skimage import data,io
#scikit  learn  
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
'''
# hog feature GET 
'''
读入彩色图像，并转换为灰度值图像, 获得图像的宽和高。
采用Gamma校正法对输入图像进行颜色空间的标准化（归一化），
目的是调节图像的对比度，降低图像局部的阴影和光照变化所造成的影响，
同时可以抑制噪音。采用的gamma值为0.5。
'''
img00 = cv.imread("HOGRawImg/persion02.jpg",cv.IMREAD_GRAYSCALE)# graady IMG
print(img00.shape)
#RESIZE 
img01 = cv.resize(img00, (480,480), interpolation=cv.INTER_CUBIC)
print(img01.shape)
#normalize 
img02 = np.sqrt(img01/float(np.max(img01)))
img02 = img02*255
cv.imwrite("imgNormalize.jpg",img02)


'''

Sobel算子依然是一种过滤器，只是其是带有方向的。在OpenCV-Python中，使用Sobel的算子的函数原型如下：

dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])

函数返回其处理结果。

前四个是必须的参数：

    第一个参数是需要处理的图像；
    第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
    dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。

其后是可选的参数：

    dst不用解释了；
    ksize是Sobel算子的大小，必须为1、3、5、7。
    scale是缩放导数的比例常数，默认情况下没有伸缩系数；
    delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
    borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT
'''
'''
计算每个像素的梯度
'''
#print(img02.shape)
height, width = img02.shape
#gradient caculate
#sobel filter
gradient_values_x = cv.Sobel(img02, cv.CV_64F, 1, 0, ksize=5)
gradient_values_y = cv.Sobel(img02, cv.CV_64F, 0, 1, ksize=5)
gradient_magnitude = cv.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
#
gradient_angle = cv.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
print (gradient_magnitude.shape, gradient_angle.shape)
'''
#HOG 为每个细胞单元构建梯度方向直方图
each cell contents cell_size*cell_size pix
bin_size:将cell的梯度方向360度分成bin_size个方向块
'''
cell_size = 8
bin_size = 8
angle_unit = 360 / bin_size
gradient_magnitude = abs(gradient_magnitude)#梯度值
cell_gradient_vector = np.zeros((np.int(height / cell_size), np.int(width / cell_size), bin_size))#vectors number 
print(cell_gradient_vector.shape)

'''
将得到的每个cell的梯度方向直方图绘出，得到特征图
'''
def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0]*8  #[0,0,0,0,0,0,0,0]
    for i in range(cell_magnitude.shape[0]):
        for j in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[i][j]#梯度值
            gradient_angle = cell_angle[i][j]#angle value 
            min_angle = int(gradient_angle / angle_unit)%8# 8 is not necissary  角度所属的index
            max_angle = (min_angle + 1) % bin_size#紧邻的下一个角度区间index
            mod = gradient_angle % angle_unit#区间内的偏移量
            #幅值按照区间内的偏移量权重进行累加至直方图中
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers

#遍历每一个CELL的梯度 直方图，并保存 to cell_gradient_vector
for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        # get cell's magnitude and angle 
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                     j * cell_size:(j + 1) * cell_size]
        #print(cell_angle.max())
        #get cell's 的梯度方向直方图
        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
        
'''
'''

hog_image= np.zeros([height, width]) 
cell_gradient = cell_gradient_vector 
cell_width = cell_size / 2 
max_mag = np.array(cell_gradient).max() 
for x in range(cell_gradient.shape[0]): 
    for y in range(cell_gradient.shape[1]): 
        cell_grad = cell_gradient[x][y] 
        cell_grad /= max_mag 
        angle = 0 
        angle_gap = angle_unit 
        for magnitude in cell_grad: 
            angle_radian = math.radians(angle) 
            x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
            y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
            x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian)) 
            y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
            cv.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude))) 
            angle += angle_gap 

cv.imwrite("hog_img.jpg",hog_image)#将得到的每个cell的梯度方向直方图绘出，得到特征图

'''
统计Block的梯度信息  block contant serveral cells
'''
hog_vector = []
for i in range(cell_gradient_vector.shape[0] - 1):
    for j in range(cell_gradient_vector.shape[1] - 1):
        block_vector = []
        block_vector.extend(cell_gradient_vector[i][j])
        block_vector.extend(cell_gradient_vector[i][j + 1])
        block_vector.extend(cell_gradient_vector[i + 1][j])
        block_vector.extend(cell_gradient_vector[i + 1][j + 1])
        '''
        由于局部光照的变化以及前景-背景对比度的变化，
        使得梯度强度的变化范围非常大。这就需要对梯度强度做归一化。
        归一化能够进一步地对光照、阴影和边缘进行压缩。
        L2-norm 归一化方法：   v/sqrt(sum(x**x))
        lambda function :a = lambda arg1,arg1...:opration ; a is lambda funtion 's point 
        '''
        mag = lambda vector: math.sqrt(sum(i**i for i in vector))#lambda function define ,mag is a function 
        magnitude = mag(block_vector)#
        if magnitude != 0:
            normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
            block_vector = normalize(block_vector, magnitude)
        hog_vector.append(block_vector)
        
print(np.array(hog_vector).shape)





