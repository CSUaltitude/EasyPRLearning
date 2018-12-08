#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:57:00 2018
using SIFT feature and bag of word module to classify data with SVM 
use sklearn lib
@author: zhouwei
"""
import cv2 
from sklearn.svm import LinearSVC
import numpy as np

img = cv2.imread("HOGRawImg/persion01.jpg",cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("HOGRawImg/persion0101.jpg",cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.xfeatures2d.SURF_create()

pos,des = sift.detectAndCompute(img,None)
pos1,des1 = sift.detectAndCompute(img1,None)
#get sift feature 
print(np.array(pos).shape)
print(np.array(des).shape)###des 
print(pos[0])
"""
13 sift对象会使用DoG检测关键点，对关键点周围的区域计算向量特征，检测并计算
14 返回 关键点pos和描述符des
15 关键点是点的列表
16 描述符是检测到的特征的局部区域图像列表
17 
18 关键点的属性：
19     pt: 点的x y坐标
20     size： 表示特征的直径
21     angle: 特征方向
22     response: 关键点的强度
23     octave: 特征所在金字塔层级
24         算法进行迭代的时候， 作为参数的图像尺寸和相邻像素会发生变化
25         octave属性表示检测到关键点所在的层级
26     ID： 检测到关键点的ID
27 
"""
'''
# 在图像上绘制关键点
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS表示对每个关键点画出圆圈和方向
'''
'''
img = cv2.drawKeypoints(image = img,outImage = img,keypoints = pos,
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color = (0,255,0))

cv2.imwrite("sift_keypoints.jpg",img)
'''

'''
KNN feature match # BFMatcher with default params
'''
bf = cv2.BFMatcher()
matches = bf.knnMatch(des, des1, k=2)
good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]
img3 = cv2.drawMatchesKnn(img, pos, img1, pos1, good, None, flags=2)

cv2.imwrite("sift_keypoints.jpg",img3)