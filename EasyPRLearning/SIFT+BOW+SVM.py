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
import os 
def calcFeatVec(features,centers):#get eachimg feature vector 
    featVec = np.zeros((1,50))
    for i in range(0,features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi,(50,1)) - centers
        sqSum = (diffMat**2).sum(axis = 1)
        dist = sqSum**0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0] # index of the nearest center 
        featVec[0][idx] += 1	
    return featVec
    
        
        
img = cv2.imread("HOGRawImg/persion01.jpg",cv2.IMREAD_GRAYSCALE)

#img = cv2.imread("HOGRawImg/persion01.jpg",cv2.COLOR_BGR2GRAY)#

img1 = cv2.imread("HOGRawImg/persion0101.jpg",cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.xfeatures2d.SURF_create()

pos,des = sift.detectAndCompute(img,None)
pos1,des1 = sift.detectAndCompute(img1,None)
#get sift feature 
print(np.array(pos).shape)
print(np.array(des).shape)###des 
print(np.array(des1).shape)###des 
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

cv2.imwrite("sift_match.jpg",img3)

'''
BOW modulle 
利用SIFT算法，从每类图像中提取视觉词汇，将所有的视觉词汇集合
1 ---get sift of all imgs those include diffrent classification 
2---利用K-Means算法构造单词表
3---是利用单词表的中词汇表示图像
'''
desfinal = np.vstack((des,des1))
print(np.array(desfinal).shape)#sift feature combine  利用SIFT算法，从每类图像中提取视觉词汇，将所有的视觉词汇集合

'''
-利用K-Means算法构造单词表
'''
# 使用kmeans进行聚类分析，设置终止条件为执行10次迭代或者精确度epsilon=1.0
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
 
#flags = cv2.KMEANS_RANDOM_CENTERS
'''
cv2.kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
输入参数：

data:np.float32类型的数据，每个特征应该放在一列。

K：聚类的最终数目。

bestLabels：预设的分类标签，没有的话就设置为None。

criteria：终止迭代的条件，当条件满足时算法的迭代就终止。它应该是一个含有三个成员的元组。

attempts：重复试验kmeans算法次数，将会返回最好的一次结果。

flags：初始类中心选择，有两个选择：cv2.KMEANS_PP_CENTERS 和 cv2.KMEANS_RANDOM_CENTERS

输出参数：

compactness：紧密度，返回每个点到相应聚类中心距离的平方和。

labels：标志数组。

centers：有聚类中心组成的数组。

'''
# 运用kmeans
# 返回值有紧密度、标志和聚类中心。标志的多少与测试数据的多少是相同的
#---------------------------------------------------50 class num 
#compactness, labels, centers = cv2.kmeans(desfinal, 50, None, criteria, 10, flags)

'''
print(np.array(compactness).shape)
print(labels)
print(np.array(labels).shape)
print(np.array(centers).shape)
'''
##get 

#feature0 = np.zeros((des.shape[0],50))
#feature0 = []
#ft = calcFeatVec(des[0],centers)
#print(ft.shape)
'''
get feature list of img0's sift feature 
'''
###3---是利用单词表的中词汇表示图像
'''
for i in range(des.shape[0]):
    ft = calcFeatVec(des[i],centers)
    list(feature0).append(ft)# list 
'''
'''
print(des.shape[0])
print(feature0.shape)
--------5863 180hz 
'''
def GetImgSiftFeature(filepath):
    listfile = os.listdir(filepath)
    desvec = np.zeros((1,128))
    for i in range(len(listfile)):
        path = os.path.join(str(filepath),str(listfile[i]))
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        pos,des = sift.detectAndCompute(img,None)
        if i == 0:
            desvec = des
        else:  
            desvec = np.vstack((desvec,des))
    return desvec

def GetSiftKmeanCenters(des,k):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(np.float32(des), k, None, criteria, 10, flags)
    return centers
'''
def GetImgSiftVecOfBOW(imgname,centers):
    img = cv2.imread(imgname,cv2.IMREAD_GRAYSCALE)
    pos,des = sift.detectAndCompute(img,None)
    feature = np.zeros((1,50))
    print(des.shape)
    for i in range(des.shape[0]):
        ft = calcFeatVec(des[i],centers)
        if i == 0 :
            feature = ft
        else:
            feature = np .vstack((feature,ft))
    return feature

def GetFilePathImgSiftVec(filepath):
    listfile = os.listdir(filepath)
    filefeature  = np.zeros((1,50))
    for i in range(len(listfile)):
        path = os.path.join(str(filepath),str(listfile[i]))
        ft = GetImgSiftVecOfBOW(path,centors)
        if i == 0:
            filefeature = ft
        else: 
            filefeature = np.vstack((filefeature,ft))
    return filefeature
'''
def GetDesBowVec(des,centers):
    desbowvec = np.zeros((1,50))
    for i in range(des.shape[0]):
        vec = calcFeatVec(des[i],centers)
        if i == 0:
            desbowvec = vec
        else:
            desbowvec = np.vstack((desbowvec,vec))
    return desbowvec


'''
#利用SIFT算法，从每类图像中提取视觉词汇，将所有的视觉词汇集合 GetImgSiftFeature
'''
des0 = GetImgSiftFeature("HOGRawImg") 
'''
#利用K-Means算法构造单词表 GetSiftKmeanCenters
'''
centors = GetSiftKmeanCenters(des0,50)
'''
#是利用单词表的中词汇表示图像 GetDesBowVec
'''
filefeature = GetDesBowVec(des0,centors)

print(des0.shape)
print(centors.shape)
print(np.array(filefeature).shape)









