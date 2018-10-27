#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:52:26 2018

@author: zhouwei
SVM classify PR 

"""
import cv2 
import math 
import numpy as np
import os
from itertools import chain
from sklearn import svm

trainpospath = "EasyPR-master/resources/train/has/train/"
trainNegpath = "EasyPR-master/resources/train/no/train/"
TestposPath = "EasyPR-master/resources/train/has/test/"
TestnegPath = "EasyPR-master/resources/train/no/test/"

def img2vector(img):
    size = 36*136*3
    vec = np.zeros((1,size))
    for i in range(36):
        for j in range(136):
            for k in range(3):
                vec[0,i*j*k]=(img[i][j][k])
    return vec
    
img1 = cv2.imread("EasyPR-master/resources/train/has/train/plate_477.jpg")
print(img1.shape)
#imgv = img1.reshape(14688,order="f")
imgv = img1.reshape(1,-1)
print(imgv.shape)
print((imgv))
'''
imga = img2vector(img1)
imga = np.array(imga)
imga=imga.transpose()
print(imga.shape)
#imga = list(chain(*img1))
print(imga)
'''
def GetImglist(filepath):
    if os.path.exists(filepath):
        print("has this file ")
    else:
        print("no this file")
        return 0
    
    filelist = os.listdir(filepath)
    listlen = len(filelist)
    print((len(filelist)))
    trainfile =[]
    imglist = []
    label =[]
    for i in range(listlen):
            trainfile.append(filelist[i])
            
            imgtmp = cv2.imread(filepath+filelist[i])
            imgtmp = imgtmp.reshape(1,-1)
            #print(imgtmp)
            imglist.append((imgtmp))
            #imglist = np.array(imglist)
            label.append(1)
            
    
    #print(trainfile)
    return imglist,label

trainlist,trainlabel = GetImglist(trainpospath)
#print((trainlist))
print(np.array(trainlabel).shape)
'''
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(np.array(trainlist), cv2.ml.ROW_SAMPLE, np.array(trainlabel))
'''
##开始训练
clf=svm.SVC()  ##默认参数：kernel='rbf'
clf.fit(trainlist,trainlabel)
