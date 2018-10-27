#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:44:01 2018
#SVM FOR easyPR
@author: zhouwei
"""
import numpy as np
import cv2 as cv
import os 

#load data 
def LoadData(fileDir,typex):
    listfile = os.listdir(fileDir)
    traindata = []
    labeldata= []
    for i in range(len(listfile)):
        path = os.path.join(str(fileDir),str(listfile[i]))
        imgtmp = cv.imread(path,0)
        imgtmp = imgtmp.flatten()#mat 2 (1,n)
        imgtmp = np.array(list(imgtmp),dtype = np.float32)#svm data formate
        traindata.append(imgtmp)
        labeldata.append(np.int32(typex))
    if os.path.exists(fileDir):
        print("file exsit")
    else:
        print("no that file")
    return traindata,labeldata

print("load data set")
#pos data
fileDir = ""
trainlist,trainlabel = LoadData(fileDir,1)
#neg data 
fileDir =""
trainlist01,trainlabel01 = LoadData(fileDir,0)



