#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:44:01 2018
#SVM FOR easyPR
#very simple SVM moudle 
#just use all pixerlvalue of each img
@author: zhouwei
"""
import numpy as np
import cv2 as cv
import os 

#load data 
def LoadData(fileDir,typex):
    listfile = os.listdir(fileDir)
    
    if not os.path.exists(fileDir):
        print("no that file")
        return
    traindata = []
    labeldata= []
    for i in range(len(listfile)):
        path = os.path.join(str(fileDir),str(listfile[i]))
        imgtmp = cv.imread(path,0)
        imgtmp = imgtmp.flatten()#mat 2 (1,n)
        imgtmp = np.array(list(imgtmp),dtype = np.float32)#svm data formate
        traindata.append(imgtmp)
        labeldata.append(np.int32(typex))

    return traindata,labeldata
print("zhouwei first SVM moudle test:)")
print("load train data set")
#pos data
fileDir = "../resources/train/has/train"
trainlist,trainlabel = LoadData(fileDir,1)
#neg data 
fileDir ="../resources/train/no/train"
trainlist01,trainlabel01 = LoadData(fileDir,0)

#train data set proc --format for SVM 
trainlist = np.vstack((trainlist,trainlist01))#train data set
trainlabel.extend(trainlabel01)#train label set

print("load test data set")
#pos data
fileDir = "../resources/train/has/test/"
testlist,testlabel = LoadData(fileDir,1)
#neg data 
fileDir ="../resources/train/no/test/"
testlist01,testlabel01 = LoadData(fileDir,0)
#test set format for SVM
testlist = np.vstack((testlist,testlist01))
testlabel.extend(testlabel01)

#create  SVM module
svm = cv.ml.SVM_create()
#SVM CONGIFG 
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,1000,1E-2)
svm.setTermCriteria(criteria)
svm.setGamma(1)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setP(0.0)
svm.setC(1)

print("train SVM moudle")
svm.train(np.array(trainlist),cv.ml.ROW_SAMPLE,np.array(trainlabel))
print("save moudle")
svm.save("mySvmMoudle01.mat")

print("testing ...")
m = np.array(testlist).shape[0]# test set size 
n = 0
for i in range(m):
    testmat = np.mat(testlist[i])# predic format is 1XN ,so format a mat
    p = svm.predict(testmat)#p 
    #print(p)#debug for p contents
    if p[1][0][0] == testlabel[i]:
        n += 1
accurace = n/np.float32(m)

print("accurace is :%f"%(accurace))


        
    



