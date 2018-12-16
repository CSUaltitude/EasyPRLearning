from __future__ import division, print_function, absolute_import

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:03:17 2018

@author: zhouwei
"""
'''
define a dnn net with tflern lib
'''
import numpy as np
import tflearn 
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
'''
use blow 2 lines codeto resolve issu:
feed_dict[net_inputs[i]] = x

IndexError: list index out of range

'''
import tensorflow as tf
tf.reset_default_graph()

'''
# Download the Titanic dataset
'''
#from tflearn.datasets import titanic

#titanic.download_dataset('titanic_dataset.csv')

'''
load titanic_dataset.csv
'''
from tflearn.data_utils import load_csv
#load_csv()
#data,label = load_csv("titanic_dataset.csv")
'''
#label return binary must set  categorical_labels=True, n_classes=2
'''
'''
data,label = load_csv('titanic_dataset.csv', target_column=0,
                                categorical_labels=True, n_classes=2)
'''
data,label = load_csv(filepath = "titanic_dataset.csv",target_column=0,
                      categorical_labels=True,
                      columns_to_ignore=[2,7],n_classes=2)

'''
# Preprocessing functio7
def preprocess(data, columns_to_ignore):
        # Sort by descending id and delete columns
        for id in sorted(columns_to_ignore, reverse=True):
                [r.pop(id) for r in data]
        for i in range(len(data)):
                # Converting 'sex' field to float (id is 1 after removing labels column)
                data[i][1] = 1. if data[i][1] == 'female' else 0.
        return np.array(data, dtype=np.float32)
 
# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]
# Preprocess data
data = preprocess(data, to_ignore)
'''
def processData(inputdata):
    for i in range(len(data)):
        if inputdata[i][1]=="female":
            inputdata[i][1] = 0
        else:
            inputdata[i][1] = 1
        if len(inputdata[i]) < 6:
            print("data loss ")
    return inputdata

data = processData(data)
#data = np.float32(data[:10])
#label =np.float32(label[:10])
'''
data = np.array(data)
label = np.array(label)

print(data.shape)
print(type(data))
print(label.shape)
print(type(label))
#print(label)
'''


#define a netural net
net = input_data(shape=[None,6])#same shape as data 
net = fully_connected(net,32)#
net = fully_connected(net,32)
net = fully_connected(net,16)
net = fully_connected(net,2,activation="softmax")#output 
#
net = regression(net)


'''
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 64,activation='relu')
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)
'''
#define nn 

traindata = data[:1000]
trainlabel = label[:1000]
testdata = data[1001:]
testlabel = label[1001:]
model = tflearn.DNN(net)
'''
model.fit(traindata, trainlabel, n_epoch=10, batch_size=16, show_metric=True)
model.save('titanic_model.tflearn')
'''

model.load("titanic_model.tflearn")


#pred = model.predict([testdata[0]])
#print(pred)


cnt = 0.0
result = []
for i in range(len(testdata)):
    pred = model.predict([testdata[i]])
    if pred[0][1] > 0.5:
        result.append(1)
    else:
        result.append(0)
    if result[i] == testlabel[i][1]:
        cnt+=1
accurace = np.float32(cnt)/len(testdata)

print("accurace is ",accurace)





