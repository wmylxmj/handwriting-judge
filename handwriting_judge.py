# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 19:57:50 2018

@author: wmy
"""

import numpy
import operator
from os import listdir
import PIL.Image
import matplotlib.pyplot as plt

def kNNClassify(dataset,trainvalue,trainlabels,k):
    #shape为numpy模块中的方法 shape[0]为矩阵第二维的长度
    traindatasize=trainvalue.shape[0]
    #计算各个维度的差值并储存在向量diffmat中
    diffmat=numpy.tile(dataset,(traindatasize,1))-trainvalue
    #计算误差的平方
    sqdiffmat=diffmat**2
    #计算向量间的欧式距离
    sqdistances=sqdiffmat.sum(axis=1)
    distances=sqdistances**0.5
    #排序
    sorteddistindicies=distances.argsort()
    classcount={}
    for i in range(k):
        selectedlabel=trainlabels[sorteddistindicies[i]]
        classcount[selectedlabel]=classcount.get(selectedlabel,0)+1
    sortedclasscount=sorted(classcount.items(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]
        
def ImageBinaryzationToVector(filename):
    #转换成灰度图像
    image=PIL.Image.open(filename).convert('L')
    imagevector=numpy.array(image)
    height,width=imagevector.shape
    for h in range(height):
        for w in range(width):
            if imagevector[h,w]<=128:
                imagevector[h,w]=0
            else:
                imagevector[h,w]=1
    return imagevector

def GrayscaleImageShow(dataarray):
    plt.imshow(dataarray,cmap='gray')
    plt.axis('off')
    plt.show()
    return 0

def MatrixToOneDimension(dataarray):
    returnvector=numpy.array(dataarray)
    returnvector=returnvector.flatten()
    return returnvector

trainfilename=['0_1.jpg','1_1.jpg','2_1.jpg','3_1.jpg',
               '4_1.jpg','5_1.jpg','6_1.jpg','7_1.jpg',
               '8_1.jpg','9_1.jpg']

def TrainTheMachine(trainfilename):
    traindata=[]
    trainlabels=[]
    for i in range(len(trainfilename)):
        image=ImageBinaryzationToVector(trainfilename[i])
        vector=MatrixToOneDimension(image)
        traindata.append(vector)
        trainlabels.append(trainfilename[i][0])
    traindata=numpy.array(traindata)
    return traindata,trainlabels

traindata,trainlabels=TrainTheMachine(trainfilename)

#print(traindata)
print('i have learned '+str(trainlabels))
    
testfilename=input('please input the test file name which in the catalogue:')  
 
testimage=ImageBinaryzationToVector(testfilename)
testarray=MatrixToOneDimension(testimage)

answer=kNNClassify(testarray,traindata,trainlabels,3)
print('your handwriting is'+' '+str(answer))
