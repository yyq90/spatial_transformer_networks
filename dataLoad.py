import os
from os.path import join
import cv2
import numpy as np
from collections import defaultdict


def dictload(dirpath = 'data/train.txt'):
    f = open(dirpath, "r")
    labelDict = dict()
    validateDict = defaultdict(list)

    validateList = list()
    while True:
        line = f.readline()
        if line:
            pass  # do something here
            name = line.split(' ')[0]
            label = line.split(' ')[1].strip()
            key,ext = os.path.splitext(name)
            labelDict[key]=label
            validateDict[label].append(key)
        else:
            break
    f.close()
    for k, v in validateDict.items():
        counter= 0
        for w in v:
            counter=counter+1
            if counter<10:
                validateList.append(w)

    return labelDict,validateList
def testdictload(dirpath = 'data/train.txt'):
    f = open(dirpath, "r")
    picDict = list()


    while True:
        line = f.readline()
        if line:
            pass  # do something here
            # name = line.split(' ')[0]
            # key,ext = os.path.splitext(name)
            picDict.append(line.strip())
        else:
            break
    f.close()

    return picDict

def dataload(img_w=300,img_h=300,val_ratio = 0.95,gray=0):
    # load y dict
    # labelDict,validateList= dictload("c:/tempProjects/keras-resnet/data/train.txt")
    labelDict,validateList= dictload("d:/git/keras-resnet/data/train.txt")


    # img_dirpath = "c:/tempProjects/keras-resnet/data/train"
    img_dirpath = "d:/git/keras-resnet/data/train"
    # X=[]
    # y=[]
    X_train = []
    y_train=[]
    X_val=[]
    y_val=[]
    for filename in os.listdir(img_dirpath):
        name, ext = os.path.splitext(filename)
        if ext in ['.jpg']:
            img_filepath = join(img_dirpath, filename)
            img = cv2.imread(img_filepath)
            if gray==1:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (img_w, img_h))
            img = img.astype(np.float32)

            # img /= 255
            # X.append(img)
            # y.append(labelDict[name])
            X_train.append(img)
            y_train.append(labelDict[name])
            if name in validateList:
                X_val.append(img)
                y_val.append(labelDict[name])


    # X,y=np.asarray(X), np.asarray(y)
    # X =X.astype(np.float32)
    # y =y.astype(np.float32)
    #train validate

    # trainLen = int(len(X)*0.95)
    # valLen = len(X)-trainLen
    # X_train=[]
    # y_train=[]
    # X_val=[]
    # y_val=[]

    # for index,value in enumerate(X):
    #
    #     X_train.append(value)
    #     y_train.append(y[index])
    #
    #
    #     if index<trainLen:
    #         X_train.append(value)
    #         y_train.append(y[index])
    #
    #     else:
    #         #########
    #         X_train.append(value)
    #         y_train.append(y[index])
    #         ##############
    #         X_val.append(value)
    #         y_val.append(y[index])



    X_train,y_train,X_val,y_val = np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.int32)
    y_val = np.reshape(y_val,(len(y_val),1))
    y_train = np.reshape(y_train,(len(y_train),1))


    return X_train,y_train-1,X_val,y_val-1
def testLoad(img_w=300,img_h=300,val_ratio = 0.95):
    # load y dict
    # picDict= testdictload("c:/tempProjects/keras-resnet/data/test.txt")
    picDict = list()

    # img_dirpath = "c:/tempProjects/keras-resnet/data/test"
    img_dirpath = "d:/git/keras-resnet/data/test"
    # X=[]
    # y=[]
    X_test = []

    for filename in os.listdir(img_dirpath):
        img_filepath = join(img_dirpath, filename)
        img = cv2.imread(img_filepath)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_w, img_h))
        img = img.astype(np.float32)
            # img /= 255
            # X.append(img)
            # y.append(labelDict[name])
        X_test.append(img)
        picDict.append(filename)
    X_test = np.asarray(X_test)
    X_test = X_test.astype(np.float32)
    return X_test,picDict
# X_train,y_train,X_val,y_val=dataload(50,50)
# print()