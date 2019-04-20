# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:47:44 2019

@author: tien
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from numba import jit

# đọc dữ liệu
columns = ['CRIM','ZN','INDUS',
         'CHAS','NOX','RM',
         'AGE','DIS','RAD',
         'TAX','PTRATIO','B',
         'LSTAT','MEDV']

data = pd.read_csv('housing.data.txt',
                   delim_whitespace=True,
                   header = None, names = columns)
                   
# tách dữ liệu thành examples và labels tương ứng
examples = np.array(data[['CRIM','ZN','INDUS',
                'CHAS','NOX','RM',
                'AGE','DIS','RAD',
                'TAX','PTRATIO','B',
                'LSTAT']])

labels = np.array(data[['MEDV']])

# chuẩn hóa dữ liệu

xScaler = StandardScaler().fit(examples)
yScaler = StandardScaler().fit(labels)

examplesStd = xScaler.transform(examples)
labelsStd = yScaler.transform(labels)

# chia dữ liệu thành tập train và tập test
xTrain, xTest, yTrain, yTest = train_test_split(examplesStd, labelsStd, random_state = 42)

# xay dung model hoi quy tuyen tinh bang thu vien
model = LinearRegression()
model.fit(xTrain, yTrain)
wmodel = np.concatenate((model.intercept_, model.coef_.flatten()))

# hàm tính deltaE
@jit
def dE(yHat, y, x):
    m = x.shape[0]
    deltaE = np.full((x.shape[1]), 0.0, dtype='float')
    
    for i in range(x.shape[1]):
        xi = x[:,i]
        
        deltaE[i] = 2/m*np.sum((yHat - y)*xi)
    return deltaE

# hàm tính hàm giả thiết H
@jit
def h(x, w):
    return np.sum(x*w, axis = 1)

# hàm tính độ lỗi
@jit
def Er(x, y, w):
    m = x.shape[0]
    yHat = h(x, w)
    return (1/m)*np.sum((y-yHat)**2)

# hàm thêm cột 1 vào ma trận examples
@jit
def add1Col(x):
     m = x.shape[0]
     return np.hstack((np.full((m, 1), 1), x))

# hàm chạy batch gradient descent
@jit
def BGD(x, y, learningRate, w = None, epsilon= 0.001):
    
    if w == None:
        w = np.zeros((x.shape[1]+1))
        
    x = add1Col(x)
    
    y = np.array(y).flatten()
    
    E = float('Inf')
    Enext =  0
    wnext = np.array([])
    
    while True:
        yHat = h(x, w)
        
        deltaE = dE(yHat, y, x)
        
        wnext = w - learningRate*deltaE
        
        Enext = Er(x, y, wnext)
        
        if abs(E - Enext) < epsilon:
            E = Enext
            w = wnext
            break
        E = Enext
        w = wnext
    return w, E

# hàm dự đoán từ mô hình đã học được, có chuyển về scale dữ liệu ban đầu
@jit
def linearRegressionPredict(x, w, scaler=None):
    x = add1Col(x)
    yPred = np.sum(x*w, axis = 1)
    if scaler == None:
        return yPred
    else:
        return scaler.inverse_transform(yPred)


# tìm bộ trọng số và độ lỗi dựa trên tập train, pp BGD
w, ETrain = BGD(xTrain, yTrain, learningRate=0.05, epsilon = 0.00000000001)

xTe = add1Col(xTest)
yTe = yTest.flatten()

xTr = add1Col(xTrain)
yTr = yTrain.flatten()

# tìm độ lỗi dựa của w trên tập test
ETest = Er(xTe, yTe, w)
# tìm độ lỗi của wmodel từ tập train
EmodelTrain = Er(xTr, yTr, wmodel)
# tìm độ lỗi của wmodel trên tập test
EmodelTest = Er(xTe, yTe, wmodel)

# dự đoán giá nhà theo examples trong tập test, có scale về khoảng giá ban đầu
yPred = linearRegressionPredict(xTest, w, yScaler)


