# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:47:44 2019

@author: tien
"""

import numpy as np
import pandas as pd
from numba import jit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

'''
    TIỀN XỬ LÍ
'''
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

'''
    XÂY DỰNG MODEL BẰNG THƯ VIỆN
'''
# xây dựng model hồi qui tuyến tính bằng thư viện sklearn
model = LinearRegression()
model.fit(xTrain, yTrain)
wmodel = np.concatenate((model.intercept_, model.coef_.flatten()))
yPredModel = model.predict(xTest)
ETestModel = mean_squared_error(yScaler.inverse_transform(yPredModel), yScaler.inverse_transform(yTest))
ETrainModel = mean_squared_error(yScaler.inverse_transform(model.predict(xTrain)), yScaler.inverse_transform(yTrain)) 

'''
    XÂY DỰNG MODEL BẰNG CÁC HÀM TỰ CODE
'''
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
    return w

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
w = BGD(xTrain, yTrain, learningRate=0.05, epsilon = 1e-16)

# dự đoán giá nhà theo examples trong tập test, có scale về khoảng giá ban đầu
yPred = linearRegressionPredict(xTest, w, yScaler)

# tìm độ lỗi với w tìm được trên tập test và tập train
ETest = mean_squared_error(yScaler.inverse_transform(yTest), yPred)
ETrain = mean_squared_error(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, w, yScaler))

print("E test:", ETest)
print("E train:", ETrain)

plt.scatter(yScaler.inverse_transform(yTest), linearRegressionPredict(xTest, w, yScaler), s=1, c='blue', marker='o', label = 'Test set')
plt.scatter(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, w, yScaler), s=1, c='red', marker='o', label = 'Training set')
plt.legend()
plt.ylabel('y_pred')
plt.xlabel('y')
plt.axis([0,50,0,50])
plt.title('Batch Gradient Descent')
plt.plot([0,50],[0,50],'g-')
plt.savefig('BGD.png')
plt.show()

'''
    XÂY DỰNG MODEL BẰNG PP NORMAL EQUATION
'''
# normal equations
def NormalEquations(x, y):
    x = add1Col(x)
    y = y.flatten()
    w = np.linalg.inv((x.transpose().dot(x))).dot(x.transpose().dot(y))    
    return w
    
wNormal = NormalEquations(xTrain, yTrain)
ETestNormal = mean_squared_error(yScaler.inverse_transform(yTest), linearRegressionPredict(xTest, wNormal, yScaler))
ETrainNormal = mean_squared_error(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, wNormal, yScaler))
print("E test:", ETestNormal)
print("E train:", ETrainNormal)


plt.scatter(yScaler.inverse_transform(yTest), linearRegressionPredict(xTest, wNormal, yScaler), s=1, c='blue', marker='o', label = 'Test set')
plt.scatter(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, wNormal, yScaler), s=1, c='red', marker='o', label = 'Training set')
plt.legend()
plt.ylabel('y_pred')
plt.xlabel('y')
plt.axis([0,50,0,50])
plt.title('Normal Equations')
plt.plot([0,50],[0,50],'g-')
plt.savefig('NormEqua.png')
plt.show()