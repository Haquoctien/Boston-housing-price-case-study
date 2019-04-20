# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:47:44 2019

@author: tien
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from numba import jit

# đọc dữ liệu
columns = ['CRIM','ZN','INDUS',
         'CHAS','NOX','RM',
         'AGE','DIS','RAD',
         'TAX','PTRATIO','B',
         'LSTAT','MEDV']

rawData = pd.read_csv('housing.data.txt',
                   delim_whitespace=True,
                   header = None, names = columns)

data = pd.DataFrame(normalize(rawData))
data.columns = columns
                   
# tách dữ liệu thành examples và labels tương ứng
examples = data[['CRIM','ZN','INDUS',
                'CHAS','NOX','RM',
                'AGE','DIS','RAD',
                'TAX','PTRATIO','B',
                'LSTAT']]
labels = data[['MEDV']]

# chia dữ liệu thành tập train và tập test
xTrain, xTest, yTrain, yTest = train_test_split(examples, labels, random_state = 42)

#
model = LinearRegression()
model.fit(xTrain, yTrain)
@jit
def dE(yHat, y, x):
    m = x.shape[0]
    deltaE = np.full((x.shape[1]), 0, dtype='float')
    
    for i in range(x.shape[1]):
        xi = x[:,i]
        
        deltaE[i] = 2/m*np.sum((yHat - y)*xi)
    return deltaE
@jit
def h(x, w):
    return np.sum(x*w, axis = 1)

def Er(x, y, w):
    x = np.array(x)
    y = np.array(y)
    m = x.shape[0]
    yHat = h(x, w)
    return (1/m)*np.sum((y-yHat)**2)
 
def xP(x):
     m = x.shape[0]
     return np.hstack((np.full((m, 1), 1), x))
    
@jit
def BGD(x, y, learningRate, w = None, epsilon= 0.001):
    
    if w == None:
        w = np.zeros((x.shape[1]+1))
        
    x = xP(x)
    
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

x = xP(xTest)
y = np.array(yTest).flatten()
wmodel = np.concatenate((model.intercept_, model.coef_.flatten()))

w, E = BGD(xTrain, yTrain, learningRate=0.075, epsilon = 0.000000000001)

Emodel = Er(x, y, wmodel)
Emy = Er(x, y, w)
E-Emodel