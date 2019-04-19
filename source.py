# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:47:44 2019

@author: tien
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

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

