# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:30:38 2019

@author: tien
"""
'''
    NE
'''
from BGD import add1Col
import numpy as np

def NormalEquation(x, y):
    x = add1Col(x)
    y = y.flatten()
    w = np.linalg.inv((x.transpose().dot(x))).dot(x.transpose().dot(y))    
    return w