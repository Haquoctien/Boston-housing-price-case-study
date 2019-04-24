import numpy as np
from numba import jit

'''
    BGD
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
