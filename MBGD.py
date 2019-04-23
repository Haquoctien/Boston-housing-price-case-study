import numpy as np
from numba import jit

'''
    MBGD
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
def MiniBatch(x, y, learningRate, w):  
    
    yHat = h(x, w)
    
    deltaE = dE(yHat, y, x)
    
    wnext = w - learningRate*deltaE
    
    Enext = Er(x, y, w)
    return wnext, Enext

def MBGD(x, y, learningRate, w = None, epsilon= 0.001, size = 30):
    if w == None:
        w = np.zeros((x.shape[1]+1))
        
    x = add1Col(x)
    
    y = np.array(y).flatten()
    
    E = float('Inf')
    Enext =  0

    wnext = w
    cond = True
    a = 0
    while cond:
        for i in range(0,y.shape[0],size):
            wnext, Enext = MiniBatch(x[i:i+size], y[i:i+size], learningRate, wnext)
        if abs(Enext - E) < epsilon:
            E = Enext
            w = wnext
            cond = False
            break
        E = Enext
        w = wnext
        a = a + 1
    return w