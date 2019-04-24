import numpy as np
from numba import jit
'''
SGD
'''
# hàm tính deltaE
@jit
def dE(yHat, y, x):
    deltaE = np.full((x.shape[0]), 0.0, dtype='float')
    for i in range(x.shape[0]):
        xi = x[i]
        deltaE[i] = 2*(yHat-y)*xi
    return deltaE

# hàm thêm cột 1 vào ma trận examples
@jit
def add1Col(x):
     m = x.shape[0]
     return np.hstack((np.full((m, 1), 1), x))

# hàm chạy batch gradient descent
@jit
def SGD(X, Y, learningRate = 0.0005, w = None, epsilon= 1e-5, numberOfIterations = 100):
    
    if w == None:
        w = np.zeros((X.shape[1]+1))
        
    X = add1Col(X)
    
    Y = np.array(Y).flatten()
    
    E = float('Inf')
    Enext =  0
    wnext = np.array([])
    
    done = False
    i = 0
    while not done and i < numberOfIterations:
        for Xi, Yi in zip(X, Y):
            yHat = np.sum(Xi*w)
            
            deltaE = dE(yHat, Yi, Xi)
            
            wnext = w - learningRate*deltaE
            
            Enext = (Yi-yHat)**2
            
            if abs(E - Enext) < epsilon:
                E = Enext
                w = wnext
                done = True
                break
            E = Enext
            w = wnext
        i += 1
    return w