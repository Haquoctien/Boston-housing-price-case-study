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
wLib = np.concatenate((model.intercept_, model.coef_.flatten()))
yPredLib = model.predict(xTest)
ETestLib = mean_squared_error(yScaler.inverse_transform(yPredLib), yScaler.inverse_transform(yTest))
ETrainLib = mean_squared_error(yScaler.inverse_transform(model.predict(xTrain)), yScaler.inverse_transform(yTrain)) 

# hàm dự đoán từ mô hình đã học được, có chuyển về scale dữ liệu ban đầu
@jit
def linearRegressionPredict(x, w, scaler=None):
    x = add1Col(x)
    yPred = np.sum(x*w, axis = 1)
    if scaler == None:
        return yPred
    else:
        return scaler.inverse_transform(yPred)
   
# hàm vẽ biểu đồ thể hiện độ chính xác của mô hình tìm được trên tập train và tập test
def plot(xTrain, yTrain, xTest, yTest, w, scaler, y_max, title, outputName, xlabel = 'y', ylabel = 'y_pred'):
    plt.scatter(yScaler.inverse_transform(yTest), linearRegressionPredict(xTest, w, yScaler), s=1, c='blue', marker='o', label = 'Test set')
    plt.scatter(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, w, yScaler), s=1, c='red', marker='o', label = 'Training set')
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.axis([0,y_max,0,y_max])
    plt.title(title)
    plt.plot([0,y_max],[0,y_max],'g-')
    plt.savefig(outputName)
    plt.show()

'''
    BGD
'''
from BGD import BGD, add1Col
# tìm bộ trọng số và độ lỗi dựa trên tập train
wBGD = BGD(xTrain, yTrain, learningRate=0.05, epsilon = 1e-15)

# dự đoán giá nhà theo examples trong tập test, có scale về khoảng giá ban đầu
yPredBGD = linearRegressionPredict(xTest, wBGD, yScaler)

# tìm độ lỗi với w tìm được trên tập test và tập train
ETestBGD = mean_squared_error(yScaler.inverse_transform(yTest), yPredBGD)
ETrainBGD = mean_squared_error(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, wBGD, yScaler))

print("Kết quả dự đoán giá nhà của 10 examples đầu trong tập test: ", yPredBGD[:10])
print("E test:", ETestBGD)
print("E train:", ETrainBGD)

plot(xTrain, yTrain, xTest, yTest, wBGD, yScaler, 50 ,'Batch Gradient Descent', 'BGD.png')

'''
    SGD
'''
from SGD import SGD
# tìm bộ trọng số và độ lỗi dựa trên tập train
wSGD = SGD(xTrain, yTrain, learningRate=0.00005, epsilon = 1e-15, numberOfIterations=1000)

# dự đoán giá nhà theo examples trong tập test, có scale về khoảng giá ban đầu
yPredSGD = linearRegressionPredict(xTest, wSGD, yScaler)

# tìm độ lỗi với w tìm được trên tập test và tập train
ETestSGD = mean_squared_error(yScaler.inverse_transform(yTest), yPredSGD)
ETrainSGD = mean_squared_error(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, wSGD, yScaler))

print("Kết quả dự đoán giá nhà của 10 examples đầu trong tập test: ", yPredSGD[:10])
print("E test:", ETestSGD)
print("E train:", ETrainSGD)
plot(xTrain, yTrain, xTest, yTest, wSGD, yScaler, 50 ,'Stochastic Gradient Descent', 'SGD.png')

'''
    MBGD
'''
from MBGD import MBGD
# tìm bộ trọng số và độ lỗi dựa trên tập train
wMBGD = MBGD(xTrain, yTrain, learningRate=0.005, epsilon = 1e-16)

# dự đoán giá nhà theo examples trong tập test, có scale về khoảng giá ban đầu
yPredMBGD = linearRegressionPredict(xTest, wMBGD, yScaler)

# tìm độ lỗi với w tìm được trên tập test và tập train
ETestMBGD = mean_squared_error(yScaler.inverse_transform(yTest), yPredMBGD)
ETrainMBGD = mean_squared_error(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, wMBGD, yScaler))

print("Kết quả dự đoán giá nhà của 10 examples đầu trong tập test: ", yPredMBGD[:10])
print("E test:", ETestMBGD)
print("E train:", ETrainMBGD)
plot(xTrain, yTrain, xTest, yTest, wMBGD, yScaler, 50 ,'Minibatch Gradient Descent', 'MBGD.png')

'''
    XÂY DỰNG MODEL BẰNG PP NORMAL EQUATION
'''
from NE import NormalEquation
# tìm bộ trọng số và độ lỗi dựa trên tập train
wNormal = NormalEquation(xTrain, yTrain)

# dự đoán giá nhà theo examples trong tập test, có scale về khoảng giá ban đầu
yPredNormal = linearRegressionPredict(xTest, wNormal, yScaler)

# tìm độ lỗi với w tìm được trên tập test và tập train
ETestNormal = mean_squared_error(yScaler.inverse_transform(yTest), yPredNormal)
ETrainNormal = mean_squared_error(yScaler.inverse_transform(yTrain), linearRegressionPredict(xTrain, wNormal, yScaler))

print("Kết quả dự đoán giá nhà của 10 examples đầu trong tập test: ", yPredNormal[:10])
print("E test:", ETestNormal)
print("E train:", ETrainNormal)

plot(xTrain, yTrain, xTest, yTest, wNormal, yScaler, 50 ,'Normal Equation', 'NE.png')