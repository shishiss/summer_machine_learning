import numpy as np

#在测试集上用计算出的w值预测y，并计算均方误差
def predict(x_arr, y_arr, w):
    Y = np.mat(y_arr).T
    result = x_arr.dot(w) - Y
    
    return result.T.dot(result) / len(Y)