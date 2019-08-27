import numpy as np


def linear_regression(x_arr, y_arr, lamb = 0.2):
	Y = np.mat(y_arr).T
	X_T = x_arr.T
	#print(X_T.shape, x_arr.shape, Y.shape)
	#x_tx是X的转置乘以X的乘积
	x_tx =  X_T.dot(x_arr)

	w = np.linalg.inv(x_tx).dot(X_T).dot(Y)
	return w