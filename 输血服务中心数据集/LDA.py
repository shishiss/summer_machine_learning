import numpy as np
from matplotlib import pyplot as plt

#从txt中获取数据
def get_Data(filename):
	charactor = []
	tag = []
	with open(filename) as txtData:
		lines = txtData.readlines()
		for line in lines:
			line = line.strip().split(',')
			charactor.append([float(tk) for tk in line[0:-1]])
			tag.append(int(line[-1]))
		#print(charactor[0:5], tag[0:5])
	return np.array(charactor), np.array(tag)

#计算协方差矩阵
def Cal_cov_matrix(X):
	#print(X.shape)
	#print(X.mean(axis=0))
	#print((X - X.mean(axis=0))[0:2])
	cov_matrix = (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))
	return cov_matrix

def fit(X, y):
	#X.shape(748, 4), y.shape(748,)
	print(X.shape, y.shape)

	X = X.reshape(X.shape[0], -1)
	X0 = X[y == 0]
	X1 = X[y == 1]
	y = y.reshape(y.shape[0], -1)
	#print(X1[0:2], X2[0:2], y[0:5])

	#计算均值向量
	u0 = X0.mean(axis=0) #axis=0 是计算矩阵中每一列的均值(1是行)
	u1 = X1.mean(axis=0)
	#print(u1, u2)
	mean_diff = np.atleast_1d(u0 - u1)
	#print(mean_diff)
	mean_diff = mean_diff.reshape(X.shape[1], -1)
	#print(mean_diff)

	#计算两类的协方差矩阵
	S0 = Cal_cov_matrix(X0)
	S1 = Cal_cov_matrix(X1)
	#计算类内散度矩阵
	S_w = S0 + S1
	#计算w,其中w = S_w^{-1}(u0 - u1)
	w = np.linalg.pinv(S_w).dot(mean_diff)

	print('print projection direction w: \n', w)
	return w


def plot_lda(X, y, w):

    #得到投影后的坐标new_y
    X0 = X[y == 0]
    X1 = X[y == 1]
    projection_y0 = X0.dot(w)
    projection_y1 = X1.dot(w)
    y0_mean = projection_y0.mean()
    y1_mean = projection_y1.mean()
    
    print(projection_y0.max(), projection_y0.min(), projection_y1.max(), projection_y1.min())
    print(y0_mean, y1_mean)

    #print(new_y)
    plt.hlines(2, -8, 8)
    plt.hlines(1, -8, 8)
    #plt.vlines(1, 1, 20)
    plt.xlim(-8.5, 8.5)
    plt.ylim(0.5, 2.5)
    
    
    y0 = np.ones(np.shape(projection_y0))
    y1 = np.ones(np.shape(projection_y1))
    print(y0.shape, y1.shape)
    plt.plot(projection_y0 * 1000, y0, '^', ms=10)
    plt.plot(projection_y1 * 1000, 2 * y1, 'x', ms=10)
    
    plt.vlines(y0_mean * 1000, 0.5, 1, 'red', '--')
    plt.vlines(y1_mean * 1000, 0.5, 2, 'green', '--')
    plt.show()


if __name__ == '__main__':
	file_path = '/Users/yuqishi/Documents/machine_learning/data/输血服务中心数据集/blood_data.txt'
	#X,y都是numpy.ndarray型的
	X, y = get_Data(file_path)
	w = fit(X, y)
	plot_lda(X, y, w)


	